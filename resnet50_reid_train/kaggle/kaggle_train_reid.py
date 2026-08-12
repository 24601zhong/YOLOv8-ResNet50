"""
============================================================
Kaggle 版: ReID V2 联合训练脚本
适配 Kaggle 输入/输出路径

用法:
  python kaggle_train_reid.py --num_epochs 120

依赖 (需作为 Kaggle Dataset 上传):
  - model.py
  - model_v2.py
  - train_reid.py
  - test_reid.py
============================================================
"""

import os
import sys
import time
import math
import random
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============================================================
# [*] Kaggle 路径配置
# ============================================================
# 代码文件所在目录 (上传为 Kaggle Dataset)
CODE_CANDIDATES = [
    Path("/kaggle/input/datasets/lsrhky11/reid-training-code"),
    Path("/kaggle/input/reid-training-code"),
]
CODE_DIR = None
for cand in CODE_CANDIDATES:
    if cand.exists():
        CODE_DIR = cand
        break
if CODE_DIR is None:
    CODE_DIR = Path("/kaggle/working")
print(f"[代码] CODE_DIR = {CODE_DIR}")

sys.path.insert(0, str(CODE_DIR))

from model_v2 import create_model_v2, CombinedReIDLoss
from train_reid import ReIDDataset, get_train_transforms, get_test_transforms
from test_reid import ReIDTestDataset

# ============================================================
# [*] 数据集路径 (Kaggle)
# ============================================================
DEFAULT_MARKET1501_DIR = "/kaggle/input/datasets/rayiooo/reid_market-1501"
DEFAULT_MOT17_DIR = "/kaggle/working/mot17_reid_clean"        # 预处理后
DEFAULT_OUTPUT_DIR = "/kaggle/working/reid_output"


# ============================================================
# 联合数据集
# ============================================================
class CombinedReIDDataset(Dataset):
    """合并 Market-1501 + MOT17，统一 PID 索引"""

    def __init__(self, data_configs, transform=None):
        self.data_configs = data_configs
        self.transform = transform
        self.images = []
        self.labels = []
        self.sources = []

        total_ids = 0
        for cfg in data_configs:
            ds = ReIDDataset(
                str(cfg["path"]),
                transform=None,
                is_train=cfg["is_train"],
            )
            offset = cfg["pid_offset"]
            for img_path, label in zip(ds.images, ds.labels):
                self.images.append(img_path)
                self.labels.append(label + offset)
                self.sources.append(str(cfg["path"]))
            n_ids = ds.num_classes
            print(f"  [{Path(cfg['path']).name}] {len(ds.images)} images, "
                  f"{n_ids} IDs -> offset {offset} (indices {offset}..{offset + n_ids - 1})")
            total_ids += n_ids

        self.num_classes = total_ids
        print(f"  [Combined] {len(self.images)} images, {self.num_classes} unified IDs")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (128, 256), color=(128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, label, str(img_path)


# ============================================================
# PK 采样器
# ============================================================
class RandomIdentitySampler(Sampler):
    """P 个身份 × K 张图 = batch_size"""

    def __init__(self, dataset, num_instances=4):
        self.dataset = dataset
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for idx, label in enumerate(dataset.labels):
            self.index_dic[label].append(idx)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = []
        pids_shuffled = self.pids.copy()
        random.shuffle(pids_shuffled)
        for pid in pids_shuffled:
            pid_indices = self.index_dic[pid]
            if len(pid_indices) >= self.num_instances:
                sampled = random.sample(pid_indices, self.num_instances)
            else:
                sampled = random.choices(pid_indices, k=self.num_instances)
            indices.extend(sampled)
        return iter(indices)

    def __len__(self):
        return len(self.pids) * self.num_instances


# ============================================================
# Warmup 调度器
# ============================================================
class GradualWarmupScheduler:
    """线性预热 + 余弦退火"""

    def __init__(self, optimizer, warmup_epochs, after_scheduler, warmup_factor=1e-3):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.warmup_factor = warmup_factor
        self.current_epoch = 0
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            alpha = float(self.current_epoch + 1) / float(max(1, self.warmup_epochs))
            factor = self.warmup_factor * (1.0 - alpha) + alpha
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = base_lr * factor
        else:
            self.after_scheduler.step()
        self.current_epoch += 1

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]

    def state_dict(self):
        return {
            'current_epoch': self.current_epoch,
            'after_scheduler': self.after_scheduler.state_dict(),
        }

    def load_state_dict(self, state):
        self.current_epoch = state['current_epoch']
        self.after_scheduler.load_state_dict(state['after_scheduler'])


# ============================================================
# 距离矩阵
# ============================================================
def compute_distance_matrix(query_feats, gallery_feats, metric="euclidean"):
    m, n = query_feats.size(0), gallery_feats.size(0)
    distmat = (
        torch.pow(query_feats, 2).sum(dim=1, keepdim=True).expand(m, n)
        + torch.pow(gallery_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    distmat = distmat - 2 * torch.mm(query_feats, gallery_feats.t())
    distmat = distmat.clamp(min=1e-12).sqrt()
    return distmat


# ============================================================
# 特征提取
# ============================================================
@torch.no_grad()
def extract_features_v2(model, loader, device):
    model.eval()
    all_features = []
    all_labels = []
    all_paths = []
    for batch in tqdm(loader, desc="  Extracting features", unit="batch"):
        imgs = batch[0].to(device)
        labels = batch[1]
        feats = model(imgs, labels=None, return_feature=True)
        all_features.append(feats.cpu())
        all_labels.append(labels)
        if len(batch) >= 3:
            all_paths.extend(batch[2])
    if len(all_features) == 0:
        return (torch.empty(0, 2048), torch.empty(0, dtype=torch.long), [])
    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return features, labels, all_paths


# ============================================================
# 检索指标
# ============================================================
def compute_retrieval_metrics(query_feats, query_labels,
                               gallery_feats, gallery_labels, num_ranks=20):
    if query_feats.size(0) == 0 or gallery_feats.size(0) == 0:
        return {"rank1": 0.0, "rank5": 0.0, "rank10": 0.0, "rank20": 0.0, "mAP": 0.0}

    distmat = compute_distance_matrix(query_feats, gallery_feats, metric="euclidean")
    num_q = distmat.size(0)
    q_labels_np = query_labels.numpy()
    g_labels_np = gallery_labels.numpy()
    indices = distmat.argsort(dim=1)

    aps = []
    cmc = torch.zeros(num_q, num_ranks)

    for i in range(num_q):
        q_label = q_labels_np[i]
        valid_mask = indices[i] != i
        sorted_idx = indices[i][valid_mask]
        sorted_labels = g_labels_np[sorted_idx.cpu().numpy()]
        matches_top = (sorted_labels[:num_ranks] == q_label)
        cumsum = matches_top.cumsum()
        for k in range(num_ranks):
            cmc[i, k] = 1.0 if cumsum[k] >= 1 else 0.0

        all_matches = (g_labels_np == q_label)
        all_matches[i] = False
        num_pos = all_matches.sum()
        if num_pos == 0:
            aps.append(0.0)
            continue

        sorted_labels_all = g_labels_np[indices[i][valid_mask].cpu().numpy()]
        matches_all = (sorted_labels_all == q_label)
        precisions = []
        correct = 0.0
        for k_idx, match in enumerate(matches_all, 1):
            if match:
                correct += 1
                precisions.append(correct / k_idx)
        aps.append(np.mean(precisions) if precisions else 0.0)

    mAP = np.mean(aps)
    cmc_avg = cmc.mean(dim=0)
    return {
        "rank1": cmc_avg[0].item() if num_ranks >= 1 else 0.0,
        "rank5": cmc_avg[4].item() if num_ranks >= 5 else 0.0,
        "rank10": cmc_avg[9].item() if num_ranks >= 10 else 0.0,
        "rank20": cmc_avg[19].item() if num_ranks >= 20 else 0.0,
        "mAP": mAP,
    }


# ============================================================
# 分类报告
# ============================================================
@torch.no_grad()
def compute_classification_report(model, loader, device, num_classes):
    model.eval()
    tp = torch.zeros(num_classes, dtype=torch.long)
    fp = torch.zeros(num_classes, dtype=torch.long)
    fn = torch.zeros(num_classes, dtype=torch.long)

    for batch in tqdm(loader, desc="  Class Report", unit="batch"):
        imgs = batch[0].to(device)
        labels = batch[1].to(device)
        logits, _, _ = model(imgs, labels=None)
        preds = logits.argmax(dim=1)
        for c in range(num_classes):
            pred_c = (preds == c)
            true_c = (labels == c)
            tp[c] += (pred_c & true_c).sum().cpu()
            fp[c] += (pred_c & ~true_c).sum().cpu()
            fn[c] += (~pred_c & true_c).sum().cpu()

    eps = 1e-8
    per_class_p = tp.float() / (tp + fp).float().clamp(min=eps)
    per_class_r = tp.float() / (tp + fn).float().clamp(min=eps)
    per_class_f1 = 2 * per_class_p * per_class_r / (per_class_p + per_class_r + eps)

    active = (tp + fn) > 0
    n_active = active.sum().item()

    macro_p = per_class_p[active].mean().item()
    macro_r = per_class_r[active].mean().item()
    macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r + eps) if (macro_p + macro_r) > 0 else 0.0

    micro_tp = tp.sum().float()
    micro_fp = fp.sum().float()
    micro_fn = fn.sum().float()
    micro_p = (micro_tp / (micro_tp + micro_fp + eps)).item()
    micro_r = (micro_tp / (micro_tp + micro_fn + eps)).item()
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + eps) if (micro_p + micro_r) > 0 else 0.0

    accuracy = tp.sum().item() / max(len(loader.dataset), 1)

    return {
        "accuracy": accuracy,
        "micro_p": micro_p, "micro_r": micro_r, "micro_f1": micro_f1,
        "macro_p": macro_p, "macro_r": macro_r, "macro_f1": macro_f1,
        "n_active": n_active,
    }


# ============================================================
# 训练一个 Epoch
# ============================================================
def train_epoch_v2(model, loader, criterion, optimizer, device, scaler, epoch, num_classes):
    model.train()
    running_loss = 0.0
    running_ce = 0.0
    running_tri = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d}")
    for imgs, labels, _ in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast('cuda'):
            logits, feat, bn_feat = model(imgs, labels=labels)
            loss, (ce_val, tri_val) = criterion(logits, feat, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if hasattr(model, 'gem_pool'):
            model.gem_pool.p.data.clamp_(min=1e-4)

        bs = imgs.size(0)
        running_loss += loss.item() * bs
        running_ce += ce_val * bs
        running_tri += tri_val * bs
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += bs

        pbar.set_postfix({
            "L": f"{loss.item():.3f}",
            "CE": f"{ce_val:.3f}",
            "Tri": f"{tri_val:.3f}",
            "Acc": f"{correct / max(total, 1):.3f}",
        })

    return {
        "loss": running_loss / max(total, 1),
        "ce": running_ce / max(total, 1),
        "tri": running_tri / max(total, 1),
        "acc": correct / max(total, 1),
    }


# ============================================================
# 训练曲线
# ============================================================
def plot_training_curves(history, output_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    axes[0].plot(epochs, history["train_loss"], 'b-', label='Train Loss', linewidth=1)
    axes[0].set_title('Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["train_ce"], 'b-', label='CE', linewidth=1)
    ax_twin = axes[1].twinx()
    ax_twin.plot(epochs, history["train_tri"], 'r-', label='Triplet', linewidth=1)
    axes[1].set_title('CE + Triplet Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(loc='upper left')
    ax_twin.legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["train_acc"], 'g-', linewidth=1)
    axes[2].set_title('Train Accuracy')
    axes[2].set_xlabel('Epoch')
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(epochs, history["train_micro_f1"], 'b-', label='Micro F1', linewidth=1)
    axes[3].plot(epochs, history["train_macro_f1"], 'r--', label='Macro F1', linewidth=1)
    axes[3].set_title('Train F1-Score')
    axes[3].set_xlabel('Epoch')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(epochs, history["lr"], 'm-', linewidth=1)
    axes[4].set_title('Learning Rate')
    axes[4].set_xlabel('Epoch')
    axes[4].set_yscale('log')
    axes[4].grid(True, alpha=0.3)

    mot17_epochs = [e for e in epochs if history["mot17_val_rank1"][e - 1] is not None]
    mot17_r1 = [history["mot17_val_rank1"][e - 1] for e in mot17_epochs]
    mot17_map = [history["mot17_val_map"][e - 1] for e in mot17_epochs]
    if mot17_epochs:
        axes[5].plot(mot17_epochs, mot17_r1, 'b-o', label='Rank-1', markersize=4, linewidth=1)
        axes[5].plot(mot17_epochs, mot17_map, 'r-s', label='mAP', markersize=4, linewidth=1)
    axes[5].set_title('MOT17 Val Retrieval')
    axes[5].set_xlabel('Epoch')
    axes[5].legend()
    axes[5].grid(True, alpha=0.3)

    mkt_epochs = [e for e in epochs if history["market_val_rank1"][e - 1] is not None]
    mkt_r1 = [history["market_val_rank1"][e - 1] for e in mkt_epochs]
    mkt_map = [history["market_val_map"][e - 1] for e in mkt_epochs]
    if mkt_epochs:
        axes[6].plot(mkt_epochs, mkt_r1, 'b-o', label='Rank-1', markersize=4, linewidth=1)
        axes[6].plot(mkt_epochs, mkt_map, 'r-s', label='mAP', markersize=4, linewidth=1)
    axes[6].set_title('Market-1501 Val Retrieval')
    axes[6].set_xlabel('Epoch')
    axes[6].legend()
    axes[6].grid(True, alpha=0.3)

    axes[7].plot(epochs, history["train_micro_p"], 'b-', label='Micro P', linewidth=1)
    axes[7].plot(epochs, history["train_micro_r"], 'r-', label='Micro R', linewidth=1)
    axes[7].set_title('Train Micro P/R')
    axes[7].set_xlabel('Epoch')
    axes[7].legend()
    axes[7].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(output_dir) / "training_curves.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Plot] Saved to {save_path}")


# ============================================================
# 主训练函数
# ============================================================
def train_combined(config):
    device = torch.device(config.get("device", "cuda"))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("=" * 70)
    print("  Kaggle: ReID V2 Combined Training")
    print("=" * 70)
    print(f"\nConfig:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    # ===== Phase 1: Data =====
    print("[Phase 1] Loading datasets...")
    train_transform = get_train_transforms()
    test_transform = get_test_transforms()

    data_configs = [
        {"path": config["market1501_dir"], "is_train": True, "pid_offset": 0},
        {"path": config["mot17_dir"], "is_train": True, "pid_offset": 751},
    ]
    train_dataset = CombinedReIDDataset(data_configs, transform=train_transform)
    num_classes = train_dataset.num_classes
    config["num_classes"] = num_classes

    # MOT17 val
    mot17_val_ds = ReIDDataset(
        str(config["mot17_dir"]), transform=test_transform, is_train=False
    )
    mot17_val_loader = DataLoader(mot17_val_ds, batch_size=128, shuffle=False,
                                  num_workers=2, pin_memory=True)

    # Market-1501 val
    mkt_gallery_ds = ReIDTestDataset(config["market1501_dir"],
                                     transform=test_transform, is_query=False)
    mkt_query_ds = ReIDTestDataset(config["market1501_dir"],
                                   transform=test_transform, is_query=True)
    mkt_gallery_loader = DataLoader(mkt_gallery_ds, batch_size=128, shuffle=False,
                                    num_workers=2, pin_memory=True)
    mkt_query_loader = DataLoader(mkt_query_ds, batch_size=128, shuffle=False,
                                  num_workers=2, pin_memory=True)

    # 训练集评估 loader
    train_eval_loader = DataLoader(train_dataset, batch_size=128, shuffle=False,
                                   num_workers=2, pin_memory=True)

    print(f"\n  MOT17 val:  {len(mot17_val_ds)} images, {mot17_val_ds.num_classes} IDs")
    print(f"  MKT gallery: {len(mkt_gallery_ds)} images")
    print(f"  MKT query:   {len(mkt_query_ds)} images")

    # PK 采样器
    num_instances = config.get("num_instances", 4)
    sampler = RandomIdentitySampler(train_dataset, num_instances=num_instances)
    batch_size = config.get("batch_size", 64)
    num_ids_per_batch = batch_size // num_instances
    print(f"\n  PK Sampler: {num_ids_per_batch} IDs x {num_instances} images = "
          f"{num_ids_per_batch * num_instances} batch_size")

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=sampler, num_workers=2,
                              pin_memory=True, drop_last=True)

    # ===== Phase 2: Model =====
    print("\n[Phase 2] Creating model...")
    model = create_model_v2(
        num_classes=num_classes,
        use_cbam=True,
        use_dilation=True,
        arc_scale=config.get("arc_scale", 30.0),
        arc_margin=config.get("arc_margin", 0.3),
        gem_p_init=config.get("gem_p_init", 3.0),
    )
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")

    # ===== Phase 3: Optimizer & Scheduler =====
    print("\n[Phase 3] Optimizer & Scheduler...")
    lr = config.get("lr", 3.5e-4)
    weight_decay = config.get("weight_decay", 5e-4)
    num_epochs = config.get("num_epochs", 120)
    warmup_epochs = config.get("warmup_epochs", 10)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=lr * 0.01
    )
    scheduler = GradualWarmupScheduler(
        optimizer, warmup_epochs, cosine_scheduler, warmup_factor=1e-3
    )
    scaler = GradScaler('cuda') if config.get("use_amp", True) else None

    criterion = CombinedReIDLoss(
        num_classes=num_classes,
        margin=config.get("tri_margin", 0.3),
        tri_weight=config.get("tri_weight", 1.0),
        label_smooth=config.get("label_smooth", 0.1),
    )

    # ===== Phase 4: Training Loop =====
    print(f"\n[Phase 4] Training ({num_epochs} epochs)...")
    print("=" * 70)

    history = defaultdict(list)
    best_mot17_rank1 = 0.0
    best_epoch = 0
    best_model_path = output_dir / "best_combined_v2.pth"
    early_stop_patience = config.get("early_stop_patience", 15)
    early_stop_counter = 0
    eval_retrieval_every = config.get("eval_retrieval_every", 5)
    log_file = output_dir / "training_log.txt"

    for key in ["train_loss", "train_ce", "train_tri", "train_acc",
                "train_micro_p", "train_micro_r", "train_micro_f1",
                "train_macro_p", "train_macro_r", "train_macro_f1",
                "mot17_val_tri", "mot17_val_rank1", "mot17_val_rank5",
                "mot17_val_rank10", "mot17_val_rank20", "mot17_val_map",
                "market_val_rank1", "market_val_rank5", "market_val_rank10",
                "market_val_rank20", "market_val_map", "lr"]:
        history[key] = []

    def log_and_print(msg):
        print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    log_and_print(f"{'Epoch':>5s} | {'Loss':>8s} {'CE':>8s} {'Tri':>8s} {'Acc':>7s} | "
                  f"{'MicroF1':>8s} {'MacroF1':>8s} | "
                  f"{'MOT17-R1':>9s} {'MOT17-mAP':>9s} | "
                  f"{'MKT-R1':>8s} {'MKT-mAP':>8s} | {'LR':>8s}")
    log_and_print("-" * 107)

    epoch_start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # Train
        train_metrics = train_epoch_v2(
            model, train_loader, criterion, optimizer, device, scaler, epoch, num_classes
        )

        # Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Classification report on full train set
        cls_report = compute_classification_report(model, train_eval_loader, device, num_classes)

        # MOT17 val triplet loss
        model.eval()
        mot17_tri = 0.0
        mot17_tri_count = 0
        with torch.no_grad():
            for imgs, labels, _ in mot17_val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                _, feat, _ = model(imgs, labels=labels)
                tri = criterion.tri_loss(feat, labels)
                mot17_tri += tri.item() * imgs.size(0)
                mot17_tri_count += imgs.size(0)
        mot17_val_tri = mot17_tri / max(mot17_tri_count, 1)

        # Full Retrieval Eval
        mot17_rank1 = mot17_rank5 = mot17_rank10 = mot17_rank20 = mot17_map = None
        mkt_rank1 = mkt_rank5 = mkt_rank10 = mkt_rank20 = mkt_map = None

        if epoch == 1 or epoch % eval_retrieval_every == 0 or epoch == num_epochs:
            print(f"\n  [Full Retrieval Eval @ Epoch {epoch}]")

            print(f"  MOT17 val...")
            mot17_qf, mot17_ql, _ = extract_features_v2(model, mot17_val_loader, device)
            mot17_metrics = compute_retrieval_metrics(mot17_qf, mot17_ql, mot17_qf, mot17_ql)
            mot17_rank1 = mot17_metrics["rank1"]
            mot17_rank5 = mot17_metrics["rank5"]
            mot17_rank10 = mot17_metrics["rank10"]
            mot17_rank20 = mot17_metrics["rank20"]
            mot17_map = mot17_metrics["mAP"]
            print(f"    Rank-1: {mot17_rank1:.4f}, Rank-5: {mot17_rank5:.4f}, "
                  f"Rank-10: {mot17_rank10:.4f}, mAP: {mot17_map:.4f}")

            print(f"  Market-1501 val...")
            mkt_gf, mkt_gl, _ = extract_features_v2(model, mkt_gallery_loader, device)
            mkt_qf, mkt_ql, _ = extract_features_v2(model, mkt_query_loader, device)
            mkt_metrics = compute_retrieval_metrics(mkt_qf, mkt_ql, mkt_gf, mkt_gl)
            mkt_rank1 = mkt_metrics["rank1"]
            mkt_rank5 = mkt_metrics["rank5"]
            mkt_rank10 = mkt_metrics["rank10"]
            mkt_rank20 = mkt_metrics["rank20"]
            mkt_map = mkt_metrics["mAP"]
            print(f"    Rank-1: {mkt_rank1:.4f}, Rank-5: {mkt_rank5:.4f}, "
                  f"Rank-10: {mkt_rank10:.4f}, mAP: {mkt_map:.4f}")

        # Logging
        history["train_loss"].append(train_metrics["loss"])
        history["train_ce"].append(train_metrics["ce"])
        history["train_tri"].append(train_metrics["tri"])
        history["train_acc"].append(train_metrics["acc"])
        history["train_micro_p"].append(cls_report["micro_p"])
        history["train_micro_r"].append(cls_report["micro_r"])
        history["train_micro_f1"].append(cls_report["micro_f1"])
        history["train_macro_p"].append(cls_report["macro_p"])
        history["train_macro_r"].append(cls_report["macro_r"])
        history["train_macro_f1"].append(cls_report["macro_f1"])
        history["mot17_val_tri"].append(mot17_val_tri)
        history["mot17_val_rank1"].append(mot17_rank1)
        history["mot17_val_rank5"].append(mot17_rank5)
        history["mot17_val_rank10"].append(mot17_rank10)
        history["mot17_val_rank20"].append(mot17_rank20)
        history["mot17_val_map"].append(mot17_map)
        history["market_val_rank1"].append(mkt_rank1)
        history["market_val_rank5"].append(mkt_rank5)
        history["market_val_rank10"].append(mkt_rank10)
        history["market_val_rank20"].append(mkt_rank20)
        history["market_val_map"].append(mkt_map)
        history["lr"].append(current_lr)

        log_line = (
            f"{epoch:5d} | "
            f"{train_metrics['loss']:8.4f} "
            f"{train_metrics['ce']:8.4f} "
            f"{train_metrics['tri']:8.4f} "
            f"{train_metrics['acc']:7.4f} | "
            f"{cls_report['micro_f1']:8.4f} "
            f"{cls_report['macro_f1']:8.4f} | "
            f"{mot17_rank1 or 0.0:9.4f} "
            f"{mot17_map or 0.0:9.4f} | "
            f"{mkt_rank1 or 0.0:8.4f} "
            f"{mkt_map or 0.0:8.4f} | "
            f"{current_lr:8.2e}"
        )
        log_and_print(log_line)
        log_and_print(
            f"       Train: Micro P={cls_report['micro_p']:.4f} "
            f"R={cls_report['micro_r']:.4f} "
            f"| Macro P={cls_report['macro_p']:.4f} "
            f"R={cls_report['macro_r']:.4f} "
            f"(active: {cls_report['n_active']}/{num_classes})"
        )

        epoch_time = time.time() - t0
        log_and_print(f"       Epoch time: {epoch_time:.1f}s")

        # Checkpoint
        is_best = False
        if mot17_rank1 is not None and mot17_rank1 > best_mot17_rank1:
            best_mot17_rank1 = mot17_rank1
            best_epoch = epoch
            early_stop_counter = 0
            is_best = True

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mot17_rank1': best_mot17_rank1,
                'config': config,
            }
            torch.save(checkpoint, best_model_path)
            log_and_print(f"       [BEST] Saved at epoch {epoch} (MOT17 Rank-1={best_mot17_rank1:.4f})")

        if epoch % 10 == 0:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mot17_rank1': best_mot17_rank1,
                'config': config,
                'history': dict(history),
            }, ckpt_path)
            log_and_print(f"       [CKPT] Saved {ckpt_path.name}")

        if epoch % 10 == 0 or is_best:
            plot_training_curves(history, output_dir)

        # Early stopping
        if mot17_rank1 is not None:
            if not is_best:
                early_stop_counter += eval_retrieval_every
            if early_stop_counter >= early_stop_patience:
                log_and_print(f"\n  Early stopping at epoch {epoch}")
                break

    # ===== Phase 5: Finalize =====
    total_time = time.time() - epoch_start_time
    print(f"\n{'=' * 70}")
    print(f"  Training Complete")
    print(f"  Total time: {total_time/3600:.2f}h")
    print(f"  Best epoch: {best_epoch}, Best MOT17 Rank-1: {best_mot17_rank1:.4f}")
    print(f"{'=' * 70}")

    final_path = output_dir / "final_combined_v2.pth"
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': dict(history),
        'best_mot17_rank1': best_mot17_rank1,
    }, final_path)
    print(f"  Final model saved: {final_path}")

    plot_training_curves(history, output_dir)

    log_and_print(f"\n{'=' * 60}")
    log_and_print(f"  Final Report")
    log_and_print(f"{'=' * 60}")
    log_and_print(f"  Best Epoch:             {best_epoch}")
    log_and_print(f"  Best MOT17 Rank-1:      {best_mot17_rank1:.4f}")
    log_and_print(f"  Train Micro F1 (final): {history['train_micro_f1'][-1]:.4f}")
    log_and_print(f"  Train Macro F1 (final): {history['train_macro_f1'][-1]:.4f}")
    if history["market_val_rank1"][-1] is not None:
        log_and_print(f"  Market-1501 Rank-1:     {history['market_val_rank1'][-1]:.4f}")
        log_and_print(f"  Market-1501 mAP:        {history['market_val_map'][-1]:.4f}")
    log_and_print(f"{'=' * 60}")

    return best_model_path


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle ReID V2 Combined Training")

    parser.add_argument("--market1501_dir", type=str, default=DEFAULT_MARKET1501_DIR)
    parser.add_argument("--mot17_dir", type=str, default=DEFAULT_MOT17_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--num_epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_instances", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3.5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--warmup_epochs", type=int, default=10)

    parser.add_argument("--arc_scale", type=float, default=30.0)
    parser.add_argument("--arc_margin", type=float, default=0.3)
    parser.add_argument("--gem_p_init", type=float, default=3.0)
    parser.add_argument("--tri_margin", type=float, default=0.3)
    parser.add_argument("--tri_weight", type=float, default=1.0)
    parser.add_argument("--label_smooth", type=float, default=0.1)

    parser.add_argument("--early_stop_patience", type=int, default=15)
    parser.add_argument("--eval_retrieval_every", type=int, default=5)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    config = {
        "market1501_dir": args.market1501_dir,
        "mot17_dir": args.mot17_dir,
        "output_dir": args.output_dir,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "num_instances": args.num_instances,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "arc_scale": args.arc_scale,
        "arc_margin": args.arc_margin,
        "gem_p_init": args.gem_p_init,
        "tri_margin": args.tri_margin,
        "tri_weight": args.tri_weight,
        "label_smooth": args.label_smooth,
        "early_stop_patience": args.early_stop_patience,
        "eval_retrieval_every": args.eval_retrieval_every,
        "use_amp": not args.no_amp,
        "seed": args.seed,
        "device": args.device,
    }

    train_combined(config)
