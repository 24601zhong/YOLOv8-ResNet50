"""
============================================================
ReID 联合训练脚本 train_combined.py

联合 Market-1501 + MOT17 数据集，使用 V2 模型端到端训练。

改进点:
  1. 数据集: Market-1501 (751 ID) + MOT17 (223 ID) → 974 统一 ID
  2. 采样:   PK 采样 (P 个身份，各 K 张图)，保证每 batch 有正样本对
  3. 模型:   ImprovedResNet50V2 (GeM + ArcFace + BNNeck)
  4. 损失:   LabelSmoothingCE + TripletLoss (hard mining)
  5. 优化器: AdamW (decoupled weight decay) + Warmup + CosineAnnealingLR
  6. 评估:   每 epoch: Micro/Macro P/R/F1 (全训练集，消除 batch 偏差)
            每 5 epoch: MOT17 + Market-1501 双域检索评估 (Rank-K/mAP)

指标定义:
  - Micro:  每个样本等权 → 反映全局分类质量
  - Macro:  每个类别等权 → 反映尾部类别的表现
  - Rank-K: 检索 top-K 命中率
  - mAP:    检索平均精度

用法:
  python train_combined.py
  python train_combined.py --epochs 80 --batch_size 64 --num_instances 4
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

# 路径设置
ROOT = Path(__file__).resolve().parent.parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT / 'resnet50_reid_train'))

from model_v2 import create_model_v2, CombinedReIDLoss
from train_reid import ReIDDataset, get_train_transforms, get_test_transforms
from test_reid import ReIDTestDataset

warnings.filterwarnings('ignore')

# ============================================================
# 全局配置默认值
# ============================================================
DEFAULT_MARKET1501_DIR = r"C:\D\Myproject\Data-processing\Market-1501-v15.09.15"
DEFAULT_MOT17_DIR = ROOT / "dataset" / "mot17_reid_clean"
DEFAULT_OUTPUT_DIR = ROOT / "resnet50_reid_train" / "train_output" / "combined_v2_log"


# ============================================================
# 联合数据集
# ============================================================
class CombinedReIDDataset(Dataset):
    """
    合并多个 ReID 数据集，统一 PID 索引。

    通过 pid_offset 机制避免不同数据集的 PID 冲突：
      Market-1501 train: 751 IDs → offset 0 (indices 0..750)
      MOT17 train:       223 IDs → offset 751 (indices 751..973)
    """

    def __init__(self, data_configs, transform=None):
        """
        Args:
            data_configs: list of dicts, each:
                {"path": str, "is_train": bool, "pid_offset": int}
            transform: 图像变换
        """
        self.data_configs = data_configs
        self.transform = transform

        self.images = []       # List[Path]
        self.labels = []       # List[int] 统一后的标签
        self.sources = []      # List[str] 来源标记

        total_ids = 0
        for cfg in data_configs:
            ds = ReIDDataset(
                str(cfg["path"]),
                transform=None,   # 先不 transform，在 __getitem__ 中做
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
# PK 采样器 (Random Identity Sampler)
# ============================================================
class RandomIdentitySampler(Sampler):
    """
    每个 batch 包含 P 个身份，每个身份 K 张图 (batch_size = P × K)。

    确保每个 batch 都有正样本对，让 TripletLoss 能够有效挖掘难样本。
    """

    def __init__(self, dataset, num_instances=4):
        """
        Args:
            dataset: 必须具有 .labels 属性 (List[int])
            num_instances: 每个身份的图片数 K
        """
        self.dataset = dataset
        self.num_instances = num_instances

        # 构建 per-class 索引字典
        self.index_dic = defaultdict(list)
        for idx, label in enumerate(dataset.labels):
            self.index_dic[label].append(idx)

        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        # 1. 随机排列所有 PID
        indices = []
        pids_shuffled = self.pids.copy()
        random.shuffle(pids_shuffled)

        # 2. 对每个 PID, 采样 K 张图
        for pid in pids_shuffled:
            pid_indices = self.index_dic[pid]
            if len(pid_indices) >= self.num_instances:
                sampled = random.sample(pid_indices, self.num_instances)
            else:
                # 样本不足时有放回采样
                sampled = random.choices(pid_indices, k=self.num_instances)
            indices.extend(sampled)

        return iter(indices)

    def __len__(self):
        return len(self.pids) * self.num_instances


# ============================================================
# Warmup 调度器
# ============================================================
class GradualWarmupScheduler:
    """
    线性预热 + 余弦退火

    前 warmup_epochs 轮: lr 从 base_lr * warmup_factor 线性增加到 base_lr
    之后: 使用 CosineAnnealingLR 衰减到 base_lr * 1e-2
    """

    def __init__(self, optimizer, warmup_epochs, after_scheduler, warmup_factor=1e-3):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.warmup_factor = warmup_factor
        self.current_epoch = 0
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # 线性预热: lr = base_lr * [warmup_factor + (1 - warmup_factor) * progress]
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
# 距离矩阵计算
# ============================================================
def compute_distance_matrix(query_feats, gallery_feats, metric="euclidean"):
    """计算 query-gallery 欧氏距离矩阵"""
    m, n = query_feats.size(0), gallery_feats.size(0)
    distmat = (
        torch.pow(query_feats, 2).sum(dim=1, keepdim=True).expand(m, n)
        + torch.pow(gallery_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    distmat = distmat - 2 * torch.mm(query_feats, gallery_feats.t())
    distmat = distmat.clamp(min=1e-12).sqrt()
    return distmat


# ============================================================
# 特征提取（V2 模型）
# ============================================================
@torch.no_grad()
def extract_features_v2(model, loader, device):
    """从 V2 模型提取 L2 归一化特征"""
    model.eval()
    all_features = []
    all_labels = []
    all_paths = []

    for batch in tqdm(loader, desc="  Extracting features", unit="batch"):
        # 兼容 2/3/4 返回值的数据集
        imgs = batch[0].to(device)
        labels = batch[1]

        feats = model(imgs, labels=None, return_feature=True)  # already L2-normalized
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
# 检索指标计算
# ============================================================
def compute_retrieval_metrics(query_feats, query_labels,
                               gallery_feats, gallery_labels,
                               num_ranks=20):
    """
    计算标准 ReID 检索指标: Rank-K, mAP

    Args:
        query_feats:   [Nq, D] L2-normalized
        query_labels:  [Nq]
        gallery_feats: [Ng, D]
        gallery_labels:[Ng]
        num_ranks: 最大 Rank-K

    Returns:
        dict: {"rank1": ..., "rank5": ..., "rank10": ..., "rank20": ..., "mAP": ...}
    """
    if query_feats.size(0) == 0 or gallery_feats.size(0) == 0:
        return {"rank1": 0.0, "rank5": 0.0, "rank10": 0.0, "rank20": 0.0, "mAP": 0.0}

    # 距离矩阵
    distmat = compute_distance_matrix(query_feats, gallery_feats, metric="euclidean")
    # distmat[i][j] = distance between query_i and gallery_j

    num_q = distmat.size(0)
    q_labels_np = query_labels.numpy()
    g_labels_np = gallery_labels.numpy()

    indices = distmat.argsort(dim=1)  # [Nq, Ng]

    aps = []
    cmc = torch.zeros(num_q, num_ranks)

    for i in range(num_q):
        q_label = q_labels_np[i]

        # 有效 gallery: 排除自身（同一个图）
        valid_mask = indices[i] != i
        sorted_idx = indices[i][valid_mask]
        sorted_labels = g_labels_np[sorted_idx.cpu().numpy()]

        # CMC
        matches_top = (sorted_labels[:num_ranks] == q_label)
        cumsum = matches_top.cumsum()
        for k in range(num_ranks):
            cmc[i, k] = 1.0 if cumsum[k] >= 1 else 0.0

        # AP: 所有正样本
        all_matches = (g_labels_np == q_label)
        all_matches[i] = False  # 排除自身
        num_pos = all_matches.sum()

        if num_pos == 0:
            aps.append(0.0)
            continue

        # 按距离排序的所有 gallery 标签（排除自身）
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
    rank1 = cmc_avg[0].item() if num_ranks >= 1 else 0.0
    rank5 = cmc_avg[4].item() if num_ranks >= 5 else 0.0
    rank10 = cmc_avg[9].item() if num_ranks >= 10 else 0.0
    rank20 = cmc_avg[19].item() if num_ranks >= 20 else 0.0

    return {
        "rank1": rank1, "rank5": rank5, "rank10": rank10, "rank20": rank20,
        "mAP": mAP,
    }


# ============================================================
# 分类报告计算（全训练集 Micro/Macro P/R/F1）
# ============================================================
@torch.no_grad()
def compute_classification_report(model, loader, device, num_classes):
    """
    在全量训练集上计算 Micro/Macro P/R/F1，消除 batch-size 偏差。

    Returns:
        dict: accuracy, micro_p/r/f1, macro_p/r/f1
    """
    model.eval()
    tp = torch.zeros(num_classes, dtype=torch.long)
    fp = torch.zeros(num_classes, dtype=torch.long)
    fn = torch.zeros(num_classes, dtype=torch.long)

    for batch in tqdm(loader, desc="  Class Report", unit="batch"):
        imgs = batch[0].to(device)
        labels = batch[1].to(device)

        # 评估时不传 labels → ArcFace 不做 margin，用 raw cos_theta * scale
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

    # Macro
    macro_p = per_class_p[active].mean().item()
    macro_r = per_class_r[active].mean().item()
    macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r + eps) if (macro_p + macro_r) > 0 else 0.0

    # Micro
    micro_tp = tp.sum().float()
    micro_fp = fp.sum().float()
    micro_fn = fn.sum().float()
    micro_p = (micro_tp / (micro_tp + micro_fp + eps)).item()
    micro_r = (micro_tp / (micro_tp + micro_fn + eps)).item()
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + eps) if (micro_p + micro_r) > 0 else 0.0

    # Accuracy
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
    """训练一个 epoch (V2 模型)"""
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

        # GeM p 值裁剪（防止小到数值不稳定）
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
# 训练曲线绘制
# ============================================================
def plot_training_curves(history, output_dir):
    """绘制综合训练曲线"""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    # 1. Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], 'b-', label='Train Loss', linewidth=1)
    ax.set_title('Train Loss')
    ax.set_xlabel('Epoch')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. CE / Triplet
    ax = axes[1]
    ax.plot(epochs, history["train_ce"], 'b-', label='CE', linewidth=1)
    ax_twin = ax.twinx()
    ax_twin.plot(epochs, history["train_tri"], 'r-', label='Triplet', linewidth=1)
    ax.set_title('CE + Triplet Loss')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper left')
    ax_twin.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 3. Train Accuracy
    ax = axes[2]
    ax.plot(epochs, history["train_acc"], 'g-', linewidth=1)
    ax.set_title('Train Accuracy')
    ax.set_xlabel('Epoch')
    ax.grid(True, alpha=0.3)

    # 4. Micro / Macro F1
    ax = axes[3]
    ax.plot(epochs, history["train_micro_f1"], 'b-', label='Micro F1', linewidth=1)
    ax.plot(epochs, history["train_macro_f1"], 'r--', label='Macro F1', linewidth=1)
    ax.set_title('Train F1-Score')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Learning Rate
    ax = axes[4]
    ax.plot(epochs, history["lr"], 'm-', linewidth=1)
    ax.set_title('Learning Rate')
    ax.set_xlabel('Epoch')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 6. MOT17 Val Rank-1/mAP
    ax = axes[5]
    mot17_epochs = [e for e in epochs if history["mot17_val_rank1"][e - 1] is not None]
    mot17_r1 = [history["mot17_val_rank1"][e - 1] for e in mot17_epochs]
    mot17_map = [history["mot17_val_map"][e - 1] for e in mot17_epochs]
    if mot17_epochs:
        ax.plot(mot17_epochs, mot17_r1, 'b-o', label='Rank-1', markersize=4, linewidth=1)
        ax.plot(mot17_epochs, mot17_map, 'r-s', label='mAP', markersize=4, linewidth=1)
    ax.set_title('MOT17 Val Retrieval')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 7. Market-1501 Val Rank-1/mAP
    ax = axes[6]
    mkt_epochs = [e for e in epochs if history["market_val_rank1"][e - 1] is not None]
    mkt_r1 = [history["market_val_rank1"][e - 1] for e in mkt_epochs]
    mkt_map = [history["market_val_map"][e - 1] for e in mkt_epochs]
    if mkt_epochs:
        ax.plot(mkt_epochs, mkt_r1, 'b-o', label='Rank-1', markersize=4, linewidth=1)
        ax.plot(mkt_epochs, mkt_map, 'r-s', label='mAP', markersize=4, linewidth=1)
    ax.set_title('Market-1501 Val Retrieval')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 8. Micro P/R
    ax = axes[7]
    ax.plot(epochs, history["train_micro_p"], 'b-', label='Micro P', linewidth=1)
    ax.plot(epochs, history["train_micro_r"], 'r-', label='Micro R', linewidth=1)
    ax.set_title('Train Micro P/R')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(output_dir) / "training_curves.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Plot] Saved to {save_path}")


# ============================================================
# 主训练函数
# ============================================================
def train_combined(config):
    """
    联合训练主函数

    Args:
        config: dict, 训练超参配置
    """
    # ===== Phase 0: Setup =====
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
    print("  ReID V2 Combined Training")
    print("=" * 70)
    print(f"\nConfig:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    # ===== Phase 1: Data =====
    print("[Phase 1] Loading datasets...")
    train_transform = get_train_transforms()
    test_transform = get_test_transforms()

    # 联合训练集
    data_configs = [
        {"path": config["market1501_dir"], "is_train": True, "pid_offset": 0},
        {"path": config["mot17_dir"], "is_train": True, "pid_offset": 751},
    ]
    train_dataset = CombinedReIDDataset(data_configs, transform=train_transform)
    num_classes = train_dataset.num_classes
    config["num_classes"] = num_classes

    # 验证集
    # MOT17 val (cross-domain)
    mot17_val_ds = ReIDDataset(
        str(config["mot17_dir"]), transform=test_transform, is_train=False
    )
    mot17_val_loader = DataLoader(mot17_val_ds, batch_size=128, shuffle=False,
                                  num_workers=0, pin_memory=True)

    # Market-1501 val (retrieval)
    mkt_gallery_ds = ReIDTestDataset(config["market1501_dir"],
                                     transform=test_transform, is_query=False)
    mkt_query_ds = ReIDTestDataset(config["market1501_dir"],
                                   transform=test_transform, is_query=True)
    mkt_gallery_loader = DataLoader(mkt_gallery_ds, batch_size=128, shuffle=False,
                                    num_workers=0, pin_memory=True)
    mkt_query_loader = DataLoader(mkt_query_ds, batch_size=128, shuffle=False,
                                   num_workers=0, pin_memory=True)

    # 训练集评估用 loader（不打乱，用于全量分类报告）
    train_eval_loader = DataLoader(train_dataset, batch_size=128, shuffle=False,
                                   num_workers=0, pin_memory=True)

    print(f"\n  MOT17 val:  {len(mot17_val_ds)} images, {mot17_val_ds.num_classes} IDs")
    print(f"  MKT gallery: {len(mkt_gallery_ds)} images")
    print(f"  MKT query:   {len(mkt_query_ds)} images")
    print(f"  Train eval:  {len(train_dataset)} images (for classification report)")

    # PK 采样器
    num_instances = config.get("num_instances", 4)
    sampler = RandomIdentitySampler(train_dataset, num_instances=num_instances)
    batch_size = config.get("batch_size", 64)
    # P = batch_size / K
    num_identities_per_batch = batch_size // num_instances
    print(f"\n  PK Sampler: {num_identities_per_batch} IDs x {num_instances} images = "
          f"{num_identities_per_batch * num_instances} batch_size")

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=sampler, num_workers=0,
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
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:    {total_params:,}")
    print(f"  Trainable params:{trainable_params:,}")

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

    # ===== Phase 3.5: Resume from checkpoint (if specified) =====
    start_epoch = 1
    resume_epoch = 0
    history = defaultdict(list)
    best_mot17_rank1 = 0.0
    best_epoch = 0
    early_stop_counter = 0
    resume_path = config.get("resume")

    if resume_path:
        print(f"\n[Phase 3.5] Resuming from checkpoint: {resume_path}")
        if not Path(resume_path).exists():
            print(f"  [ERROR] Checkpoint not found: {resume_path}")
            sys.exit(1)

        ckpt = torch.load(resume_path, map_location=device, weights_only=False)

        # Restore model
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Model state restored")

        # Restore optimizer
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print(f"  Optimizer state restored")

        # Restore scheduler
        if 'scheduler_state_dict' in ckpt:
            try:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                print(f"  Scheduler state restored (warmup epoch={scheduler.current_epoch})")
            except Exception as e:
                print(f"  [WARN] Scheduler state load failed: {e}, starting scheduler fresh")

        # Restore AMP scaler
        if 'scaler_state_dict' in ckpt and scaler is not None:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
            print(f"  AMP scaler state restored")

        # Restore best metrics
        best_mot17_rank1 = ckpt.get('best_mot17_rank1', 0.0)
        best_epoch = ckpt.get('epoch', 0)
        resume_epoch = best_epoch
        start_epoch = resume_epoch + 1
        print(f"  Best MOT17 Rank-1 so far: {best_mot17_rank1:.4f} (epoch {best_epoch})")

        # Restore history (for continued plotting)
        if 'history' in ckpt and ckpt['history']:
            for k, v in ckpt['history'].items():
                history[k] = list(v)
            print(f"  History restored ({len(history.keys())} metrics, "
                  f"{len(history.get('train_loss', []))} epochs)")

        # Early stop counter from checkpoint
        early_stop_counter = ckpt.get('early_stop_counter', 0)

        # Override num_epochs from checkpoint config if CLI didn't change it
        if 'config' in ckpt:
            ckpt_config = ckpt['config']
            # Respect CLI overrides: only use checkpoint total epochs if CLI used default
            if config.get('num_epochs') == 120:  # default not changed
                config['num_epochs'] = ckpt_config.get('num_epochs', num_epochs)
            num_epochs = config['num_epochs']

        if start_epoch > num_epochs:
            print(f"  [INFO] Checkpoint epoch ({resume_epoch}) >= target epochs ({num_epochs}). "
                  f"Training already complete.")
            sys.exit(0)

        print(f"  Resuming from epoch {start_epoch} / {num_epochs}")
        print(f"  Remaining epochs: {num_epochs - start_epoch + 1}")

    # ===== Phase 4: Training Loop =====
    print(f"\n[Phase 4] Training ({num_epochs} epochs, start={start_epoch})...")
    print("=" * 70)

    best_model_path = output_dir / "best_combined_v2.pth"
    early_stop_patience = config.get("early_stop_patience", 15)
    eval_retrieval_every = config.get("eval_retrieval_every", 5)
    log_file = output_dir / "training_log.txt"

    # 初始化 history 占位
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
    for epoch in range(start_epoch, num_epochs + 1):
        t0 = time.time()

        # ---- Train ----
        train_metrics = train_epoch_v2(
            model, train_loader, criterion, optimizer, device, scaler,
            epoch, num_classes
        )

        # ---- Scheduler ----
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ---- Quick Validation (every epoch) ----
        # Classification report on full train set
        cls_report = compute_classification_report(
            model, train_eval_loader, device, num_classes
        )

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

        # ---- Full Retrieval Eval (every N epochs) ----
        mot17_rank1 = mot17_rank5 = mot17_rank10 = mot17_rank20 = mot17_map = None
        mkt_rank1 = mkt_rank5 = mkt_rank10 = mkt_rank20 = mkt_map = None

        if epoch == 1 or epoch % eval_retrieval_every == 0 or epoch == num_epochs:
            print(f"\n  [Full Retrieval Eval @ Epoch {epoch}]")

            # MOT17 retrieval
            print(f"  MOT17 val...")
            mot17_qf, mot17_ql, _ = extract_features_v2(model, mot17_val_loader, device)
            mot17_metrics = compute_retrieval_metrics(
                mot17_qf, mot17_ql, mot17_qf, mot17_ql
            )
            mot17_rank1 = mot17_metrics["rank1"]
            mot17_rank5 = mot17_metrics["rank5"]
            mot17_rank10 = mot17_metrics["rank10"]
            mot17_rank20 = mot17_metrics["rank20"]
            mot17_map = mot17_metrics["mAP"]
            print(f"    Rank-1: {mot17_rank1:.4f}, Rank-5: {mot17_rank5:.4f}, "
                  f"Rank-10: {mot17_rank10:.4f}, mAP: {mot17_map:.4f}")

            # Market-1501 retrieval
            print(f"  Market-1501 val...")
            mkt_gf, mkt_gl, _ = extract_features_v2(model, mkt_gallery_loader, device)
            mkt_qf, mkt_ql, _ = extract_features_v2(model, mkt_query_loader, device)
            mkt_metrics = compute_retrieval_metrics(
                mkt_qf, mkt_ql, mkt_gf, mkt_gl
            )
            mkt_rank1 = mkt_metrics["rank1"]
            mkt_rank5 = mkt_metrics["rank5"]
            mkt_rank10 = mkt_metrics["rank10"]
            mkt_rank20 = mkt_metrics["rank20"]
            mkt_map = mkt_metrics["mAP"]
            print(f"    Rank-1: {mkt_rank1:.4f}, Rank-5: {mkt_rank5:.4f}, "
                  f"Rank-10: {mkt_rank10:.4f}, mAP: {mkt_map:.4f}")

        # ---- Logging ----
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

        # ---- Console log ----
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

        # Detailed metrics every epoch
        log_and_print(
            f"       Train: Micro P={cls_report['micro_p']:.4f} "
            f"R={cls_report['micro_r']:.4f} "
            f"| Macro P={cls_report['macro_p']:.4f} "
            f"R={cls_report['macro_r']:.4f} "
            f"(active: {cls_report['n_active']}/{num_classes})"
        )

        epoch_time = time.time() - t0
        log_and_print(f"       Epoch time: {epoch_time:.1f}s")

        # ---- Checkpoint ----
        # Best model (primary: MOT17 Rank-1)
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
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'best_mot17_rank1': best_mot17_rank1,
                'early_stop_counter': early_stop_counter,
                'config': config,
            }
            torch.save(checkpoint, best_model_path)
            log_and_print(f"       [BEST] Saved at epoch {epoch} (MOT17 Rank-1={best_mot17_rank1:.4f})")

        # Periodic checkpoint
        if epoch % 10 == 0:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'best_mot17_rank1': best_mot17_rank1,
                'early_stop_counter': early_stop_counter,
                'config': config,
                'history': dict(history),
            }, ckpt_path)
            log_and_print(f"       [CKPT] Saved {ckpt_path.name}")

        # Plot curves periodically
        if epoch % 10 == 0 or is_best:
            plot_training_curves(history, output_dir)

        # ---- Early stopping ----
        if mot17_rank1 is not None:
            if not is_best:
                early_stop_counter += eval_retrieval_every
            if early_stop_counter >= early_stop_patience:
                log_and_print(f"\n  Early stopping at epoch {epoch} "
                              f"(MOT17 Rank-1 not improved for {early_stop_counter} epochs)")
                break

    # ===== Phase 5: Finalize =====
    total_time = time.time() - epoch_start_time
    print(f"\n{'=' * 70}")
    print(f"  Training Complete")
    print(f"  Total time: {total_time/3600:.2f}h")
    print(f"  Best epoch: {best_epoch}, Best MOT17 Rank-1: {best_mot17_rank1:.4f}")
    print(f"{'=' * 70}")

    # Save final model
    final_path = output_dir / "final_combined_v2.pth"
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': dict(history),
        'best_mot17_rank1': best_mot17_rank1,
    }, final_path)
    print(f"  Final model saved: {final_path}")

    # Final plots
    plot_training_curves(history, output_dir)

    # Final report
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
    parser = argparse.ArgumentParser(description="ReID V2 Combined Training")

    # Paths
    parser.add_argument("--market1501_dir", type=str, default=DEFAULT_MARKET1501_DIR,
                        help="Market-1501 dataset root directory")
    parser.add_argument("--mot17_dir", type=str, default=str(DEFAULT_MOT17_DIR),
                        help="MOT17 clean dataset directory")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Output directory for weights/logs/plots")

    # Training
    parser.add_argument("--num_epochs", type=int, default=120, help="Total epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="P x K")
    parser.add_argument("--num_instances", type=int, default=4,
                        help="K: images per identity per batch")
    parser.add_argument("--lr", type=float, default=3.5e-4, help="Base learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="AdamW weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Linear warmup epochs")

    # Model
    parser.add_argument("--arc_scale", type=float, default=30.0, help="ArcFace scale s")
    parser.add_argument("--arc_margin", type=float, default=0.3, help="ArcFace margin m")
    parser.add_argument("--gem_p_init", type=float, default=3.0, help="GeM pooling initial p")

    # Loss
    parser.add_argument("--tri_margin", type=float, default=0.3, help="Triplet loss margin")
    parser.add_argument("--tri_weight", type=float, default=1.0, help="Triplet loss weight")
    parser.add_argument("--label_smooth", type=float, default=0.1, help="Label smoothing epsilon")

    # Misc
    parser.add_argument("--early_stop_patience", type=int, default=15,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--eval_retrieval_every", type=int, default=5,
                        help="Full retrieval eval frequency (epochs)")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path (e.g. checkpoint_epoch20.pth)")

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
        "resume": args.resume,
    }

    train_combined(config)
