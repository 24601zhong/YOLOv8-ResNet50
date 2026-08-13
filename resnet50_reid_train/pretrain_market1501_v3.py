"""
============================================================
Stage 1: Market-1501 Pretraining with IBNetResNet50 V3
============================================================
预训练 Market-1501 (751 IDs) → 为 MOT17 微调提供跨域基础权重

架构: IBNet50-a + CBAM + Dilated + GeM + ArcFace
损失: LabelSmoothingCE(ε=0.1) + BatchHardTriplet(λ=0.5)
采样: PK Sampler (16 IDs × 4 images = batch 64)
优化: AdamW + GradualWarmup → CosineAnnealing
输入: 256×128
============================================================
"""

import os
import sys
import time
import random
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# Path setup
# ============================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from model_v3 import create_model_v3, create_loss_v3, BatchHardTripletLoss

# ============================================================
# Config
# ============================================================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ============================================================
# PK Sampler (from train_combined.py)
# ============================================================
class RandomIdentitySampler(Sampler):
    """PK 采样: 每 batch 含 P 个身份, 每个身份 K 张图"""

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
# Warmup Scheduler
# ============================================================
class GradualWarmupScheduler:
    """线性预热 + 余弦退火"""

    def __init__(self, optimizer, warmup_epochs, after_scheduler, warmup_factor=1e-3):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.warmup_factor = warmup_factor
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.current_step = 0
        self.warmup_steps = 0  # 由 epoch 长度动态确定

    def set_warmup_steps(self, steps_per_epoch):
        self.warmup_steps = self.warmup_epochs * steps_per_epoch

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            # 线性从 warmup_factor * base_lr → base_lr
            progress = self.current_step / max(self.warmup_steps, 1)
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = base_lr * (self.warmup_factor + (1.0 - self.warmup_factor) * progress)
        else:
            self.after_scheduler.step()


# ============================================================
# Dataset
# ============================================================
class Market1501Dataset(Dataset):
    """Market-1501 数据集加载器"""

    def __init__(self, data_dir, transform=None, is_train=True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_train = is_train

        subdir = "bounding_box_train" if is_train else "bounding_box_test"
        img_dir = self.data_dir / subdir
        if not img_dir.exists():
            img_dir = self.data_dir

        self.images = []
        self.labels = []
        person_ids = set()
        raw_items = []

        for img_path in sorted(img_dir.glob("*.jpg")):
            fname = img_path.stem
            try:
                pid = int(fname.split("_")[0])
            except (ValueError, IndexError):
                pid = hash(fname) % 10000

            if pid == -1:  # junk
                continue

            raw_items.append((img_path, pid))
            person_ids.add(pid)

        self.pid_to_idx = {pid: i for i, pid in enumerate(sorted(person_ids))}
        self.images = [p for p, _ in raw_items]
        self.labels = [self.pid_to_idx[pid] for _, pid in raw_items]
        self.num_classes = len(self.pid_to_idx)

        print(f"[Market-1501] {'Train' if is_train else 'Test'}: "
              f"{len(self.images)} images, {self.num_classes} IDs")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx], str(self.images[idx])


# ============================================================
# Transforms
# ============================================================
def get_train_transforms_v3(height=256, width=128):
    return T.Compose([
        T.Resize((height, width)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.15),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])


def get_test_transforms_v3(height=256, width=128):
    return T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def extract_features(model, loader, device):
    """提取所有图像的特征"""
    model.eval()
    all_features = []
    all_labels = []

    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        feats = model(imgs, return_feature=True)
        all_features.append(feats.cpu())
        all_labels.append(labels)

    if not all_features:
        return torch.empty(0, 2048), torch.empty(0, dtype=torch.long)

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


def compute_mAP_rank1(query_feats, query_labels, gallery_feats, gallery_labels):
    """计算 mAP 和 Rank-1 (single query)"""
    # 欧氏距离矩阵
    m, n = query_feats.size(0), gallery_feats.size(0)
    distmat = (
        torch.pow(query_feats, 2).sum(dim=1, keepdim=True).expand(m, n)
        + torch.pow(gallery_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    distmat = distmat - 2 * torch.mm(query_feats, gallery_feats.t())
    distmat = distmat.clamp(min=1e-12).sqrt()

    ql = query_labels.numpy()
    gl = gallery_labels.numpy()

    indices = distmat.argsort(dim=1)
    aps = []
    cmc_hits = 0

    for i in range(m):
        # 排除自身 (相同特征)
        sorted_idx = indices[i]
        sorted_labels = gl[sorted_idx.cpu().numpy()]

        # 对于 Market-1501: query 和 gallery 是同一个集合时需要排除自身
        # 这里 gallery=query set, 跳过第一个结果 (距离=0 即自身)
        matches = (sorted_labels[1:] == ql[i])
        num_positives = (gl == ql[i]).sum() - 1  # 排除自身

        if num_positives <= 0:
            aps.append(0.0)
            continue

        # Rank-1
        if matches[0]:
            cmc_hits += 1

        # mAP
        correct = 0.0
        precisions = []
        for k, match in enumerate(matches, 1):
            if match:
                correct += 1
                precisions.append(correct / k)
        aps.append(np.mean(precisions) if precisions else 0.0)

    mAP = np.mean(aps)
    rank1 = cmc_hits / m
    return mAP, rank1


@torch.no_grad()
def evaluate(model, loader, device):
    """在验证集上评估 mAP 和 Rank-1"""
    feats, labels = extract_features(model, loader, device)
    if feats.size(0) == 0:
        return 0.0, 0.0
    mAP, rank1 = compute_mAP_rank1(feats, labels, feats, labels)
    return mAP, rank1


# ============================================================
# Training
# ============================================================
def train_epoch(model, loader, criterion, optimizer, scheduler, device, scaler, epoch):
    """训练一个 epoch"""
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
            logits, feat, _ = model(imgs, labels=labels)
            loss, (ce_val, tri_val) = criterion(logits, feat, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if isinstance(scheduler, GradualWarmupScheduler):
            scheduler.step()  # per-batch warmup

        bs = imgs.size(0)
        running_loss += loss.item() * bs
        running_ce += ce_val * bs
        running_tri += tri_val * bs
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += bs

        pbar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "ce": f"{ce_val:.3f}",
            "tri": f"{tri_val:.4f}",
            "acc": f"{correct / max(total, 1):.3f}",
        })

    # Cosine scheduler steps per epoch (not per batch)
    if not isinstance(scheduler, GradualWarmupScheduler):
        scheduler.step()

    return {
        "loss": running_loss / max(total, 1),
        "ce": running_ce / max(total, 1),
        "tri": running_tri / max(total, 1),
        "acc": correct / max(total, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Market-1501 Pretraining V3")
    parser.add_argument("--market_dir", type=str,
                        default=str(PROJECT_DIR / "Market-1501-v15.09.15"),
                        help="Market-1501 dataset directory")
    parser.add_argument("--output", type=str,
                        default=str(SCRIPT_DIR / "train_output" / "market1501_v3"),
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--batch_p", type=int, default=16, help="P: IDs per batch")
    parser.add_argument("--batch_k", type=int, default=4, help="K: images per ID")
    parser.add_argument("--lr", type=float, default=3.5e-4, help="Base learning rate")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_p * args.batch_k
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Stage 1: Market-1501 Pretraining (IBNetResNet50 V3)")
    print(f"  PK Sampler: P={args.batch_p}, K={args.batch_k}, Batch={batch_size}")
    print(f"  LR={args.lr}, Warmup={args.warmup} epochs, Total={args.epochs} epochs")
    print("=" * 70)

    # ---- Dataset ----
    train_transform = get_train_transforms_v3(256, 128)
    test_transform = get_test_transforms_v3(256, 128)

    train_dataset = Market1501Dataset(args.market_dir, transform=train_transform, is_train=True)
    test_dataset = Market1501Dataset(args.market_dir, transform=test_transform, is_train=False)

    num_classes = train_dataset.num_classes
    print(f"\n[Data] Train: {len(train_dataset)} images, {num_classes} IDs")
    print(f"[Data] Test:  {len(test_dataset)} images, {test_dataset.num_classes} IDs")

    # PK Sampler
    pk_sampler = RandomIdentitySampler(train_dataset, num_instances=args.batch_k)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=pk_sampler,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # ---- Model ----
    print(f"\n[Model] Creating IBNetResNet50 V3 (num_classes={num_classes})")
    model = create_model_v3(num_classes=num_classes, use_cbam=True, use_dilation=True,
                            arc_scale=30.0, arc_margin=0.3)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_params:,}")

    # ---- Loss ----
    # Stage 1: CE dominant, triplet weight=0.5
    criterion = create_loss_v3(num_classes=num_classes, margin=0.3, tri_weight=0.5,
                               label_smooth=0.1, tri_distance='euclidean')

    # ---- Optimizer ----
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    after_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup,
                                        eta_min=args.lr * 0.01)
    scheduler = GradualWarmupScheduler(
        optimizer, warmup_epochs=args.warmup, after_scheduler=after_scheduler
    )
    # Set warmup steps after loader is created
    scheduler.set_warmup_steps(len(train_loader))
    scaler = GradScaler('cuda')

    # ---- Resume ----
    start_epoch = 0
    best_mAP = 0.0
    history = {"train_loss": [], "train_acc": [], "val_mAP": [], "val_rank1": []}

    if args.resume:
        print(f"\n[Resume] Loading: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        best_mAP = ckpt.get("best_mAP", 0.0)
        history = ckpt.get("history", history)
        print(f"  Resuming from epoch {start_epoch}, best mAP={best_mAP:.4f}")

    # ---- Training Loop ----
    log_path = output_dir / "training_log.txt"
    log_file = open(log_path, "w", encoding="utf-8")
    log_file.write(f"Stage 1: Market-1501 Pretraining V3\n")
    log_file.write(f"Model: IBNetResNet50 + CBAM + Dilated + GeM + ArcFace\n")
    log_file.write(f"PK: P={args.batch_p} K={args.batch_k} Batch={batch_size}\n")
    log_file.write(f"LR={args.lr} Warmup={args.warmup} Epochs={args.epochs}\n")
    log_file.write(f"Loss: LabelSmoothingCE(ε=0.1) + BatchHardTriplet(λ=0.5, m=0.3)\n")
    log_file.write("-" * 70 + "\n")
    log_file.flush()

    print(f"\n[Training] Start: epoch {start_epoch + 1} → {args.epochs}")
    print("-" * 70)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, scaler, epoch
        )
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])

        # Eval every 10 epochs
        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            mAP, rank1 = evaluate(model, test_loader, device)
            history["val_mAP"].append(mAP)
            history["val_rank1"].append(rank1)

            is_best = mAP > best_mAP
            if is_best:
                best_mAP = mAP
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_mAP": best_mAP,
                    "history": history,
                }, output_dir / "best_market1501_v3.pth")
                print(f"  >>> Best model: mAP={mAP:.4f}, Rank-1={rank1:.4f}")

            eta = (time.time() - t0) * (args.epochs - epoch) / 60
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"Train Loss:{train_metrics['loss']:.4f} Acc:{train_metrics['acc']:.3f} | "
                  f"Val mAP:{mAP:.4f} Rank-1:{rank1:.4f} | "
                  f"ETA:{eta:.0f}m")

            log_line = (f"Epoch {epoch:3d} | "
                       f"Train Loss:{train_metrics['loss']:.4f} Acc:{train_metrics['acc']:.3f} | "
                       f"Val mAP:{mAP:.4f} Rank-1:{rank1:.4f}\n")
            log_file.write(log_line)
            if is_best:
                log_file.write(f"  [BEST] mAP={mAP:.4f} Rank-1={rank1:.4f}\n")
        else:
            eta = (time.time() - t0) * (args.epochs - epoch) / 60
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"Train Loss:{train_metrics['loss']:.4f} Acc:{train_metrics['acc']:.3f} | "
                  f"ETA:{eta:.0f}m (eval @ every 10)")

        # Save checkpoint every 20 epochs
        if epoch % 20 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_mAP": best_mAP,
                "history": history,
            }, output_dir / f"checkpoint_epoch_{epoch:03d}.pth")

        log_file.flush()

    log_file.close()

    # ---- Final ----
    print(f"\n{'=' * 70}")
    print(f"  Stage 1 Complete! Best mAP={best_mAP:.4f}")
    print(f"  Model: {output_dir / 'best_market1501_v3.pth'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
