"""
============================================================
Stage 2: MOT17 All-Sequence Fine-tuning (IBNetResNet50 V3)
============================================================
加载 Stage 1 Market-1501 预训练权重，在全部 7 个 MOT17 序列上微调。

架构: IBNet50-a + CBAM + Dilated + GeM + ArcFace (margin=0.4)
输入: 384×128 (1.5× 分辨率提升)
损失: LabelSmoothingCE + BatchHardTriplet(λ=1.0)
采样: PK (6 IDs × 4 = BS 24) + Gradient Accumulation ×2
优化: 判别式学习率 + AdamW + CosineAnnealing
早停: Rank-1 连续 15 轮不提升
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

from model_v3 import create_model_v3, create_loss_v3
from train_reid import ReIDDataset

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
# PK Sampler
# ============================================================
class RandomIdentitySampler(Sampler):
    """PK 采样"""

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
# Transforms for 384×128
# ============================================================
def get_train_transforms_stage2(height=384, width=128):
    return T.Compose([
        T.Resize((height, width)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.15),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])


def get_test_transforms_stage2(height=384, width=128):
    return T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ============================================================
# Evaluation: ReID metrics (mAP, Rank-1/5/10/20)
# ============================================================
@torch.no_grad()
def extract_features(model, loader, device):
    """提取特征"""
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


def evaluate_reid(model, loader, device):
    """计算 mAP 和 CMC Rank-K (exclude self-match)"""
    feats, labels = extract_features(model, loader, device)
    if feats.size(0) == 0:
        return {"mAP": 0.0, "rank1": 0.0, "rank5": 0.0, "rank10": 0.0, "rank20": 0.0}

    # 欧氏距离矩阵
    N = feats.size(0)
    distmat = (
        torch.pow(feats, 2).sum(dim=1, keepdim=True).expand(N, N)
        + torch.pow(feats, 2).sum(dim=1, keepdim=True).expand(N, N).t()
    )
    distmat = distmat - 2 * torch.mm(feats, feats.t())
    distmat = distmat.clamp(min=1e-12).sqrt()

    labels_np = labels.numpy()
    indices = distmat.argsort(dim=1)

    aps = []
    cmc = np.zeros(20)

    for i in range(N):
        # 排除自身 (距离=0 的样本)
        sorted_idx = indices[i][1:]  # skip self
        sorted_labels = labels_np[sorted_idx.cpu().numpy()]

        matches_all = (sorted_labels == labels_np[i])
        num_positives = (labels_np == labels_np[i]).sum() - 1  # exclude self

        if num_positives <= 0:
            aps.append(0.0)
            continue

        # CMC: first 20 ranks (after excluding self)
        cumsum = matches_all[:20].cumsum()
        for k in range(20):
            if cumsum[k] >= 1:
                cmc[k] += 1.0

        # mAP
        correct = 0.0
        precisions = []
        for k, match in enumerate(matches_all, 1):
            if match:
                correct += 1
                precisions.append(correct / k)
        aps.append(np.mean(precisions) if precisions else 0.0)

    mAP = np.mean(aps)
    cmc = cmc / N

    return {
        "mAP": mAP,
        "rank1": cmc[0] if len(cmc) >= 1 else 0.0,
        "rank5": cmc[4] if len(cmc) >= 5 else 0.0,
        "rank10": cmc[9] if len(cmc) >= 10 else 0.0,
        "rank20": cmc[19] if len(cmc) >= 20 else 0.0,
    }


# ============================================================
# Training
# ============================================================
def train_epoch_v3(model, loader, criterion, optimizer, device, scaler, epoch,
                   grad_accum=1):
    """训练一个 epoch (支持梯度累积)"""
    model.train()
    running_loss = 0.0
    running_ce = 0.0
    running_tri = 0.0
    correct = 0
    total = 0
    accum_count = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d}")
    optimizer.zero_grad()

    for imgs, labels, _ in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        with autocast('cuda'):
            logits, feat, _ = model(imgs, labels=labels)
            loss, (ce_val, tri_val) = criterion(logits, feat, labels)
            loss = loss / grad_accum  # normalize for accumulation

        scaler.scale(loss).backward()
        accum_count += 1

        if accum_count % grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        bs = imgs.size(0)
        running_loss += loss.item() * grad_accum * bs
        running_ce += ce_val * bs
        running_tri += tri_val * bs
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += bs

        pbar.set_postfix({
            "loss": f"{loss.item() * grad_accum:.3f}",
            "ce": f"{ce_val:.3f}",
            "tri": f"{tri_val:.4f}",
            "acc": f"{correct / max(total, 1):.3f}",
        })

    return {
        "loss": running_loss / max(total, 1),
        "ce": running_ce / max(total, 1),
        "tri": running_tri / max(total, 1),
        "acc": correct / max(total, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 2: MOT17 Fine-tuning V3")
    parser.add_argument("--mot17_dir", type=str,
                        default=str(PROJECT_DIR / "dataset" / "mot17_reid_all"),
                        help="MOT17 all-sequence ReID dataset")
    parser.add_argument("--pretrained", type=str,
                        default=str(SCRIPT_DIR / "train_output" / "market1501_v3" /
                                     "best_market1501_v3.pth"),
                        help="Stage 1 pretrained weights")
    parser.add_argument("--output", type=str,
                        default=str(SCRIPT_DIR / "train_output" / "mot17_v3"),
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--batch_p", type=int, default=6, help="P: IDs per batch")
    parser.add_argument("--batch_k", type=int, default=4, help="K: images per ID")
    parser.add_argument("--grad_accum", type=int, default=2,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate")
    parser.add_argument("--early_stop", type=int, default=15,
                        help="Early stop patience")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint")
    parser.add_argument("--eval_every", type=int, default=10,
                        help="Evaluate every N epochs")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_p * args.batch_k
    effective_bs = batch_size * args.grad_accum
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Stage 2: MOT17 All-Sequence Fine-tuning (IBNetResNet50 V3)")
    print(f"  Resolution: 384×128")
    print(f"  PK: P={args.batch_p} K={args.batch_k} BS={batch_size} "
          f"×{args.grad_accum} = Effective BS={effective_bs}")
    print(f"  LR={args.lr}, EarlyStop={args.early_stop}")
    print("=" * 70)

    # ---- Dataset ----
    train_transform = get_train_transforms_stage2(384, 128)
    test_transform = get_test_transforms_stage2(384, 128)

    train_dataset = ReIDDataset(
        str(args.mot17_dir), transform=train_transform, is_train=True
    )
    test_dataset = ReIDDataset(
        str(args.mot17_dir), transform=test_transform, is_train=False,
        pid_to_idx=None,  # 独立建映射 (跨域)
    )

    num_classes = train_dataset.num_classes

    # Detect cross-domain
    train_pids = set(train_dataset.pid_to_idx.keys())
    test_pids = set(test_dataset.pid_to_idx.keys())
    pid_overlap = train_pids & test_pids
    cross_domain = len(pid_overlap) == 0

    print(f"\n[Data] Train: {len(train_dataset)} images, {num_classes} IDs")
    print(f"[Data] Val:   {len(test_dataset)} images, {test_dataset.num_classes} IDs")
    print(f"[Data] PID overlap: {len(pid_overlap)}/{len(test_pids)} "
          f"({'Cross-domain' if cross_domain else 'Same-domain'})")

    # PK Sampler
    pk_sampler = RandomIdentitySampler(train_dataset, num_instances=args.batch_k)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=pk_sampler,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # ---- Model ----
    print(f"\n[Model] Creating IBNetResNet50 V3 (num_classes={num_classes})")
    model = create_model_v3(
        num_classes=num_classes, use_cbam=True, use_dilation=True,
        arc_scale=30.0, arc_margin=0.4  # ★ harder margin for similar IDs
    )

    # ---- Load Stage 1 weights ----
    pretrained_path = Path(args.pretrained)
    if pretrained_path.exists():
        print(f"[Weights] Loading Stage 1: {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location=device, weights_only=False)

        # Handle both checkpoint dict and raw state_dict
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        # Remove ArcFace weights (different num_classes)
        arc_keys = [k for k in state_dict if k.startswith("arcface.")]
        for k in arc_keys:
            print(f"  Removing: {k} (num_classes mismatch)")
            del state_dict[k]

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys (will init randomly): {len(missing)}")
            for k in missing[:3]:
                print(f"    - {k}")
        print(f"  [OK] Stage 1 weights loaded (backbone + GeM + BNNeck)")
    else:
        print(f"[WARN] Stage 1 weights not found: {pretrained_path}")
        print(f"  Training from scratch!")

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {total_params:,} total, {trainable_params:,} trainable")

    # ---- Discriminative Learning Rates ----
    # layer1: frozen (lr=0)
    # layer2: lr × 0.5
    # layer3: lr × 1.0
    # layer4: lr × 1.5
    # gem_pool + bottleneck + arcface: lr × 2.0
    param_groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("layer1"):
            # Freeze layer1
            param.requires_grad = False
            continue
        elif name.startswith("layer2"):
            lr_mult = 0.5
        elif name.startswith("layer3"):
            lr_mult = 1.0
        elif name.startswith("layer4"):
            lr_mult = 1.5
        else:
            # gem_pool, bottleneck, arcface, conv1, bn1, cbam
            lr_mult = 2.0

        param_groups.append({
            "params": param,
            "lr": args.lr * lr_mult,
            "name": name,
        })

    frozen_params = sum(1 for p in model.parameters() if not p.requires_grad)
    print(f"\n[LR Groups] Frozen params: {frozen_params}")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("layer2"):
            print(f"  layer2: lr={args.lr * 0.5:.1e}")
            break
    print(f"  layer3: lr={args.lr * 1.0:.1e}")
    print(f"  layer4: lr={args.lr * 1.5:.1e}")
    print(f"  head:   lr={args.lr * 2.0:.1e}")

    # ---- Loss (triplet weight=1.0 for Stage 2) ----
    criterion = create_loss_v3(
        num_classes=num_classes, margin=0.3, tri_weight=1.0,
        label_smooth=0.1, tri_distance='euclidean'
    )

    # ---- Optimizer ----
    optimizer = optim.AdamW(param_groups, lr=args.lr, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.001)
    scaler = GradScaler('cuda')

    # ---- Resume ----
    start_epoch = 0
    best_rank1 = 0.0
    best_epoch = 0
    best_metrics = None
    early_stop_counter = 0
    history = {
        "train_loss": [], "train_acc": [],
        "val_mAP": [], "val_rank1": [], "val_rank5": [],
        "val_rank10": [], "val_rank20": [],
    }

    if args.resume:
        print(f"\n[Resume] {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        best_rank1 = ckpt.get("best_rank1", 0.0)
        best_epoch = ckpt.get("best_epoch", start_epoch)
        history = ckpt.get("history", history)
        early_stop_counter = ckpt.get("early_stop_counter", 0)
        print(f"  Resuming from epoch {start_epoch}, best Rank-1={best_rank1:.4f}")

    # ---- Training Loop ----
    log_path = output_dir / "training_log.txt"
    log_file = open(log_path, "w", encoding="utf-8")
    log_file.write(f"Stage 2: MOT17 All-Sequence Fine-tuning V3\n")
    log_file.write(f"Model: IBNetResNet50 + CBAM + Dilated + GeM + ArcFace(m=0.4)\n")
    log_file.write(f"Input: 384×128\n")
    log_file.write(f"PK: P={args.batch_p} K={args.batch_k} BS={batch_size} "
                   f"×{args.grad_accum} = EffBS={effective_bs}\n")
    log_file.write(f"LR={args.lr} (discriminative), EarlyStop={args.early_stop}\n")
    log_file.write(f"Data: {len(train_dataset)} train, {len(test_dataset)} val\n")
    log_file.write("-" * 70 + "\n")
    log_file.flush()

    print(f"\n[Training] Epoch {start_epoch + 1} → {args.epochs}")
    print("-" * 70)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_metrics = train_epoch_v3(
            model, train_loader, criterion, optimizer, device, scaler, epoch,
            grad_accum=args.grad_accum
        )
        scheduler.step()
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])

        # Eval
        do_eval = (epoch % args.eval_every == 0) or (epoch == 1) or (epoch == args.epochs)
        if do_eval:
            reid_metrics = evaluate_reid(model, test_loader, device)
            history["val_mAP"].append(reid_metrics["mAP"])
            history["val_rank1"].append(reid_metrics["rank1"])
            history["val_rank5"].append(reid_metrics["rank5"])
            history["val_rank10"].append(reid_metrics["rank10"])
            history["val_rank20"].append(reid_metrics["rank20"])

            current_rank1 = reid_metrics["rank1"]
            eta = (time.time() - t0) * (args.epochs - epoch) / 60

            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"Train Loss:{train_metrics['loss']:.4f} Acc:{train_metrics['acc']:.3f} | "
                  f"R1:{reid_metrics['rank1']:.4f} R5:{reid_metrics['rank5']:.4f} "
                  f"mAP:{reid_metrics['mAP']:.4f} | ETA:{eta:.0f}m")

            log_line = (f"Epoch {epoch:3d} | "
                       f"Train Loss:{train_metrics['loss']:.4f} Acc:{train_metrics['acc']:.3f} | "
                       f"R1:{reid_metrics['rank1']:.4f} R5:{reid_metrics['rank5']:.4f} "
                       f"mAP:{reid_metrics['mAP']:.4f}\n")
            log_file.write(log_line)

            # Best model
            if current_rank1 > best_rank1 + 1e-6:
                best_rank1 = current_rank1
                best_epoch = epoch
                best_metrics = reid_metrics
                early_stop_counter = 0

                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_rank1": best_rank1,
                    "best_epoch": best_epoch,
                    "history": history,
                }, output_dir / "best_mot17_v3.pth")
                print(f"  >>> [BEST] Rank-1={best_rank1:.4f} mAP={reid_metrics['mAP']:.4f}")
                log_file.write(f"  [BEST] Rank-1={best_rank1:.4f} mAP={reid_metrics['mAP']:.4f}\n")
            else:
                early_stop_counter += 1
                if not do_eval:
                    pass  # only count eval epochs
        else:
            eta = (time.time() - t0) * (args.epochs - epoch) / 60
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"Train Loss:{train_metrics['loss']:.4f} Acc:{train_metrics['acc']:.3f} | "
                  f"ETA:{eta:.0f}m (eval @ {args.eval_every} ep)")

        # Early stop (only based on eval epochs)
        if do_eval and early_stop_counter >= args.early_stop:
            print(f"\n  Early stop triggered! Rank-1 stagnant for {args.early_stop} evals")
            print(f"  Best: epoch {best_epoch}, Rank-1={best_rank1:.4f}")
            break

        # Checkpoint every 20 epochs
        if epoch % 20 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_rank1": best_rank1,
                "best_epoch": best_epoch,
                "history": history,
                "early_stop_counter": early_stop_counter,
            }, output_dir / f"checkpoint_epoch_{epoch:03d}.pth")

        log_file.flush()

    log_file.close()

    # ---- Final ----
    final_path = output_dir / "final_mot17_v3.pth"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_rank1": best_rank1,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
    }, final_path)

    print(f"\n{'=' * 70}")
    print(f"  Stage 2 Complete!")
    print(f"  Best: epoch {best_epoch}, Rank-1={best_rank1:.4f}")
    if best_metrics:
        print(f"  mAP={best_metrics['mAP']:.4f} "
              f"R5={best_metrics['rank5']:.4f} "
              f"R10={best_metrics['rank10']:.4f}")
    print(f"  Model: {output_dir / 'best_mot17_v3.pth'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
