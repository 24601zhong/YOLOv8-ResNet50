"""
============================================================
改进 ResNet50 行人重识别微调训练脚本: resnet50_mot17_finetune.py
============================================================
功能:
  1. 加载 Stage1 Market-1501 预训练权重
  2. 在 MOT17 室内清洗数据上全解冻微调
  3. 联合损失: CrossEntropy + TripletLoss (hard mining, margin=0.3, λ=1.0)
  4. 每轮评估 Rank-1 / Rank-5 / mAP,绘制 t-SNE 特征图
  5. 早停: Rank-1 连续 12 轮不提升停止
  6. 每 10 轮保存断点,支持断点续训
  7. 异常捕获: 路径/显存/权重加载

网络固定结构:
  - ResNet50 骨干
  - Layer3/Layer4 嵌入 CBAM 注意力
  - Layer3 空洞卷积 d=2, Layer4 空洞卷积 d=4
  - 输入尺寸: 256x128

训练超参:
  - Epochs=80, Adam lr=1e-4, 余弦退火, batch=16
============================================================
"""

import os
import sys
import time
import random
import warnings
import traceback
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# [*] 全局路径配置（Windows 反斜杠兼容）
# ============================================================
BASE_DIR = Path(__file__).parent.parent.resolve()  # 脚本在 resnet50_reid_train/,项目根为上级目录

# 预训练权重路径 (Stage1 Market-1501 训练产出)
PRETRAINED_WEIGHTS = (
    BASE_DIR / "resnet50_reid_train"
            / "train_output" / "reid_log" / "best_market1501_improved.pth"
)

# MOT17 清洗后 ReID 数据集路径
MOT17_REID_DIR = BASE_DIR / "dataset" / "mot17_reid_clean"

# 微调训练日志/权重输出路径
OUTPUT_DIR = BASE_DIR / "resnet50_reid_train" / "train_output" / "mot17_finetune_log"

# ============================================================
# [*] 训练超参（硬性约束）
# ============================================================
NUM_EPOCHS = 80
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
TRI_MARGIN = 0.3
TRI_WEIGHT = 1.0
INPUT_SIZE = (256, 128)  # (H, W)
EARLY_STOP_PATIENCE = 12  # Rank-1 连续 12 轮不提升停止
CHECKPOINT_INTERVAL = 10  # 每 10 轮保存断点
DEVICE_ID = 0             # GPU 0
RANDOM_SEED = 42

# ============================================================
# 随机种子
# ============================================================
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ============================================================
# 导入现有模型架构（同目录下的 model.py 和 train_reid.py）
# ============================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from model import create_model, ReIDLoss
    from train_reid import ReIDDataset, get_train_transforms, get_test_transforms
except ImportError as e:
    print(f"[错误] 无法导入模型或数据集类: {e}")
    print(f"  请确认 model.py 和 train_reid.py 在以下目录:")
    print(f"  {SCRIPT_DIR}")
    sys.exit(1)


# ============================================================
# ReID 评估工具函数:Rank-K / mAP / CMC
# ============================================================
@torch.no_grad()
def extract_features(model, loader, device):
    """从数据加载器提取所有图像特征向量和标签
    Args:
        model: ReID 模型
        loader: DataLoader
        device: torch device
    Returns:
        features: [N, 2048] 特征矩阵
        labels: [N] 标签向量
        paths: [N] 图像路径列表
    """
    model.eval()
    all_features = []
    all_labels = []
    all_paths = []

    for imgs, labels, paths in tqdm(loader, desc="  提取特征", unit="batch"):
        imgs = imgs.to(device)
        feats = model(imgs, return_feature=True)
        all_features.append(feats.cpu())
        all_labels.append(labels)
        all_paths.extend(paths)

    if len(all_features) == 0:
        return (
            torch.empty(0, 2048),
            torch.empty(0, dtype=torch.long),
            [],
        )

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return features, labels, all_paths


def compute_distance_matrix(query_feats, gallery_feats, metric="euclidean"):
    """计算 query-gallery 距离矩阵
    Args:
        query_feats: [Nq, D]
        gallery_feats: [Ng, D]
        metric: "euclidean" or "cosine"
    Returns:
        distmat: [Nq, Ng]
    """
    if metric == "cosine":
        # 余弦距离 = 1 - 余弦相似度
        q_norm = nn.functional.normalize(query_feats, p=2, dim=1)
        g_norm = nn.functional.normalize(gallery_feats, p=2, dim=1)
        distmat = 1.0 - torch.mm(q_norm, g_norm.t())
    else:
        # 欧氏距离
        m, n = query_feats.size(0), gallery_feats.size(0)
        distmat = (
            torch.pow(query_feats, 2).sum(dim=1, keepdim=True).expand(m, n)
            + torch.pow(gallery_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        )
        distmat = distmat - 2 * torch.mm(query_feats, gallery_feats.t())
        distmat = distmat.clamp(min=1e-12).sqrt()
    return distmat


def evaluate_reid_metrics(model, loader, device, num_ranks=20):
    """ReID 核心评估: Rank-K / mAP
    使用 single-query 模式,gallery 和 query 为同一集合（排除自身匹配）

    Args:
        model: ReID 模型
        loader: 数据加载器（val set）
        device: torch device
        num_ranks: 计算的 Rank-K 最大 K 值

    Returns:
        metrics: dict {
            "rank1", "rank5", "rank10", "rank20", "mAP",
            "cmc": [Nq, num_ranks] CMC 矩阵
        }
    """
    # Step 1: 提取特征
    features, labels, _ = extract_features(model, loader, device)

    if features.size(0) == 0:
        return {"rank1": 0.0, "rank5": 0.0, "rank10": 0.0, "rank20": 0.0, "mAP": 0.0}

    # Step 2: 计算距离矩阵
    distmat = compute_distance_matrix(features, features, metric="euclidean")
    # distmat[i][j] = distance between query_i and gallery_j

    num_q = distmat.size(0)
    num_g = distmat.size(1)
    labels_np = labels.numpy()

    # Step 3: 按距离排序
    indices = distmat.argsort(dim=1)  # [Nq, Ng]

    # Step 4: 对每个 query 计算 AP 和 CMC
    aps = []
    cmc = torch.zeros(num_q, num_ranks)

    for i in range(num_q):
        # 有效 gallery: 排除自身
        valid_gallery = (indices[i] != i)

        sorted_indices = indices[i][valid_gallery][:num_ranks]
        sorted_labels = labels_np[sorted_indices.cpu().numpy()]

        # 匹配: 与 query 标签相同的 gallery 图像
        matches = (sorted_labels == labels_np[i])

        # CMC: rank-k 命中
        cumsum = matches.cumsum()
        for k in range(num_ranks):
            cmc[i, k] = 1.0 if cumsum[k] >= 1 else 0.0

        # AP: average precision
        # 统计所有正样本（不包括自身）
        all_positives = (labels_np == labels_np[i])
        all_positives[i] = False  # 排除自身
        num_positives = all_positives.sum()

        if num_positives == 0:
            aps.append(0.0)
            continue

        # 对所有 gallery（排除自身）计算 precision@k
        valid_indices_all = indices[i][indices[i] != i]
        sorted_labels_all = labels_np[valid_indices_all.cpu().numpy()]
        matches_all = (sorted_labels_all == labels_np[i])

        precisions = []
        correct = 0.0
        for k_idx, match in enumerate(matches_all, 1):
            if match:
                correct += 1
                precisions.append(correct / k_idx)
        aps.append(np.mean(precisions) if precisions else 0.0)

    mAP = np.mean(aps)

    # CMC 均值
    cmc_avg = cmc.mean(dim=0)  # [num_ranks]
    rank1 = cmc_avg[0].item() if num_ranks >= 1 else 0.0
    rank5 = cmc_avg[4].item() if num_ranks >= 5 else 0.0
    rank10 = cmc_avg[9].item() if num_ranks >= 10 else 0.0
    rank20 = cmc_avg[19].item() if num_ranks >= 20 else 0.0

    return {
        "rank1": rank1,
        "rank5": rank5,
        "rank10": rank10,
        "rank20": rank20,
        "mAP": mAP,
        "cmc": cmc_avg.cpu().numpy() if cmc_avg.is_cuda else cmc_avg.numpy(),
    }


# ============================================================
# t-SNE 特征可视化
# ============================================================
def plot_tsne_features(model, loader, device, output_dir, epoch, max_samples=2000):
    """绘制 t-SNE 特征分布图
    Args:
        model: ReID 模型
        loader: 数据加载器
        device: torch device
        output_dir: 图片输出目录
        epoch: 当前 epoch（用于标题）
        max_samples: 最大采样数（避免 t-SNE 过慢）
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  [跳过] sklearn 未安装,无法绘制 t-SNE")
        return

    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(device)
            feats = model(imgs, return_feature=True)
            all_features.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())

    if not all_features:
        return

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # 限制样本数以加速 t-SNE
    if len(features) > max_samples:
        indices = np.random.choice(len(features), max_samples, replace=False)
        features = features[indices]
        labels = labels[indices]

    n_samples = len(features)
    if n_samples < 5:
        return

    # t-SNE 降维
    perplexity = min(30, max(5, n_samples // 10))
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED,
                perplexity=perplexity, n_iter=1000, verbose=0)
    features_2d = tsne.fit_transform(features)

    # 绘图
    unique_labels = np.unique(labels)
    n_colors = len(unique_labels)

    fig, ax = plt.subplots(figsize=(12, 10))

    # 使用 tab20 颜色映射
    cmap = plt.cm.get_cmap("tab20", max(n_colors, 20))
    colors = [cmap(i % 20) for i in range(n_colors)]

    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   c=[colors[i]], label=f"ID {lbl}", s=8, alpha=0.7)

    ax.set_title(f"t-SNE Feature Visualization (Epoch {epoch})", fontsize=14)
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")

    # 如果类别太多,不显示 legend
    if n_colors <= 20:
        ax.legend(loc="upper right", fontsize=6, ncol=2,
                  markerscale=2, framealpha=0.5)
    else:
        ax.text(0.02, 0.98,
                f"Total {n_colors} IDs (legend suppressed)",
                transform=ax.transAxes, fontsize=9, va="top")

    plt.tight_layout()
    save_path = output_dir / f"tsne_epoch_{epoch:03d}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  t-SNE 图已保存: {save_path}")


# ============================================================
# 训练一个 epoch
# ============================================================
def train_epoch(model, loader, criterion, optimizer, scheduler, device, scaler, epoch, num_classes=None):
    """训练一个 epoch
    Returns:
        metrics: loss, ce, tri, acc, precision, recall, f1
    """
    model.train()
    running_loss = 0.0
    running_ce = 0.0
    running_tri = 0.0
    correct = 0
    total = 0

    # 累积混淆统计（用于计算 Precision/Recall/F1）
    if num_classes is not None:
        tp = torch.zeros(num_classes, device=device, dtype=torch.long)
        fp = torch.zeros(num_classes, device=device, dtype=torch.long)
        fn = torch.zeros(num_classes, device=device, dtype=torch.long)
    else:
        tp = fp = fn = None

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [Train]", unit="batch")
    for imgs, labels, _ in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        with autocast("cuda"):
            logits, features = model(imgs)
            loss, (ce, tri) = criterion(logits, features, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = imgs.size(0)
        running_loss += loss.item() * bs
        running_ce += ce.item() * bs
        running_tri += tri.item() * bs
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += bs

        # 累积逐类 TP/FP/FN（macro-average）
        if tp is not None:
            for c in range(num_classes):
                pred_c = (preds == c)
                true_c = (labels == c)
                tp[c] += (pred_c & true_c).sum()
                fp[c] += (pred_c & ~true_c).sum()
                fn[c] += (~pred_c & true_c).sum()

        # 动态计算当前宏平均指标
        pbar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "ce": f"{ce.item():.3f}",
            "tri": f"{tri.item():.3f}",
            "acc": f"{correct / max(total, 1):.3f}",
        })

    scheduler.step()

    # 计算宏平均 Precision / Recall / F1
    if tp is not None and tp.sum() > 0:
        # 避免除零：按类计算后取平均
        eps = 1e-8
        per_class_precision = tp.float() / (tp + fp).float().clamp(min=eps)
        per_class_recall = tp.float() / (tp + fn).float().clamp(min=eps)
        # 仅对实际出现的类别取平均
        active_classes = ((tp + fn) > 0).float()
        macro_precision = (per_class_precision * active_classes).sum() / active_classes.sum().clamp(min=1)
        macro_recall = (per_class_recall * active_classes).sum() / active_classes.sum().clamp(min=1)
        macro_f1 = 2.0 * macro_precision * macro_recall / (macro_precision + macro_recall + eps)
        macro_precision = macro_precision.item()
        macro_recall = macro_recall.item()
        macro_f1 = macro_f1.item()
    else:
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0

    return {
        "loss": running_loss / max(total, 1),
        "ce": running_ce / max(total, 1),
        "tri": running_tri / max(total, 1),
        "acc": correct / max(total, 1),
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
    }


# ============================================================
# 验证函数（含 ReID 指标）
# ============================================================
@torch.no_grad()
def validate_epoch(model, loader, criterion, device, train_num_classes, cross_domain=False):
    """验证一个 epoch: Loss + ReID 指标
    Args:
        cross_domain: 显式跨域标记（训练/验证 ID 不重叠）,仅用 triplet + 距离指标
    """
    model.eval()
    running_loss = 0.0
    running_ce = 0.0
    running_tri = 0.0
    correct = 0
    total = 0

    for imgs, labels, _ in tqdm(loader, desc="Validating", unit="batch"):
        imgs, labels = imgs.to(device), labels.to(device)

        logits, features = model(imgs)

        # 跨域检测:显式标记 OR 标签越界检测
        if cross_domain or (labels.max().item() >= train_num_classes):
            cross_domain = True
            from model import TripletLoss
            tri_fn = TripletLoss(margin=TRI_MARGIN).to(device)
            tri = tri_fn(features, labels)
            loss = tri
            ce_val = 0.0
            tri_val = tri.item()
        else:
            loss, (ce_val, tri_val) = criterion(logits, features, labels)
            _, preds = logits.max(1)
            correct += preds.eq(labels).sum().item()

        bs = imgs.size(0)
        running_loss += loss.item() * bs
        running_ce += ce_val * bs if isinstance(ce_val, float) else ce_val.item() * bs
        running_tri += tri_val * bs if isinstance(tri_val, float) else tri_val.item() * bs
        total += bs

    if total == 0:
        return {"loss": 0.0, "ce": 0.0, "tri": 0.0, "acc": 0.0,
                "rank1": 0.0, "rank5": 0.0, "mAP": 0.0}

    base_metrics = {
        "loss": running_loss / total,
        "ce": running_ce / total,
        "tri": running_tri / total,
        "acc": correct / max(total, 1),
    }

    # ReID 评估指标 (Rank-K / mAP)
    reid_metrics = evaluate_reid_metrics(model, loader, device, num_ranks=20)
    base_metrics.update(reid_metrics)

    return base_metrics


# ============================================================
# 绘制训练曲线
# ============================================================
def plot_training_curves(history, output_dir):
    """绘制 Loss、Rank-1、mAP 训练曲线"""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=1.5)
    ax.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, history["train_acc"], "b-", label="Train Acc", linewidth=1.5)
    if history.get("precision") and len(history["precision"]) == len(epochs):
        ax.plot(epochs, history["precision"], "g-", label="Precision", linewidth=1.5)
        ax.plot(epochs, history["recall"], "orange", label="Recall", linewidth=1.5)
        ax.plot(epochs, history["f1"], "purple", label="F1-Score", linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_title("Train: Acc / Precision / Recall / F1")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.plot(epochs, history["val_tri"], "m-", label="Val Triplet Loss", linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Triplet Loss")
    ax.set_title("Validation Triplet Loss")
    ax.legend(); ax.grid(alpha=0.3)

    # Row 2: ReID metrics
    ax = axes[1, 0]
    ax.plot(epochs, history["rank1"], "g-", label="Rank-1", linewidth=1.5)
    ax.plot(epochs, history["rank5"], "orange", label="Rank-5", linewidth=1.5)
    ax.plot(epochs, history["rank10"], "c-", label="Rank-10", linewidth=1.2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_title("CMC Rank-K Accuracy")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, history["mAP"], "purple", label="mAP", linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("mAP")
    ax.set_title("Mean Average Precision")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.plot(epochs, history["lr"], "brown", label="Learning Rate", linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
    ax.set_title("Learning Rate Schedule")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "training_curves_mot17_finetune.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  训练曲线已保存: {save_path}")


# ============================================================
# 保存训练结果
# ============================================================
def save_results(history, best_metrics, best_epoch, total_params, output_dir):
    """保存训练结果文本报告"""
    results_path = output_dir / "results_mot17_finetune.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("ResNet50 MOT17 微调训练结果\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"预训练权重: {PRETRAINED_WEIGHTS}\n")
        f.write(f"训练数据:   {MOT17_REID_DIR}\n")
        f.write(f"总参数量:   {total_params:,}\n")
        f.write(f"训练轮数:   {len(history['train_loss'])} / {NUM_EPOCHS} (计划)\n")
        f.write(f"批次大小:   {BATCH_SIZE}\n")
        f.write(f"初始学习率: {LEARNING_RATE}\n\n")

        f.write(f"最佳轮次:   Epoch {best_epoch}\n")
        f.write(f"最佳 Rank-1: {best_metrics['rank1']:.4f}\n")
        f.write(f"最佳 Rank-5: {best_metrics['rank5']:.4f}\n")
        f.write(f"最佳 mAP:    {best_metrics['mAP']:.4f}\n")
        f.write(f"对应 Val Loss: {best_metrics['loss']:.4f}\n")

        if len(history["rank1"]) > 0:
            f.write(f"\n最终轮次 Rank-1: {history['rank1'][-1]:.4f}\n")
            f.write(f"最终轮次 mAP:    {history['mAP'][-1]:.4f}\n")

    print(f"\n  结果报告已保存: {results_path}")


# ============================================================
# 保存 / 加载检查点
# ============================================================
def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_metrics,
                    history, output_dir, is_best=False):
    """保存训练检查点"""
    if is_best:
        ckpt_path = output_dir / "best_mot17_finetune.pth"
        # 仅保存模型权重（兼容标准加载）
        torch.save(model.state_dict(), ckpt_path)
        print(f"  >>> 最佳模型已保存: {ckpt_path}")
    else:
        ckpt_path = output_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_metrics": best_metrics,
            "best_epoch": best_metrics.get("_epoch", epoch),
            "history": history,
        }, ckpt_path)
        print(f"  检查点已保存: {ckpt_path}")

    return ckpt_path


def load_checkpoint(model, optimizer, scheduler, scaler, output_dir, device):
    """加载最新的训练检查点
    Returns:
        (start_epoch, best_metrics, history) or (0, None, empty_history) if no checkpoint
    """
    # 查找所有检查点文件
    ckpt_files = sorted(output_dir.glob("checkpoint_epoch_*.pth"))
    if not ckpt_files:
        return 0, None, _init_history()

    latest_ckpt = ckpt_files[-1]
    print(f"\n[续训] 发现检查点: {latest_ckpt}")

    try:
        ckpt = torch.load(latest_ckpt, map_location=device)
    except Exception as e:
        print(f"  [警告] 检查点加载失败: {e},将从头训练")
        return 0, None, _init_history()

    # 恢复模型权重
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except Exception as e:
        print(f"  [警告] 模型权重加载异常: {e}")
        print(f"  将尝试继续训练（可能使用未完全恢复的权重）")

    # 恢复优化器 & 调度器
    try:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    except Exception as e:
        print(f"  [警告] 优化器/调度器状态加载异常: {e}")
        print(f"  将从当前状态继续")

    start_epoch = ckpt.get("epoch", 0)
    best_metrics = ckpt.get("best_metrics", None)
    history = ckpt.get("history", _init_history())

    print(f"  续训起始轮次: Epoch {start_epoch + 1}")
    if best_metrics:
        print(f"  历史最佳 Rank-1: {best_metrics.get('rank1', 'N/A')}")

    return start_epoch, best_metrics, history


def _init_history():
    """初始化训练历史记录"""
    return {
        "train_loss": [], "train_acc": [],
        "precision": [], "recall": [], "f1": [],
        "val_loss": [], "val_acc": [], "val_tri": [],
        "rank1": [], "rank5": [], "rank10": [], "rank20": [],
        "mAP": [], "lr": [],
    }


# ============================================================
# 主训练函数
# ============================================================
def train_mot17_finetune():
    """MOT17 微调训练主函数"""
    print("=" * 70)
    print("  改进 ResNet50 -- MOT17 室内微调训练")
    print("  CBAM 注意力 + 空洞卷积 + Joint Loss")
    print("=" * 70)

    # ================================================================
    # [*] 异常捕获块 1: 路径检查
    # ================================================================
    print(f"\n[路径检查]")

    # 检查预训练权重
    pretrained_path = Path(PRETRAINED_WEIGHTS)
    if not pretrained_path.exists():
        print(f"  [错误] 预训练权重不存在: {pretrained_path}")
        print(f"  请先完成 Stage1 Market-1501 训练。")
        sys.exit(1)
    print(f"  [OK] 预训练权重: {pretrained_path} ({pretrained_path.stat().st_size / 1024**2:.1f} MB)")

    # 检查 MOT17 清洗数据
    train_img_dir = Path(MOT17_REID_DIR) / "bounding_box_train"
    test_img_dir = Path(MOT17_REID_DIR) / "bounding_box_test"
    if not train_img_dir.exists():
        print(f"  [错误] MOT17 训练集不存在: {train_img_dir}")
        print(f"  请先运行 mot17_reid_data_clean.py 清洗数据。")
        sys.exit(1)
    if not test_img_dir.exists():
        print(f"  [错误] MOT17 验证集不存在: {test_img_dir}")
        print(f"  请先运行 mot17_reid_data_clean.py 清洗数据。")
        sys.exit(1)

    train_jpg_count = len(list(train_img_dir.glob("*.jpg")))
    test_jpg_count = len(list(test_img_dir.glob("*.jpg")))
    print(f"  [OK] 训练集: {train_img_dir} ({train_jpg_count} 张)")
    print(f"  [OK] 验证集: {test_img_dir} ({test_jpg_count} 张)")

    if train_jpg_count == 0:
        print(f"  [错误] 训练集为空! 请检查数据清洗流程。")
        sys.exit(1)

    # 创建输出目录
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [OK] 输出目录: {output_dir}")

    # ================================================================
    # [*] 异常捕获块 2: 设备/显存检查
    # ================================================================
    print(f"\n[设备检查]")
    if not torch.cuda.is_available():
        print(f"  [错误] CUDA 不可用! 本训练需要 GPU。")
        print(f"  请确认 PyTorch CUDA 版本安装正确。")
        sys.exit(1)

    device_count = torch.cuda.device_count()
    print(f"  可用 GPU 数量: {device_count}")

    if DEVICE_ID >= device_count:
        print(f"  [错误] GPU {DEVICE_ID} 不可用 (可用: 0~{device_count - 1})")
        sys.exit(1)

    device = torch.device(f"cuda:{DEVICE_ID}")
    gpu_name = torch.cuda.get_device_name(DEVICE_ID)
    gpu_mem_total = torch.cuda.get_device_properties(DEVICE_ID).total_memory / 1024**3
    print(f"  [OK] 使用 GPU {DEVICE_ID}: {gpu_name} ({gpu_mem_total:.1f} GB)")

    # 显存估算
    estimated_mem = BATCH_SIZE * 3 * 256 * 128 * 4 / 1024**3  # ~0.05 GB 仅输入
    print(f"  预估输入显存: ~{estimated_mem:.2f} GB (batch={BATCH_SIZE})")

    try:
        # 快速显存压力测试（小 batch）
        test_tensor = torch.randn(4, 3, *INPUT_SIZE).to(device)
        del test_tensor
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"  [错误] GPU 显存不足: {e}")
        print(f"  建议减小 BATCH_SIZE 或清理 GPU 显存。")
        sys.exit(1)

    # ================================================================
    # 数据集 & DataLoader
    # ================================================================
    print(f"\n[数据加载]")

    train_transform = get_train_transforms(height=INPUT_SIZE[0], width=INPUT_SIZE[1])
    test_transform = get_test_transforms(height=INPUT_SIZE[0], width=INPUT_SIZE[1])

    try:
        train_dataset = ReIDDataset(
            str(MOT17_REID_DIR), transform=train_transform, is_train=True
        )
        # [*] 跨域微调:训练/验证人员 ID 不重叠,验证集独立建 PID 映射
        test_dataset = ReIDDataset(
            str(MOT17_REID_DIR), transform=test_transform, is_train=False,
            pid_to_idx=None,  # 独立建映射（跨域评估场景）
        )
    except Exception as e:
        print(f"  [错误] 数据集加载失败: {e}")
        traceback.print_exc()
        sys.exit(1)

    num_classes = train_dataset.num_classes

    # [*] 检测跨域:比较训练集/验证集的 PID 集合是否重叠
    train_pids = set(train_dataset.pid_to_idx.keys())
    test_pids = set(test_dataset.pid_to_idx.keys())
    pid_overlap = train_pids & test_pids
    cross_domain = len(pid_overlap) == 0

    print(f"  训练集: {len(train_dataset)} 张, {num_classes} 个 ID")
    print(f"  验证集: {len(test_dataset)} 张, {test_dataset.num_classes} 个 ID")
    print(f"  PID 重叠: {len(pid_overlap)}/{len(test_pids)}")
    if cross_domain:
        print(f"  [*] 跨域微调:训练/验证人员 ID 不重叠,验证仅用 triplet + ReID 距离指标")
    else:
        print(f"  [*] 同域微调:训练/验证人员 ID 有重叠,验证含 CE loss")
    print(f"  有效验证 ID: {len(test_pids & train_pids)} (与训练重叠)")

    if len(test_dataset) == 0:
        print(f"  [错误] 验证集为空! (可能 PID 映射失败)")
        sys.exit(1)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # ================================================================
    # [*] 异常捕获块 3: 模型加载 & 权重加载
    # ================================================================
    print(f"\n[模型加载]")
    try:
        model = create_model(
            num_classes=num_classes,
            use_cbam=True,
            use_dilation=True,
        )
    except Exception as e:
        print(f"  [错误] 模型创建失败: {e}")
        traceback.print_exc()
        sys.exit(1)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量:     {total_params:,}")
    print(f"  可训练参数:   {trainable_params:,}")

    # 加载 Market-1501 预训练权重
    print(f"\n[权重加载] 正在加载 Market-1501 预训练权重...")
    try:
        state_dict = torch.load(pretrained_path, map_location=device)
    except Exception as e:
        print(f"  [错误] 权重文件读取失败: {e}")
        print(f"  文件路径: {pretrained_path}")
        sys.exit(1)

    # 处理分类器维度不匹配（Market-1501 有 751 类,MOT17 人数不同）
    if "classifier.weight" in state_dict:
        pretrained_num_classes = state_dict["classifier.weight"].size(0)
        if pretrained_num_classes != num_classes:
            print(f"  [信息] 分类器维度不匹配: "
                  f"预训练 {pretrained_num_classes} 类 → MOT17 {num_classes} 类")
            print(f"  [信息] 移除预训练分类器权重 (保留骨干网络权重)")
            del state_dict["classifier.weight"]
            if "classifier.bias" in state_dict:
                del state_dict["classifier.bias"]

    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"  [信息] 缺失的键 (将随机初始化): {len(missing_keys)} 个")
            for k in missing_keys[:5]:
                print(f"    - {k}")
            if len(missing_keys) > 5:
                print(f"    ... 还有 {len(missing_keys) - 5} 个")
        if unexpected_keys:
            print(f"  [信息] 未使用的键: {len(unexpected_keys)} 个")
    except Exception as e:
        print(f"  [错误] 权重加载异常: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"  [OK] 预训练权重加载成功")

    # [*] 完整解冻网络
    for param in model.parameters():
        param.requires_grad = True
    print(f"  [OK] 网络已完全解冻 (fine-tune all layers)")

    model = model.to(device)

    # ================================================================
    # 损失函数 / 优化器 / 调度器
    # ================================================================
    criterion = ReIDLoss(
        num_classes=num_classes, margin=TRI_MARGIN, tri_weight=TRI_WEIGHT
    )
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=LEARNING_RATE * 0.01
    )
    scaler = GradScaler("cuda")

    # ================================================================
    # 尝试恢复检查点（断点续训）
    # ================================================================
    start_epoch, best_metrics, history = load_checkpoint(
        model, optimizer, scheduler, scaler, output_dir, device
    )

    if best_metrics is None:
        best_metrics = {"rank1": 0.0, "rank5": 0.0, "mAP": 0.0, "loss": float("inf"),
                        "_epoch": 0}
    if history is None:
        history = _init_history()

    # ================================================================
    # 早停计数器
    # ================================================================
    early_stop_counter = 0
    best_rank1 = best_metrics.get("rank1", 0.0)
    best_epoch = best_metrics.get("_epoch", start_epoch)

    # ================================================================
    # [*] 训练循环
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"  开始训练: Epoch {start_epoch + 1} → {NUM_EPOCHS}")
    print(f"  LR={LEARNING_RATE}, Batch={BATCH_SIZE}, "
          f"EarlyStop={EARLY_STOP_PATIENCE}")
    print(f"{'=' * 70}\n")

    try:
        # epoch 预初始化（用于异常处理中的紧急保存）
        epoch = start_epoch
        for epoch in range(start_epoch + 1, NUM_EPOCHS + 1):
            epoch_start_time = time.time()

            # ---- 训练 ----
            train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, scheduler,
                device, scaler, epoch, num_classes=num_classes
            )

            # ---- 验证 ----
            val_metrics = validate_epoch(
                model, test_loader, criterion, device,
                train_num_classes=num_classes,
                cross_domain=cross_domain,
            )

            # ---- 记录历史 ----
            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["acc"])
            history["precision"].append(train_metrics["precision"])
            history["recall"].append(train_metrics["recall"])
            history["f1"].append(train_metrics["f1"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["acc"])
            history["val_tri"].append(val_metrics["tri"])
            history["rank1"].append(val_metrics["rank1"])
            history["rank5"].append(val_metrics["rank5"])
            history["rank10"].append(val_metrics["rank10"])
            history["rank20"].append(val_metrics["rank20"])
            history["mAP"].append(val_metrics["mAP"])
            history["lr"].append(scheduler.get_last_lr()[0])

            # ---- 日志 ----
            epoch_time = time.time() - epoch_start_time
            current_lr = scheduler.get_last_lr()[0]
            print(f"  -- Epoch {epoch:3d}/{NUM_EPOCHS} "
                  f"({epoch_time:.0f}s, LR={current_lr:.2e}) --")
            print(f"  Train | Loss:{train_metrics['loss']:.4f} "
                  f"CE:{train_metrics['ce']:.4f} "
                  f"Tri:{train_metrics['tri']:.4f} | "
                  f"Acc:{train_metrics['acc']:.3f} "
                  f"P:{train_metrics['precision']:.3f} "
                  f"R:{train_metrics['recall']:.3f} "
                  f"F1:{train_metrics['f1']:.3f}")
            print(f"  Val   | Loss:{val_metrics['loss']:.4f} "
                  f"Tri:{val_metrics['tri']:.4f} | "
                  f"R1:{val_metrics['rank1']:.4f} "
                  f"R5:{val_metrics['rank5']:.4f} "
                  f"R10:{val_metrics['rank10']:.4f} "
                  f"mAP:{val_metrics['mAP']:.4f}")

            # ---- 早停判断 (Rank-1) ----
            current_rank1 = val_metrics["rank1"]
            if current_rank1 > best_rank1 + 1e-6:  # 浮点精度容差
                best_rank1 = current_rank1
                best_epoch = epoch
                best_metrics = {
                    "rank1": val_metrics["rank1"],
                    "rank5": val_metrics["rank5"],
                    "rank10": val_metrics["rank10"],
                    "rank20": val_metrics["rank20"],
                    "mAP": val_metrics["mAP"],
                    "loss": val_metrics["loss"],
                    "_epoch": epoch,
                }
                early_stop_counter = 0

                # 保存最佳模型
                save_checkpoint(model, optimizer, scheduler, scaler,
                                epoch, best_metrics, history, output_dir,
                                is_best=True)
                print(f"  [*] 最佳 Rank-1 更新: {best_rank1:.4f} (mAP={val_metrics['mAP']:.4f})")
            else:
                early_stop_counter += 1
                print(f"  Rank-1 未提升 ({early_stop_counter}/{EARLY_STOP_PATIENCE}) "
                      f"当前={current_rank1:.4f}, 最佳={best_rank1:.4f}")

            # ---- 定期保存检查点 ----
            if epoch % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(model, optimizer, scheduler, scaler,
                                epoch, best_metrics, history, output_dir,
                                is_best=False)
                # 定期绘制训练曲线
                plot_training_curves(history, output_dir)

            # ---- 每 20 轮绘制 t-SNE ----
            if epoch % 20 == 0 or epoch == NUM_EPOCHS:
                print(f"\n  [t-SNE] 正在绘制特征分布图 (epoch {epoch})...")
                plot_tsne_features(model, test_loader, device, output_dir, epoch)

            # ---- 早停触发 ----
            if early_stop_counter >= EARLY_STOP_PATIENCE:
                print(f"\n{'-' * 60}")
                print(f"  早停触发! Rank-1 连续 {EARLY_STOP_PATIENCE} 轮未提升")
                print(f"  最佳轮次: Epoch {best_epoch}, Rank-1={best_rank1:.4f}")
                print(f"{'-' * 60}")
                break

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n[错误] GPU 显存溢出 (OOM)!")
            print(f"  当前 BATCH_SIZE={BATCH_SIZE}")
            print(f"  建议: 减小 batch_size 或使用 gradient_accumulation")
        else:
            print(f"\n[错误] 运行时异常: {e}")
        traceback.print_exc()

        # 尝试保存紧急检查点
        try:
            emergency_path = output_dir / f"emergency_checkpoint_epoch_{epoch}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, emergency_path)
            print(f"  [紧急] 已保存当前状态至: {emergency_path}")
        except Exception:
            print(f"  [紧急] 无法保存紧急检查点")
        sys.exit(1)

    except KeyboardInterrupt:
        print(f"\n[中断] 用户手动终止训练")
        try:
            interrupt_path = output_dir / f"interrupt_checkpoint_epoch_{epoch}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, interrupt_path)
            print(f"  [中断] 已保存当前状态至: {interrupt_path}")
        except Exception:
            pass
        sys.exit(0)

    # ================================================================
    # 训练结束:绘制最终曲线 & 保存结果
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"  训练完成!")
    print(f"{'=' * 70}")

    # 最终训练曲线
    plot_training_curves(history, output_dir)

    # 最终 t-SNE 可视化
    print(f"\n[最终 t-SNE] 正在绘制特征分布图...")
    plot_tsne_features(model, test_loader, device, output_dir, NUM_EPOCHS,
                       max_samples=3000)

    # 保存训练结果报告
    save_results(history, best_metrics, best_epoch, total_params, output_dir)

    # ================================================================
    # 清理旧检查点（保留最佳模型和训练曲线）
    # ================================================================
    print(f"\n[清理] 正在清理中间检查点...")
    for ckpt in sorted(output_dir.glob("checkpoint_epoch_*.pth")):
        try:
            ckpt.unlink()
        except Exception:
            pass

    # ================================================================
    # [*] 最终输出:打印最优权重路径与全部评估指标
    # ================================================================
    best_model_path = output_dir / "best_mot17_finetune.pth"
    final_ckpt_path = output_dir / "final_checkpoint.pth"

    # 保存最终检查点
    torch.save({
        "epoch": best_epoch,
        "model_state_dict": model.state_dict(),
        "best_metrics": best_metrics,
        "history": history,
    }, final_ckpt_path)

    print(f"\n{'=' * 70}")
    print(f"  [OK] MOT17 微调训练全部完成!")
    print(f"{'=' * 70}")
    print(f"\n  [最优权重路径]")
    print(f"    {best_model_path}")
    print(f"\n  [最终检查点]")
    print(f"    {final_ckpt_path}")
    print(f"\n  [训练日志/曲线]")
    print(f"    {output_dir / 'training_curves_mot17_finetune.png'}")
    print(f"    {output_dir / 'results_mot17_finetune.txt'}")
    print(f"\n  [最优模型评估指标 (Epoch {best_epoch})]")
    print(f"    Rank-1:  {best_metrics['rank1']:.4f} ({best_metrics['rank1'] * 100:.2f}%)")
    print(f"    Rank-5:  {best_metrics['rank5']:.4f} ({best_metrics['rank5'] * 100:.2f}%)")
    print(f"    Rank-10: {best_metrics['rank10']:.4f} ({best_metrics['rank10'] * 100:.2f}%)")
    print(f"    Rank-20: {best_metrics['rank20']:.4f} ({best_metrics['rank20'] * 100:.2f}%)")
    print(f"    mAP:     {best_metrics['mAP']:.4f} ({best_metrics['mAP'] * 100:.2f}%)")
    print(f"    Val Loss: {best_metrics['loss']:.4f}")
    print(f"\n  [最终轮次评估指标]")
    if len(history["rank1"]) > 0:
        print(f"    Rank-1:  {history['rank1'][-1]:.4f}")
        print(f"    Rank-5:  {history['rank5'][-1]:.4f}")
        print(f"    mAP:     {history['mAP'][-1]:.4f}")
    print(f"\n  [模型信息]")
    print(f"    总参数量:   {total_params:,}")
    print(f"    训练 ID 数: {num_classes}")
    print(f"    网络结构:   ResNet50 + CBAM + DilatedConv")
    print(f"    输入尺寸:   {INPUT_SIZE[0]}x{INPUT_SIZE[1]}")
    print(f"{'=' * 70}")

    return model, history, best_metrics


if __name__ == "__main__":
    train_mot17_finetune()
