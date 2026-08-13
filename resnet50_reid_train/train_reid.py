"""
============================================================
改进 ResNet50 ReID 模型训练脚本 train_reid.py

分两步训练：
  Step 1 - 预训练: Market-1501 数据集训练，生成基础 ReID 权重
  Step 2 - 微调: 加载预训练权重，使用酒店 ReID 数据集微调

损失函数: L = L_cls + 1.0 * L_tri
   L_cls: ID 分类交叉熵
   L_tri: 难样本挖掘 Triplet Loss (margin=0.3)

训练超参:
  优化器 Adam，余弦退火学习率，epochs=120
============================================================
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import create_model, create_baseline_model, ReIDLoss


# ============================================================
# 数据集类
# ============================================================
class ReIDDataset(Dataset):
    """
    行人重识别数据集加载器
    对齐 Market-1501 格式：bounding_box_train / bounding_box_test
    文件命名规则: 0002_c1s1_000451_03.jpg
                  人员ID_摄像头ID_序列号_帧号.jpg
    """

    def __init__(self, data_dir, transform=None, is_train=True, pid_to_idx=None):
        """
        Args:
            data_dir: 数据集目录
            transform: 图像变换
            is_train: 是否为训练模式
            pid_to_idx: 可选，外部传入的 pid->idx 映射 (用于测试集与训练集共享映射)；
                        传入则使用之，并丢弃映射中不存在的 pid；None 则自建映射。
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_train = is_train

        self.images = []
        self.labels = []
        self.person_ids = set()

        # 搜索图片
        if is_train:
            img_dir = self.data_dir / "bounding_box_train"
        else:
            img_dir = self.data_dir / "bounding_box_test"

        if not img_dir.exists():
            # 兼容无子目录的情况
            img_dir = self.data_dir

        # DEBUG
        jpg_files = list(img_dir.glob("*.jpg"))
        print(f"  [DEBUG] img_dir={img_dir}, exists={img_dir.exists()}, jpg_count={len(jpg_files)}", flush=True)

        # 先收集 (path, raw_pid)
        raw_items = []
        for img_path in sorted(jpg_files):
            if not img_path.is_file():
                continue

            # 解析人员 ID：Market-1501 格式
            # 文件名格式: 0002_c1s1_000451_03.jpg
            fname = img_path.stem
            try:
                pid = int(fname.split("_")[0])
            except (ValueError, IndexError):
                # 如果不能解析，用文件名哈希
                pid = hash(fname) % 10000

            # 跳过 Market-1501 的 junk ID (-1)：不是真实人员，不应参与分类
            if pid == -1:
                continue

            raw_items.append((img_path, pid))
            self.person_ids.add(pid)

        # 构建/复用 pid_to_idx
        if pid_to_idx is None:
            # 训练集：自建连续索引映射
            self.pid_to_idx = {pid: i for i, pid in enumerate(sorted(self.person_ids))}
        else:
            # 测试集：复用训练集映射，丢弃训练集中不存在的 pid
            # (避免测试集独立建映射导致 idx 与训练集不一致，进而引发交叉熵 label 越界)
            self.pid_to_idx = pid_to_idx
            raw_items = [(p, pid) for p, pid in raw_items if pid in pid_to_idx]
            self.person_ids = {pid for _, pid in raw_items}

        self.images = [p for p, _ in raw_items]
        self.labels = [self.pid_to_idx[pid] for _, pid in raw_items]
        self.num_classes = len(self.pid_to_idx)

        print(f"[ReID Dataset] {data_dir}")
        print(f"  图片数: {len(self.images)}, 人员ID数: {self.num_classes}")

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
# 数据增强变换
# ============================================================
def get_train_transforms(height=256, width=128):
    """训练数据增强（V3 优化：降低几何/颜色扰动强度，提升随机擦除）"""
    return T.Compose([
        T.Resize((height, width)),
        T.RandomHorizontalFlip(p=0.5),
        # ★ V3: 移除 RandomRotation(10) — 身体朝向是 ReID 关键线索
        T.ColorJitter(brightness=0.2, contrast=0.15),  # ★ V3: 从 0.4/0.3 降低
        T.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # ★ V3: 从 (0.1,0.1) 降低
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),  # ★ V3: 从 p=0.3 提升
    ])


def get_test_transforms(height=256, width=128):
    """测试变换"""
    return T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ============================================================
# 训练一个 epoch
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
            logits, features = model(imgs)
            loss, (ce, tri) = criterion(logits, features, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 统计
        bs = imgs.size(0)
        running_loss += loss.item() * bs
        running_ce += ce.item() * bs
        running_tri += tri.item() * bs
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += bs

        pbar.set_postfix({
            "loss": f"{loss.item():.3f}",
            "ce": f"{ce.item():.3f}",
            "tri": f"{tri.item():.3f}",
            "acc": f"{correct/max(total,1):.3f}",
        })

    scheduler.step()

    return {
        "loss": running_loss / max(total, 1),
        "ce": running_ce / max(total, 1),
        "tri": running_tri / max(total, 1),
        "acc": correct / max(total, 1),
    }


# ============================================================
# 验证
# ============================================================
@torch.no_grad()
def validate(model, loader, criterion, device, train_num_classes=None, cross_domain=False):
    """验证
    Args:
        train_num_classes: 训练集类别数。若测试集标签超出此范围（跨域评估），
                          则跳过 CE loss，仅计算 triplet loss。
        cross_domain: 显式指定跨域评估（训练/测试 ID 不重叠）。若为 True，
                     则跳过 range 检测，直接使用 triplet-only 验证。
    """
    model.eval()
    running_loss = 0.0
    running_ce = 0.0
    running_tri = 0.0
    correct = 0
    total = 0

    all_features = []
    all_labels = []

    for imgs, labels, _ in tqdm(loader, desc="Validating"):
        imgs, labels = imgs.to(device), labels.to(device)

        logits, features = model(imgs)

        # ★ 跨域评估：测试集 ID 与训练集不重叠，CE loss 无意义，仅用 triplet
        # 支持显式指定或自动检测（range 检测在训练/测试 ID 数相同时失效，故优先用显式）
        if cross_domain or (train_num_classes is not None and labels.max().item() >= train_num_classes):
            cross_domain = True
            # 仅计算 triplet loss
            from model import TripletLoss
            tri_fn = TripletLoss(margin=0.3).to(device)
            tri = tri_fn(features, labels)
            loss = tri
            ce_val = 0.0
            tri_val = tri.item()
            bs = imgs.size(0)
            running_loss += loss.item() * bs
            running_tri += tri_val * bs
            total += bs
        else:
            loss, (ce_val, tri_val) = criterion(logits, features, labels)
            bs = imgs.size(0)
            running_loss += loss.item() * bs
            running_ce += ce_val.item() * bs
            running_tri += tri_val.item() * bs
            _, preds = logits.max(1)
            correct += preds.eq(labels).sum().item()
            total += bs

        all_features.append(features.cpu())
        all_labels.append(labels.cpu())

    # ★ 修复: 空 loader 时安全返回
    if total == 0:
        return {
            "loss": 0.0, "ce": 0.0, "tri": 0.0, "acc": 0.0, "cross_domain": True,
            "features": torch.empty(0, device="cpu"),
            "labels": torch.empty(0, device="cpu", dtype=torch.long),
        }

    return {
        "loss": running_loss / max(total, 1),
        "ce": running_ce / max(total, 1),
        "tri": running_tri / max(total, 1),
        "acc": correct / max(total, 1) if not cross_domain else 0.0,
        "cross_domain": cross_domain,
        "features": torch.cat(all_features, dim=0),
        "labels": torch.cat(all_labels, dim=0),
    }


# ============================================================
# 训练主函数
# ============================================================
def train_reid(
    train_data_dir,
    test_data_dir,
    output_dir,
    pretrained_weights=None,
    num_epochs=120,
    batch_size=64,
    lr=3.5e-4,
    weight_decay=5e-4,
    num_classes=None,
    device=None,
    use_cbam=True,
    use_dilation=True,
    model_label="improved",
):
    """
    ReID 模型训练主函数

    Args:
        train_data_dir: 训练数据集目录
        test_data_dir: 测试数据集目录
        output_dir: 输出目录
        pretrained_weights: 预训练权重路径 (用于微调)
        num_epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        weight_decay: 权重衰减
        num_classes: 类别数 (None=自动推断)
        device: 训练设备
        use_cbam: 是否使用 CBAM
        use_dilation: 是否使用空洞卷积
        model_label: 模型标识
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  ResNet50 ReID 模型训练 [{model_label}]")
    print(f"  CBAM: {use_cbam}, Dilated Conv: {use_dilation}")
    print("=" * 70)

    # ========== 数据集 ==========
    train_transform = get_train_transforms()
    test_transform = get_test_transforms()

    train_dataset = ReIDDataset(train_data_dir, transform=train_transform, is_train=True)

    # ★ Market-1501 训练集/测试集人员 ID 不同，测试集独立建映射
    # 通过 check_test_overlap 判断：若测试集 PID 大多不在训练集中，则独立建映射
    test_dataset = ReIDDataset(
        test_data_dir, transform=test_transform, is_train=False,
        pid_to_idx=None,  # 测试集独立建映射（跨域评估场景）
    )

    if num_classes is None:
        num_classes = train_dataset.num_classes  # 模型分类头以训练集 ID 数为准

    # ★ 检测跨域评估：比较训练集/测试集的 PID 集合是否重叠
    train_pids = set(train_dataset.pid_to_idx.keys())
    test_pids = set(test_dataset.pid_to_idx.keys())
    pid_overlap = train_pids & test_pids
    cross_domain = len(pid_overlap) == 0
    print(f"\n[数据] 训练集: {len(train_dataset)} 张, {num_classes} 个ID")
    print(f"[数据] 测试集: {len(test_dataset)} 张, {test_dataset.num_classes} 个ID")
    print(f"[数据] PID 重叠: {len(pid_overlap)}/{len(test_pids)}")
    if cross_domain:
        print(f"[数据] ★ 跨域评估：训练/测试人员 ID 不重叠，验证仅用 triplet loss")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,  # ★ Windows: workers=0 防 spawn 崩溃
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,  # ★ Windows: workers=0 防 spawn 崩溃
    )

    # ========== 模型 ==========
    print(f"\n[模型] 创建改进 ResNet50, num_classes={num_classes}")
    model = create_model(
        num_classes=num_classes,
        use_cbam=use_cbam,
        use_dilation=use_dilation,
    )

    # 加载预训练权重
    if pretrained_weights and Path(pretrained_weights).exists():
        print(f"[模型] 加载预训练权重: {pretrained_weights}")
        state_dict = torch.load(pretrained_weights, map_location=device)
        # 处理分类器维度不匹配
        if "classifier.weight" in state_dict:
            if state_dict["classifier.weight"].size(0) != num_classes:
                print(f"[警告] 分类器维度不匹配，移除分类器权重")
                del state_dict["classifier.weight"]
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_params:,}")

    # ========== 损失函数 & 优化器 & 调度器 ==========
    criterion = ReIDLoss(num_classes=num_classes, margin=0.3, tri_weight=1.0)
    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)
    scaler = GradScaler('cuda')

    # ========== 训练循环 ==========
    print(f"\n[训练] epochs={num_epochs}, lr={lr}, batch={batch_size}")
    print("=" * 70)

    best_acc = float('-inf')  # 跨域用 -tri_loss，同域用 acc，统一起始值
    best_epoch = 0
    best_path = None  # 防止从未触发最佳模型保存时未定义
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        # 训练
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, scaler, epoch
        )
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])

        # 验证
        val_metrics = validate(model, test_loader, criterion, device,
                               train_num_classes=num_classes,
                               cross_domain=cross_domain)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])

        # 保存最佳模型（跨域评估时用 triplet loss 作为指标）
        if val_metrics.get("cross_domain"):
            val_score = -val_metrics["tri"]  # triplet loss 越低越好
        else:
            val_score = val_metrics["acc"]

        if val_score > best_acc:
            best_acc = val_score
            best_epoch = epoch
            best_path = output_dir / f"best_{model_label}.pth"
            torch.save(model.state_dict(), best_path)
            tag = "tri_loss" if val_metrics.get("cross_domain") else "acc"
            print(f"  >>> Best model saved (epoch {epoch}, {tag}={val_score:.4f})")

        # 日志
        current_lr = scheduler.get_last_lr()[0]
        if val_metrics.get("cross_domain"):
            print(f"  Epoch {epoch:3d}/{num_epochs} | "
                  f"Train Loss:{train_metrics['loss']:.4f} Acc:{train_metrics['acc']:.3f} | "
                  f"Val Tri:{val_metrics['tri']:.4f} [跨域] | "
                  f"LR:{current_lr:.2e}")
        else:
            print(f"  Epoch {epoch:3d}/{num_epochs} | "
                  f"Train Loss:{train_metrics['loss']:.4f} Acc:{train_metrics['acc']:.3f} | "
                  f"Val Loss:{val_metrics['loss']:.4f} Acc:{val_metrics['acc']:.3f} | "
                  f"LR:{current_lr:.2e}")

        # 每 10 epoch 增量保存训练曲线，防止中途崩溃丢失
        if epoch % 10 == 0:
            _plot_training_curves(history, output_dir, model_label)

    # 保存最终模型
    final_path = output_dir / f"final_{model_label}.pth"
    torch.save(model.state_dict(), final_path)

    # ========== 绘制训练曲线 ==========
    _plot_training_curves(history, output_dir, model_label)

    # ========== 保存训练结果 ==========
    _save_results(history, best_acc, best_epoch, total_params, output_dir, model_label)

    print("\n" + "=" * 70)
    print(f"  训练完成! 最佳准确率: {best_acc:.4f} (epoch {best_epoch})")
    if best_path is not None:
        print(f"  最佳模型: {best_path}")
    else:
        print("  最佳模型: (未保存，全程未触发 best_acc 提升)")
    print("=" * 70)

    return model, history


def _plot_training_curves(history, output_dir, model_label):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss 曲线
    ax1 = axes[0]
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=1.5)
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"ReID Training Loss ({model_label})")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy 曲线
    ax2 = axes[1]
    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc", linewidth=1.5)
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"ReID Accuracy ({model_label})")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / f"training_curves_{model_label}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)  # 显式关闭 fig，避免频繁调用时的内存泄漏
    print(f"  训练曲线已保存: {save_path}")


def _save_results(history, best_acc, best_epoch, total_params, output_dir, model_label):
    """保存训练结果"""
    results_path = output_dir / f"results_{model_label}.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"ResNet50 ReID Training Results [{model_label}]\n")
        f.write("=" * 60 + "\n")
        f.write(f"Best Accuracy:  {best_acc:.4f} (epoch {best_epoch})\n")
        f.write(f"Final Train Acc: {history['train_acc'][-1]:.4f}\n")
        f.write(f"Final Val Acc:   {history['val_acc'][-1]:.4f}\n")
        f.write(f"Total Params:    {total_params:,}\n")
        f.write(f"Total Epochs:    {len(history['train_loss'])}\n")


# ============================================================
# Market-1501 单阶段训练入口
# ============================================================
def train_market1501(
    market1501_dir="Market-1501-v15.09.15",
    output_dir="train_output/reid_log",
    device=None,
    num_epochs=120,
    batch_size=64,
    lr=3.5e-4,
    pretrained_weights=None,
    use_cbam=True,
    use_dilation=True,
):
    """
    Market-1501 数据集训练 ReID 模型（单阶段）

    使用 Market-1501 的 bounding_box_train 训练，
    bounding_box_test 验证，query 测试。
    """
    print("=" * 70)
    print("  ResNet50 ReID 训练 (Market-1501)")
    print(f"  改进: CBAM 注意力={use_cbam}, 空洞卷积={use_dilation}")
    print("=" * 70)

    output_dir = Path(output_dir)

    # 检查 Market-1501 路径
    market_path = Path(market1501_dir)
    if not market_path.exists():
        print(f"[错误] Market-1501 数据集不存在: {market1501_dir}")
        return None

    model, history = train_reid(
        train_data_dir=str(market_path),
        test_data_dir=str(market_path),
        output_dir=str(output_dir),
        pretrained_weights=pretrained_weights,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        num_classes=None,           # 自动推断 (Market-1501: 751 IDs)
        device=device,
        use_cbam=use_cbam,
        use_dilation=use_dilation,
        model_label="market1501_improved",
    )

    print("\n" + "=" * 70)
    print("  Market-1501 训练完成!")
    print(f"  模型保存至: {output_dir}")
    print("=" * 70)

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ResNet50 ReID 模型训练")
    parser.add_argument("--mode", type=str, default="market1501",
                        choices=["market1501", "custom"],
                        help="训练模式: market1501(默认) 或 custom")
    parser.add_argument("--train_dir", type=str,
                        default="../Market-1501-v15.09.15",  # ★ 上级目录
                        help="训练数据集目录")
    parser.add_argument("--test_dir", type=str,
                        default="../Market-1501-v15.09.15",  # ★ 上级目录
                        help="测试数据集目录")
    parser.add_argument("--output", type=str,
                        default="train_output/reid_log",
                        help="输出目录")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="预训练权重路径")
    parser.add_argument("--epochs", type=int, default=120, help="训练轮数")
    parser.add_argument("--batch", type=int, default=64, help="批次大小")
    parser.add_argument("--lr", type=float, default=3.5e-4, help="学习率")
    parser.add_argument("--no_cbam", action="store_true", help="禁用 CBAM 注意力模块")
    parser.add_argument("--no_dilated", action="store_true", help="禁用空洞卷积")
    parser.add_argument("--device", type=str, default="cuda", help="训练设备 (如 cuda / cpu)")
    args = parser.parse_args()

    if args.mode == "market1501":
        train_market1501(
            market1501_dir=args.train_dir,
            output_dir=args.output,
            device=args.device,
            num_epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            pretrained_weights=args.pretrained,
            use_cbam=not args.no_cbam,
            use_dilation=not args.no_dilated,
        )
    else:
        train_reid(
            train_data_dir=args.train_dir,
            test_data_dir=args.test_dir,
            output_dir=args.output,
            pretrained_weights=args.pretrained,
            num_epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            device=args.device,
            use_cbam=not args.no_cbam,
            use_dilation=not args.no_dilated,
        )
