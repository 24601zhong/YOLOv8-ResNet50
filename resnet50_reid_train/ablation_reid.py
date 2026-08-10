"""
============================================================
ResNet50 ReID 消融对比实验脚本 ablation_reid.py

三组实验统一数据集、统一超参：
  对照组 1: 原生 ResNet50 无改进
  对照组 2: ResNet50 仅增加 CBAM 注意力
  实验组 3: ResNet50 + CBAM + 空洞卷积完整改进

输出：对比表格、匹配精度折线图、单帧特征提取耗时
============================================================
"""

import os
import sys
import time
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import create_model, create_baseline_model, ReIDLoss
from train_reid import (
    ReIDDataset, get_train_transforms, get_test_transforms,
    train_epoch, validate,
)
from test_reid import (
    ReIDTestDataset, get_test_transform, extract_features,
    cosine_similarity_matrix, evaluate_matching, compute_cmc_map,
)


# ============================================================
# 统一训练参数（不可修改）
# ============================================================
UNIFIED_REID_PARAMS = {
    "num_epochs": 120,
    "batch_size": 64,
    "lr": 3.5e-4,
    "weight_decay": 5e-4,
    "margin": 0.3,
    "tri_weight": 1.0,
}


def train_ablation_model(train_dir, test_dir, output_dir, model_label,
                         use_cbam, use_dilation, device):
    """
    训练一个消融实验模型
    """
    print(f"\n{'='*70}")
    print(f"  [{model_label}]")
    print(f"  CBAM={use_cbam}, Dilated={use_dilation}")
    print(f"{'='*70}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 数据集
    train_transform = get_train_transforms()
    test_transform = get_test_transforms()
    train_dataset = ReIDDataset(train_dir, transform=train_transform, is_train=True)
    test_dataset = ReIDDataset(test_dir, transform=test_transform, is_train=False)

    num_classes = train_dataset.num_classes
    train_loader = DataLoader(train_dataset, batch_size=UNIFIED_REID_PARAMS["batch_size"],
                              shuffle=True, num_workers=0, pin_memory=True, drop_last=True)  # ★ workers=0
    test_loader = DataLoader(test_dataset, batch_size=UNIFIED_REID_PARAMS["batch_size"],
                             shuffle=False, num_workers=0, pin_memory=True)  # ★ workers=0

    # 模型
    model = create_model(num_classes=num_classes, use_cbam=use_cbam, use_dilation=use_dilation)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())

    # 损失 & 优化器
    criterion = ReIDLoss(num_classes=num_classes, margin=UNIFIED_REID_PARAMS["margin"],
                         tri_weight=UNIFIED_REID_PARAMS["tri_weight"])
    optimizer = torch.optim.Adam(model.parameters(), lr=UNIFIED_REID_PARAMS["lr"],
                                 weight_decay=UNIFIED_REID_PARAMS["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=UNIFIED_REID_PARAMS["num_epochs"],
        eta_min=UNIFIED_REID_PARAMS["lr"] * 0.01,
    )
    scaler = torch.cuda.amp.GradScaler()

    # 训练
    best_acc = 0.0
    for epoch in range(1, UNIFIED_REID_PARAMS["num_epochs"] + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer,
                                    scheduler, device, scaler, epoch)
        val_metrics = validate(model, test_loader, criterion, device)

        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            torch.save(model.state_dict(), output_dir / f"best_{model_label}.pth")

    # 推理耗时测试
    dummy = torch.randn(1, 3, 256, 128).to(device)
    for _ in range(10):
        _ = model(dummy, return_feature=True)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = model(dummy, return_feature=True)
    if device == "cuda":
        torch.cuda.synchronize()
    infer_time = (time.time() - start) / 100 * 1000

    return {
        "model": model,
        "best_acc": best_acc,
        "total_params": total_params,
        "infer_time": infer_time,
        "model_label": model_label,
    }


def evaluate_ablation_model(model_info, test_data_dir, device):
    """
    在 ReID 测试集上评估消融模型
    """
    model = model_info["model"]
    transform = get_test_transform()

    # 使用 gallery 测试集进行评估
    gallery_dataset = ReIDTestDataset(test_data_dir, transform=transform, is_query=False)
    gallery_loader = DataLoader(gallery_dataset, batch_size=64, shuffle=False, num_workers=0)  # ★ workers=0

    # 提取特征
    gallery_feats, gallery_labels, gallery_cams, gallery_paths = extract_features(
        model, gallery_loader, device
    )

    # 使用一半 gallery 作为 query（简化评估）
    n_query = len(gallery_dataset) // 2
    query_feats = gallery_feats[:n_query]
    query_labels = gallery_labels[:n_query]
    query_cams = gallery_cams[:n_query]

    # 余弦相似度评估 (阈值 0.85)
    results, sim_matrix = evaluate_matching(
        query_feats, query_labels, query_cams,
        gallery_feats, gallery_labels, gallery_cams,
        thresholds=(0.85,),
    )

    # CMC & mAP
    cmc_scores, mAP = compute_cmc_map(
        sim_matrix, query_labels, gallery_labels, query_cams, gallery_cams,
    )

    rank1 = cmc_scores[0].item() if len(cmc_scores) > 0 else 0

    return {
        "rank1": rank1,
        "mAP": mAP,
        "accuracy_085": results.get(0.85, {}).get("accuracy", 0),
    }


def run_reid_ablation(train_dir="Market-1501-v15.09.15",
                      test_dir="Market-1501-v15.09.15",
                      output_dir="train_output"):
    """
    ResNet50 ReID 消融实验主函数
    """
    print("=" * 70)
    print("  ResNet50 ReID 消融对比实验")
    print("  对照组 1: 原生 ResNet50 (无改进)")
    print("  对照组 2: ResNet50 + CBAM 注意力")
    print("  实验组 3: ResNet50 + CBAM + 空洞卷积 (完整改进)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(output_dir)
    ablation_dir = output_dir / "ablation_reid"
    ablation_dir.mkdir(parents=True, exist_ok=True)

    # 检查数据路径
    if not Path(train_dir).exists():
        print(f"[错误] 数据集不存在: {train_dir}")
        return

    # ============================================================
    # 三组消融实验
    # ============================================================
    ablation_configs = [
        {
            "label": "1_Baseline_ResNet50",
            "use_cbam": False,
            "use_dilation": False,
            "desc": "对照组 1: 原生 ResNet50 无改进",
        },
        {
            "label": "2_ResNet50_CBAM",
            "use_cbam": True,
            "use_dilation": False,
            "desc": "对照组 2: ResNet50 + CBAM 注意力",
        },
        {
            "label": "3_ResNet50_CBAM_Dilated",
            "use_cbam": True,
            "use_dilation": True,
            "desc": "实验组 3: ResNet50 + CBAM + 空洞卷积",
        },
    ]

    all_results = []

    for cfg in ablation_configs:
        print(f"\n[消融] {cfg['desc']}")

        model_info = train_ablation_model(
            train_dir=train_dir,
            test_dir=test_dir,
            output_dir=ablation_dir / cfg["label"],
            model_label=cfg["label"],
            use_cbam=cfg["use_cbam"],
            use_dilation=cfg["use_dilation"],
            device=device,
        )

        eval_results = evaluate_ablation_model(
            model_info, test_dir, device
        )

        result = {
            "group": cfg["desc"],
            "label": cfg["label"],
            "best_acc": model_info["best_acc"],
            "total_params": model_info["total_params"],
            "infer_time": model_info["infer_time"],
            "rank1": eval_results["rank1"],
            "mAP": eval_results["mAP"],
            "acc_085": eval_results["accuracy_085"],
        }
        all_results.append(result)

        print(f"  完成: Acc={result['best_acc']:.4f}, Rank-1={result['rank1']:.4f}, "
              f"mAP={result['mAP']:.4f}")

        torch.cuda.empty_cache()

    # ============================================================
    # 生成消融对比图表
    # ============================================================
    _plot_reid_ablation(all_results, ablation_dir)
    _save_reid_ablation_table(all_results, ablation_dir)

    print(f"\n[完成] 消融实验完成，结果保存在: {ablation_dir}")
    return all_results


def _plot_reid_ablation(all_results, output_dir):
    """绘制 ReID 消融对比图"""
    groups = [r["group"] for r in all_results]
    rank1_vals = [r["rank1"] for r in all_results]
    map_vals = [r["mAP"] for r in all_results]
    acc_vals = [r["best_acc"] for r in all_results]
    infer_times = [r["infer_time"] for r in all_results]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- 精度折线图 ---
    ax1 = axes[0]
    x = np.arange(len(groups))
    ax1.plot(x, rank1_vals, "o-", color="#2196F3", linewidth=2, markersize=8, label="Rank-1")
    ax1.plot(x, map_vals, "s-", color="#4CAF50", linewidth=2, markersize=8, label="mAP")
    ax1.plot(x, acc_vals, "^-", color="#FF9800", linewidth=2, markersize=8, label="Val Acc")
    ax1.set_xlabel("Experiment Group", fontsize=11)
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_title("ResNet50 ReID Ablation: Accuracy", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, fontsize=7, rotation=10)
    ax1.legend(loc="lower right")
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.3)

    for i, v in enumerate(rank1_vals):
        if v > 0:
            ax1.annotate(f"{v:.3f}", (i, v), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8)

    # --- 推理耗时对比 ---
    ax2 = axes[1]
    bars = ax2.bar(groups, infer_times, color=["#607D8B", "#2196F3", "#4CAF50"], alpha=0.85)
    ax2.set_xlabel("Experiment Group", fontsize=11)
    ax2.set_ylabel("Inference Time (ms)", fontsize=11)
    ax2.set_title("Single Frame Feature Extraction Time", fontsize=13, fontweight="bold")
    ax2.set_xticklabels(groups, fontsize=7, rotation=10)
    ax2.grid(axis="y", alpha=0.3)

    for bar, t in zip(bars, infer_times):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{t:.2f}ms", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "reid_ablation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  消融对比图已保存")


def _save_reid_ablation_table(all_results, output_dir):
    """保存 ReID 消融对比表格"""
    table_path = output_dir / "reid_ablation_table.txt"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write("ResNet50 ReID 消融对比实验 - 标准化指标表格\n")
        f.write("=" * 90 + "\n")
        f.write(f"{'实验组':<40s} {'Rank-1':>8s} {'mAP':>8s} {'Val Acc':>9s} {'Params':>10s} {'Infer(ms)':>10s}\n")
        f.write("-" * 90 + "\n")
        for r in all_results:
            f.write(f"{r['group']:<40s} {r['rank1']:>8.4f} {r['mAP']:>8.4f} "
                    f"{r['best_acc']:>9.4f} {r['total_params']:>10,} {r['infer_time']:>10.2f}\n")
        f.write("=" * 90 + "\n")
        f.write("\n统计规则:\n")
        f.write("  Rank-1: 首位匹配准确率 (CMC Rank-1)\n")
        f.write("  mAP:    平均精度均值 (Mean Average Precision)\n")
        f.write("  Val Acc: 分类验证准确率\n")
        f.write("  Params: 模型总参数量\n")
        f.write("  Infer:  单帧(256×128) 2048维特征提取耗时\n")
        f.write("\n统一参数: epochs=120, batch=64, lr=3.5e-4, Adam, CosineLR, margin=0.3\n")

    print(f"  消融对比表已保存: {table_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ResNet50 ReID 消融实验")
    parser.add_argument("--train_dir", type=str, default="../Market-1501-v15.09.15",  # ★ 上级目录
                        help="训练数据集目录")
    parser.add_argument("--test_dir", type=str, default="../Market-1501-v15.09.15",  # ★ 上级目录
                        help="测试数据集目录")
    parser.add_argument("--output", type=str, default="train_output",
                        help="输出目录")
    args = parser.parse_args()

    run_reid_ablation(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output,
    )
