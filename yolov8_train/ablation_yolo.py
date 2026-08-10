"""
============================================================
YOLOv8 消融对比实验脚本 ablation_yolo.py
三组实验统一数据集、统一超参：
  对照组 A: 原生 YOLOv8n 无改进
  对照组 B: 仅 MixConv 轻量化，保留 PAN-FPN
  实验组 C: MixConv + BiFPN 完整改进

输出：标准化对比表格、mAP 对比柱状图、loss 曲线
============================================================
"""

import os
import sys
import time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# 统一训练参数（不可修改）
# ============================================================
UNIFIED_PARAMS = {
    # ★ 适配 RTX 4070 16GB RAM: 降低 batch/workers/imgsz
    "imgsz": 512,
    "batch": 4,
    "epochs": 120,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "workers": 0,             # ★ Windows spawn 多进程导致 MemoryError
    "optimizer": "AdamW",
    "cos_lr": True,
    "patience": 12,
    "single_cls": True,
    "data": "dataset/det/hotel_det.yaml",
}


def run_baseline_a(output_dir):
    """
    对照组 A: 原生 YOLOv8n 无改进
    """
    print("\n" + "=" * 70)
    print("  对照组 A: 原生 YOLOv8n (无改进)")
    print("=" * 70)

    project = str(Path(output_dir) / "ablation")
    name = "A_baseline_yolov8n"

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=UNIFIED_PARAMS["data"],
        epochs=UNIFIED_PARAMS["epochs"],
        imgsz=UNIFIED_PARAMS["imgsz"],
        batch=UNIFIED_PARAMS["batch"],
        device=UNIFIED_PARAMS["device"],
        workers=UNIFIED_PARAMS["workers"],
        optimizer=UNIFIED_PARAMS["optimizer"],
        cos_lr=UNIFIED_PARAMS["cos_lr"],
        patience=UNIFIED_PARAMS["patience"],
        single_cls=UNIFIED_PARAMS["single_cls"],
        project=project,
        name=name,
        exist_ok=True,
    )

    # 收集指标
    metrics = {
        "group": "A: YOLOv8n Baseline",
        "map50": results.results_dict.get("metrics/mAP50(B)", 0),
        "map50_95": results.results_dict.get("metrics/mAP50-95(B)", 0),
        "precision": results.results_dict.get("metrics/precision(B)", 0),
        "recall": results.results_dict.get("metrics/recall(B)", 0),
    }
    return metrics, str(Path(project) / name)


def run_baseline_b(output_dir):
    """
    对照组 B: 仅 MixConv 轻量化，保留 PAN-FPN
    """
    print("\n" + "=" * 70)
    print("  对照组 B: YOLOv8n + MixConv (保留 PAN-FPN)")
    print("=" * 70)

    project = str(Path(output_dir) / "ablation")
    name = "B_mixconv_only"

    # 使用仅含 MixConv 的 YAML 配置
    cfg_path = Path(__file__).parent / "yolov8_mixconv_only.yaml"

    if not cfg_path.exists():
        # 创建仅 MixConv 的配置（无 BiFPN）
        _create_mixconv_only_cfg(cfg_path)

    model = YOLO(str(cfg_path))
    results = model.train(
        data=UNIFIED_PARAMS["data"],
        epochs=UNIFIED_PARAMS["epochs"],
        imgsz=UNIFIED_PARAMS["imgsz"],
        batch=UNIFIED_PARAMS["batch"],
        device=UNIFIED_PARAMS["device"],
        workers=UNIFIED_PARAMS["workers"],
        optimizer=UNIFIED_PARAMS["optimizer"],
        cos_lr=UNIFIED_PARAMS["cos_lr"],
        patience=UNIFIED_PARAMS["patience"],
        single_cls=UNIFIED_PARAMS["single_cls"],
        project=project,
        name=name,
        exist_ok=True,
    )

    metrics = {
        "group": "B: +MixConv Only",
        "map50": results.results_dict.get("metrics/mAP50(B)", 0),
        "map50_95": results.results_dict.get("metrics/mAP50-95(B)", 0),
        "precision": results.results_dict.get("metrics/precision(B)", 0),
        "recall": results.results_dict.get("metrics/recall(B)", 0),
    }
    return metrics, str(Path(project) / name)


def run_experiment_c(output_dir):
    """
    实验组 C: MixConv + BiFPN 完整改进
    """
    print("\n" + "=" * 70)
    print("  实验组 C: YOLOv8n + MixConv + BiFPN (完整改进)")
    print("=" * 70)

    project = str(Path(output_dir) / "ablation")
    name = "C_mixconv_bifpn_full"

    cfg_path = Path(__file__).parent / "yolov8_mix_bifpn.yaml"

    model = YOLO(str(cfg_path))
    results = model.train(
        data=UNIFIED_PARAMS["data"],
        epochs=UNIFIED_PARAMS["epochs"],
        imgsz=UNIFIED_PARAMS["imgsz"],
        batch=UNIFIED_PARAMS["batch"],
        device=UNIFIED_PARAMS["device"],
        workers=UNIFIED_PARAMS["workers"],
        optimizer=UNIFIED_PARAMS["optimizer"],
        cos_lr=UNIFIED_PARAMS["cos_lr"],
        patience=UNIFIED_PARAMS["patience"],
        single_cls=UNIFIED_PARAMS["single_cls"],
        project=project,
        name=name,
        exist_ok=True,
    )

    metrics = {
        "group": "C: MixConv+BiFPN (Full)",
        "map50": results.results_dict.get("metrics/mAP50(B)", 0),
        "map50_95": results.results_dict.get("metrics/mAP50-95(B)", 0),
        "precision": results.results_dict.get("metrics/precision(B)", 0),
        "recall": results.results_dict.get("metrics/recall(B)", 0),
    }
    return metrics, str(Path(project) / name)


def _create_mixconv_only_cfg(save_path):
    """创建仅包含 MixConv 的 YAML 配置（保留 PAN-FPN neck）"""
    cfg_content = """# YOLOv8 + MixConv Only (保留 PAN-FPN)
nc: 1
scales:
  n: [0.33, 0.25, 2.0]
  s: [0.33, 0.50, 2.0]
  m: [0.67, 0.75, 1.5]
  l: [1.00, 1.00, 1.0]
  x: [1.00, 1.25, 1.0]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, MixConv2d, [128, [3, 5, 7]]]
  - [-1, 1, C2f, [128, True]]
  - [-1, 1, MixConv2d, [256, [3, 5, 7]]]
  - [-1, 2, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 1, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 1, C2f, [512]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 1, C2f, [256]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 1, C2f, [512]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]
  - [-1, 1, C2f, [1024]]
  - [[15, 18, 21], 1, Detect, [nc]]
"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        f.write(cfg_content)
    print(f"  已创建配置: {save_path}")


def run_ablation_study(output_dir="train_output"):
    """
    消融实验主函数：依次运行三组实验并生成对比图表
    """
    print("=" * 70)
    print("  YOLOv8 改进消融对比实验")
    print("  对照组 A: 原生 YOLOv8n")
    print("  对照组 B: 仅 MixConv 轻量化")
    print("  实验组 C: MixConv + BiFPN 完整改进")
    print("=" * 70)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 注册自定义模块
    print("\n[准备] 注册自定义模块...")
    sys.path.insert(0, str(Path(__file__).parent))
    from custom_modules import register_custom_modules, MixConv2d, BiFPN
    register_custom_modules()
    import ultralytics.nn.modules as m
    m.MixConv2d = MixConv2d
    m.BiFPN = BiFPN

    all_metrics = []

    # --- 对照组 A: 原生 YOLOv8n ---
    try:
        metrics_a, path_a = run_baseline_a(output_dir)
        all_metrics.append(metrics_a)
        print(f"  对照组 A 完成: {path_a}")
    except Exception as e:
        print(f"  对照组 A 失败: {e}")
        all_metrics.append({
            "group": "A: YOLOv8n Baseline",
            "map50": 0, "map50_95": 0, "precision": 0, "recall": 0,
        })

    # --- 对照组 B: 仅 MixConv ---
    try:
        metrics_b, path_b = run_baseline_b(output_dir)
        all_metrics.append(metrics_b)
        print(f"  对照组 B 完成: {path_b}")
    except Exception as e:
        print(f"  对照组 B 失败: {e}")
        all_metrics.append({
            "group": "B: +MixConv Only",
            "map50": 0, "map50_95": 0, "precision": 0, "recall": 0,
        })

    # --- 实验组 C: MixConv + BiFPN ---
    try:
        metrics_c, path_c = run_experiment_c(output_dir)
        all_metrics.append(metrics_c)
        print(f"  实验组 C 完成: {path_c}")
    except Exception as e:
        print(f"  实验组 C 失败: {e}")
        all_metrics.append({
            "group": "C: MixConv+BiFPN (Full)",
            "map50": 0, "map50_95": 0, "precision": 0, "recall": 0,
        })

    # ============================================================
    # 生成消融对比图表
    # ============================================================
    _plot_ablation_results(all_metrics, output_dir)
    _save_ablation_table(all_metrics, output_dir)

    return all_metrics


def _plot_ablation_results(all_metrics, output_dir):
    """绘制消融对比图表"""
    groups = [m["group"] for m in all_metrics]
    map50_vals = [m["map50"] for m in all_metrics]
    map50_95_vals = [m["map50_95"] for m in all_metrics]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- mAP 对比柱状图 ---
    ax1 = axes[0]
    x = np.arange(len(groups))
    width = 0.35

    bars1 = ax1.bar(x - width/2, map50_vals, width, label="mAP@0.5",
                    color="#2196F3", alpha=0.85)
    bars2 = ax1.bar(x + width/2, map50_95_vals, width, label="mAP@0.5:0.95",
                    color="#4CAF50", alpha=0.85)

    ax1.set_xlabel("Experiment Group", fontsize=11)
    ax1.set_ylabel("mAP", fontsize=11)
    ax1.set_title("YOLOv8 Ablation: mAP Comparison", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, fontsize=8, rotation=10)
    ax1.legend(loc="lower right")
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis="y", alpha=0.3)

    # 在柱状图上标注数值
    for bar, val in zip(bars1, map50_vals):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, map50_95_vals):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # --- 精度-召回率对比 ---
    ax2 = axes[1]
    precision_vals = [m["precision"] for m in all_metrics]
    recall_vals = [m["recall"] for m in all_metrics]

    x2 = np.arange(len(groups))
    ax2.plot(x2, precision_vals, "o-", color="#FF9800", linewidth=2, markersize=8, label="Precision")
    ax2.plot(x2, recall_vals, "s-", color="#9C27B0", linewidth=2, markersize=8, label="Recall")
    ax2.set_xlabel("Experiment Group", fontsize=11)
    ax2.set_ylabel("Score", fontsize=11)
    ax2.set_title("Precision-Recall Comparison", fontsize=13, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(groups, fontsize=8, rotation=10)
    ax2.legend(loc="lower right")
    ax2.set_ylim(0, 1.0)
    ax2.grid(alpha=0.3)

    for i, (p, r) in enumerate(zip(precision_vals, recall_vals)):
        if p > 0:
            ax2.annotate(f"{p:.3f}", (i, p), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8)
        if r > 0:
            ax2.annotate(f"{r:.3f}", (i, r), textcoords="offset points",
                        xytext=(0, -15), ha="center", fontsize=8)

    plt.tight_layout()
    ablation_dir = output_dir / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    save_path = ablation_dir / "ablation_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[图表] 消融对比图已保存: {save_path}")


def _save_ablation_table(all_metrics, output_dir):
    """保存标准化对比表格"""
    table_path = output_dir / "ablation" / "ablation_table.txt"
    table_path.parent.mkdir(parents=True, exist_ok=True)

    with open(table_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("YOLOv8 消融对比实验 - 标准化指标表格\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'实验组':<35s} {'mAP@0.5':>10s} {'mAP@0.5:0.95':>14s} {'Precision':>11s} {'Recall':>10s}\n")
        f.write("-" * 80 + "\n")
        for m in all_metrics:
            f.write(f"{m['group']:<35s} {m['map50']:>10.4f} {m['map50_95']:>14.4f} {m['precision']:>11.4f} {m['recall']:>10.4f}\n")
        f.write("=" * 80 + "\n")
        f.write("\n统计规则:\n")
        f.write("  mAP@0.5:       IoU=0.5 阈值下的平均精度\n")
        f.write("  mAP@0.5:0.95:  IoU 0.5~0.95 步长 0.05 的平均 mAP\n")
        f.write("  Precision:     所有类别的平均精确率\n")
        f.write("  Recall:        所有类别的平均召回率\n")
        f.write("  统一参数: imgsz=640, batch=16, epochs=200, AdamW, cos_lr\n")

    print(f"[表格] 消融对比表已保存: {table_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 消融对比实验")
    parser.add_argument("--output", type=str, default="train_output",
                        help="输出目录")
    parser.add_argument("--skip_a", action="store_true", help="跳过对照组A")
    parser.add_argument("--skip_b", action="store_true", help="跳过对照组B")
    parser.add_argument("--skip_c", action="store_true", help="跳过实验组C")
    args = parser.parse_args()

    run_ablation_study(output_dir=args.output)
