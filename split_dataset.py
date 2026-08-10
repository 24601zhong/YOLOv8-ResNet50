# -*- coding: utf-8 -*-
"""
YOLO数据集自动划分脚本
比例：训练集80%、验证集15%、测试集5%
自动分类images/labels下的train/val/test文件夹
"""

import os
import shutil
import argparse
import random
from pathlib import Path
from collections import defaultdict


def split_dataset(source_img_dir, source_label_dir, output_base,
                  train_ratio=0.8, val_ratio=0.15, test_ratio=0.05,
                  seed=42):
    """
    自动划分数据集为train/val/test
    :param source_img_dir: 原始图片目录
    :param source_label_dir: 原始标签目录
    :param output_base: 输出根目录
    :param train_ratio: 训练集比例
    :param val_ratio: 验证集比例
    :param test_ratio: 测试集比例
    :param seed: 随机种子
    """
    random.seed(seed)

    # 验证比例之和
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "划分比例之和必须为1.0"

    source_img_dir = Path(source_img_dir)
    source_label_dir = Path(source_label_dir)
    output_base = Path(output_base)

    # 获取所有图片文件
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [
        f for f in source_img_dir.iterdir()
        if f.suffix.lower() in img_extensions
    ]

    if not image_files:
        print(f"[ERROR] 图片目录为空: {source_img_dir}")
        return

    print(f"[INFO] 共发现 {len(image_files)} 张图片")

    # 打乱顺序
    image_files.sort()
    random.shuffle(image_files)

    # 计算划分数量
    n_total = len(image_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    # 划分列表
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]

    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

    print(f"[INFO] 划分结果: 训练集={n_train}, 验证集={n_val}, 测试集={n_test}")

    # 创建输出目录并复制
    stats = {}
    for split_name, files in splits.items():
        img_out_dir = output_base / split_name / 'images'
        label_out_dir = output_base / split_name / 'labels'
        img_out_dir.mkdir(parents=True, exist_ok=True)
        label_out_dir.mkdir(parents=True, exist_ok=True)

        img_count = 0
        label_count = 0

        for img_path in files:
            # 复制图片
            dst_img = img_out_dir / img_path.name
            shutil.copy2(img_path, dst_img)
            img_count += 1

            # 查找并复制对应标签
            label_path = source_label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                dst_label = label_out_dir / label_path.name
                shutil.copy2(label_path, dst_label)
                label_count += 1

        stats[split_name] = {
            "images": img_count,
            "labels": label_count
        }
        print(f"  {split_name}: {img_count}张图片, {label_count}个标签文件")

    return stats


def generate_yaml_config(output_base, yaml_path, dataset_name="hotel_det"):
    """
    生成YOLO数据集配置文件
    """
    output_base = Path(output_base).resolve()

    yaml_content = f"""# 酒店行人检测数据集配置文件
# 数据集名称: {dataset_name}
# 生成时间: 自动生成

path: {output_base}

train: train/images
val: val/images
test: test/images

names:
  0: person

nc: 1
"""
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"[INFO] 数据集配置文件已生成: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="YOLO数据集划分工具")
    parser.add_argument("--source_img", type=str,
                        default="Hotel_Exp/dataset/det/hotel_img/images",
                        help="原始图片目录")
    parser.add_argument("--source_label", type=str,
                        default="Hotel_Exp/dataset/det/hotel_img/labels",
                        help="原始标签目录")
    parser.add_argument("--output", type=str,
                        default="Hotel_Exp/dataset/det",
                        help="输出根目录(train/val/test将存于此处)")
    parser.add_argument("--yaml_path", type=str,
                        default="Hotel_Exp/dataset/det/hotel_det.yaml",
                        help="YOLO数据集配置文件路径")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    split_dataset(
        source_img_dir=args.source_img,
        source_label_dir=args.source_label,
        output_base=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    generate_yaml_config(args.output, args.yaml_path)


if __name__ == "__main__":
    main()