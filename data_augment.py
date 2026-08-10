# -*- coding: utf-8 -*-
"""
离线数据增强脚本
生成扩充图片存入磁盘
增强策略：
  1. 几何变换：随机水平翻转、±10°旋转、随机裁剪、放射变换
  2. 色彩变换：亮度±0.4、对比度±0.3、高斯噪声
  3. 遮挡模拟：随机擦除矩形区域
"""

import os
import cv2
import numpy as np
import argparse
import random
from pathlib import Path

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("[WARN] albumentations未安装，使用opencv基础增强")


class HotelAugmentor:
    """酒店数据增强器"""

    def __init__(self, output_dir, augment_count=3):
        """
        :param output_dir: 增强图片输出目录
        :param augment_count: 每张原图生成的增强版本数
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.augment_count = augment_count

        # 定义增强流水线
        if HAS_ALBUMENTATIONS:
            self.transform = A.Compose([
                # 几何变换
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.5),
                A.RandomCrop(height=0.8, width=0.8, p=0.3),
                A.Affine(
                    scale=(0.8, 1.2),
                    translate_percent=(-0.1, 0.1),
                    shear=(-5, 5),
                    p=0.3
                ),
                # 色彩变换
                A.RandomBrightnessContrast(
                    brightness_limit=0.4,
                    contrast_limit=0.3,
                    p=0.5
                ),
                # 高斯噪声
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                # 运动模糊(模拟低照度)
                A.MotionBlur(blur_limit=3, p=0.2),
                # 随机擦除(模拟遮挡)
                A.CoarseDropout(
                    max_holes=4,
                    max_height=0.2,
                    max_width=0.2,
                    min_holes=1,
                    p=0.4
                ),
            ])
        else:
            self.transform = None
            print("[INFO] 使用OpenCV基础增强管线")

    def augment_single(self, image):
        """对单张图片执行增强"""
        augmented = []

        if self.transform and HAS_ALBUMENTATIONS:
            for _ in range(self.augment_count):
                augmented_img = self.transform(image=image)['image']
                augmented.append(augmented_img)
        else:
            # OpenCV基础增强
            h, w = image.shape[:2]

            # 1. 随机水平翻转
            if random.random() < 0.5:
                augmented.append(cv2.flip(image, 1))

            # 2. ±10度旋转
            angle = random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            augmented.append(cv2.warpAffine(image, M, (w, h)))

            # 3. 亮度/对比度调整
            alpha = random.uniform(0.6, 1.4)  # 亮度
            beta = random.uniform(-50, 50)    # 对比度偏移
            augmented.append(cv2.convertScaleAbs(image, alpha=alpha, beta=beta))

            # 4. 高斯噪声
            sigma = random.uniform(10, 50)
            noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
            noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            augmented.append(noisy)

            # 5. 随机擦除(模拟遮挡)
            for _ in range(max(0, self.augment_count - 4)):
                img_copy = image.copy()
                sh, sw = random.randint(30, int(h * 0.2)), random.randint(30, int(w * 0.2))
                y1, x1 = random.randint(0, h - sh), random.randint(0, w - sw)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.rectangle(img_copy, (x1, y1), (x1 + sw, y1 + sh), color, -1)
                augmented.append(img_copy)

        return augmented

    def run(self, input_dir):
        """执行批量增强"""
        input_dir = Path(input_dir)
        if not input_dir.exists():
            print(f"[ERROR] 输入目录不存在: {input_dir}")
            return

        img_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [
            f for f in input_dir.iterdir()
            if f.suffix.lower() in img_extensions
        ]

        print(f"[INFO] 共 {len(image_files)} 张原图，每张生成 {self.augment_count} 个增强版本")
        print(f"[INFO] 预计生成 {len(image_files) * self.augment_count} 张增强图片\n")

        output_images = []
        progress = 0

        for img_path in sorted(image_files):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # 保存原图副本
            output_images.append((img_path, img))

            # 生成增强版本
            augmented = self.augment_single(img)
            for i, aug_img in enumerate(augmented):
                aug_name = f"{img_path.stem}_aug_{i:02d}{img_path.suffix}"
                output_images.append((Path(aug_name), aug_img))

            progress += 1
            if progress % 100 == 0:
                print(f"  进度: {progress}/{len(image_files)}")

        # 保存所有图片
        for name, img in output_images:
            dst_path = self.output_dir / name.name
            cv2.imwrite(str(dst_path), img)

        total_saved = len(output_images)
        print(f"\n[完成] 共保存 {total_saved} 张图片至 {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="酒店数据离线增强工具")
    parser.add_argument("--input_dir", type=str,
                        default="Hotel_Exp/dataset/det/hotel_img/images",
                        help="输入图片目录")
    parser.add_argument("--output_dir", type=str,
                        default="Hotel_Exp/dataset/det/hotel_img/augmented",
                        help="增强图片输出目录")
    parser.add_argument("--count", type=int, default=3,
                        help="每张原图生成的增强版本数")

    args = parser.parse_args()

    augmentor = HotelAugmentor(args.output_dir, args.count)
    augmentor.run(args.input_dir)


if __name__ == "__main__":
    main()