# -*- coding: utf-8 -*-
"""
酒店数据三层清洗脚本 data_clean.py
三层过滤顺序（不可调换）：
  1. 模糊图片剔除：拉普拉斯算子方差阈值30
  2. 无效帧剔除：无行人/遮挡>60%
  3. 重复图片去重：感知哈希相似度>0.95
"""

import os
import cv2
import numpy as np
import shutil
import argparse
import json
from pathlib import Path
from collections import defaultdict

try:
    import imagehash
    from PIL import Image
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False
    print("[WARN] imagehash未安装，重复检测将使用像素差分替代")


class HotelDataCleaner:
    """酒店数据三层清洗器"""

    def __init__(self, input_dir, output_dir, blur_threshold=30,
                 occlusion_threshold=0.6, hash_threshold=0.95):
        """
        :param input_dir: 原始图片目录
        :param output_dir: 清洗后输出目录
        :param blur_threshold: 模糊阈值(拉普拉斯方差)
        :param occlusion_threshold: 遮挡率阈值
        :param hash_threshold: 感知哈希重复判定阈值
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.blur_threshold = blur_threshold
        self.occlusion_threshold = occlusion_threshold
        self.hash_threshold = hash_threshold

        # 统计信息
        self.stats = {
            "total_input": 0,
            "blur_removed": 0,
            "invalid_removed": 0,
            "duplicate_removed": 0,
            "final_kept": 0
        }

    def laplacian_variance(self, image):
        """计算拉普拉斯方差(清晰度指标)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def estimate_occlusion_ratio(self, image):
        """
        估算图片遮挡率(简化版)
        使用边缘检测+形态学操作估算有效行人区域占比
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Canny边缘检测
        edges = cv2.Canny(gray, 50, 150)

        # 形态学膨胀
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # 估算有效区域占比
        valid_ratio = np.count_nonzero(dilated) / (h * w)

        # 归一化到遮挡率(边缘越少，遮挡率越高)
        occlusion_ratio = 1.0 - min(valid_ratio * 3, 1.0)
        return occlusion_ratio

    def contains_person_hint(self, image):
        """
        简化版行人存在性检测
        使用Haar级联分类器检测上半身/人脸
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 使用预训练的Haar级联分类器
        cascade_paths = [
            cv2.data.haarcascades + 'haarcascade_upperbody.xml',
            cv2.data.haarcascades + 'haarcascade_fullbody.xml',
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        ]

        for cascade_path in cascade_paths:
            if os.path.exists(cascade_path):
                cascade = cv2.CascadeClassifier(cascade_path)
                if not cascade.empty():
                    rects = cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=3,
                        minSize=(30, 30)
                    )
                    if len(rects) > 0:
                        return True

        # 备用方案：检查图片是否有足够的纹理变化
        # (全白/全黑/纯色图片判定为无行人)
        std_val = np.std(gray)
        if std_val < 5:
            return False

        return True

    def compute_phash(self, image):
        """计算感知哈希"""
        if HAS_IMAGEHASH:
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            return str(imagehash.phash(pil_img))
        else:
            # 简化版：使用8x8缩略图的均值哈希
            small = cv2.resize(image, (8, 8))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            avg = gray.mean()
            hash_bits = (gray.flatten() > avg).astype(int)
            return ''.join(str(b) for b in hash_bits)

    def hamming_similarity(self, hash1, hash2):
        """计算两个哈希的相似度(1 - 归一化汉明距离)"""
        if not hash1 or not hash2:
            return 0.0

        len_diff = abs(len(hash1) - len(hash2))
        max_len = max(len(hash1), len(hash2))
        if max_len == 0:
            return 0.0

        if len_diff > 0:
            hash1 = hash1.ljust(max_len, '0')
            hash2 = hash2.ljust(max_len, '0')

        distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
        return 1.0 - distance / max_len

    def clean(self):
        """执行三层清洗流程"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        clean_dir = self.output_dir / "images"
        clean_dir.mkdir(parents=True, exist_ok=True)

        # 获取所有图片
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = sorted([
            f for f in self.input_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ])

        self.stats["total_input"] = len(image_files)
        print(f"[INFO] 共发现 {len(image_files)} 张待清洗图片\n")

        # ========== 第一层：模糊图片剔除 ==========
        print("=" * 50)
        print("[阶段1] 模糊图片剔除 (拉普拉斯方差阈值={})".format(self.blur_threshold))
        print("=" * 50)

        sharp_images = []
        blur_count = 0

        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            lap_var = self.laplacian_variance(img)
            if lap_var < self.blur_threshold:
                blur_count += 1
                continue
            sharp_images.append((img_path, img))

        self.stats["blur_removed"] = blur_count
        print(f"  模糊图片剔除: {blur_count} 张")
        print(f"  剩余清晰图片: {len(sharp_images)} 张\n")

        # ========== 第二层：无效帧剔除 ==========
        print("=" * 50)
        print("[阶段2] 无效帧剔除 (无行人/遮挡>{}%)".format(
            int(self.occlusion_threshold * 100)))
        print("=" * 50)

        valid_images = []
        invalid_count = 0

        for img_path, img in sharp_images:
            # 检查是否包含行人特征
            has_person = self.contains_person_hint(img)

            # 估算遮挡率
            occ_ratio = self.estimate_occlusion_ratio(img)

            if not has_person or occ_ratio > self.occlusion_threshold:
                invalid_count += 1
                continue

            valid_images.append((img_path, img))

        self.stats["invalid_removed"] = invalid_count
        print(f"  无效帧剔除: {invalid_count} 张")
        print(f"  剩余有效图片: {len(valid_images)} 张\n")

        # ========== 第三层：重复图片去重 ==========
        print("=" * 50)
        print("[阶段3] 重复图片去重 (感知哈希相似度阈值={})".format(
            self.hash_threshold))
        print("=" * 50)

        # 计算所有图片的感知哈希
        hash_list = []
        for img_path, img in valid_images:
            phash = self.compute_phash(img)
            hash_list.append((img_path, img, phash))

        # 去重：贪心策略，保留第一张，后续相似度过高的丢弃
        unique_images = []
        duplicate_count = 0

        for img_path, img, phash in hash_list:
            is_duplicate = False
            for _, _, saved_hash in unique_images:
                sim = self.hamming_similarity(phash, saved_hash)
                if sim >= self.hash_threshold:
                    is_duplicate = True
                    break
            if is_duplicate:
                duplicate_count += 1
            else:
                unique_images.append((img_path, img, phash))

        self.stats["duplicate_removed"] = duplicate_count
        print(f"  重复图片剔除: {duplicate_count} 张")
        print(f"  最终保留图片: {len(unique_images)} 张\n")

        # ========== 保存清洗后的图片 ==========
        print("=" * 50)
        print("[保存] 清洗后图片输出")
        print("=" * 50)

        for img_path, img, _ in unique_images:
            dst_path = clean_dir / img_path.name
            cv2.imwrite(str(dst_path), img)

        self.stats["final_kept"] = len(unique_images)
        print(f"  已保存至: {clean_dir}")
        print(f"  共保存 {len(unique_images)} 张图片\n")

        # ========== 输出统计报告 ==========
        print("=" * 50)
        print("[完成] 数据清洗统计报告")
        print("=" * 50)
        for key, val in self.stats.items():
            print(f"  {key}: {val}")

        # 保存统计JSON
        report_path = self.output_dir / "clean_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        print(f"\n  统计报告已保存: {report_path}")

        return self.stats


def main():
    parser = argparse.ArgumentParser(description="酒店数据三层清洗工具")
    parser.add_argument("--input_dir", type=str,
                        default="Hotel_Exp/dataset/det/hotel_img/images",
                        help="输入图片目录")
    parser.add_argument("--output_dir", type=str,
                        default="Hotel_Exp/dataset/det/hotel_img",
                        help="输出目录")
    parser.add_argument("--blur_threshold", type=float, default=30.0,
                        help="模糊阈值(拉普拉斯方差)")
    parser.add_argument("--occlusion_threshold", type=float, default=0.6,
                        help="遮挡率阈值")
    parser.add_argument("--hash_threshold", type=float, default=0.95,
                        help="感知哈希重复判定阈值")

    args = parser.parse_args()

    if not Path(args.input_dir).exists():
        print(f"[ERROR] 输入目录不存在: {args.input_dir}")
        return

    cleaner = HotelDataCleaner(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        blur_threshold=args.blur_threshold,
        occlusion_threshold=args.occlusion_threshold,
        hash_threshold=args.hash_threshold
    )
    cleaner.clean()


if __name__ == "__main__":
    main()