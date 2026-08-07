# -*- coding: utf-8 -*-
"""
YOLOv8推理测试脚本 test_yolo.py
功能：批量验证测试集、输出检测精度指标、单帧测速
量化指标：
  - mAP@0.5
  - <30x30像素小目标召回率
  - 模型总参数量
  - 模型FLOPs
  - 单帧GPU推理耗时
  - 单帧CPU推理耗时
"""

import os
import sys
import json
import time
import argparse
import yaml
import torch
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from custom_modules import YOLOv8MixBiFPN


class YOLOv8Tester:
    """YOLOv8模型测试器"""

    def __init__(self, model_path, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config

        # 加载模型
        self.model = YOLOv8MixBiFPN(num_classes=config['num_classes'])
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # 指标统计
        self.metrics = {}

        print(f"[INFO] 模型加载完成: {model_path}")
        print(f"[INFO] 设备: {self.device}")

    def _count_parameters(self):
        """统计模型参数量和FLOPs"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # FLOPs估算(简化版)
        flops = 0
        dummy_input = torch.randn(1, 3, self.config['imgsz'], self.config['imgsz']).to(self.device)
        with torch.no_grad():
            cls_scores, bbox_preds = self.model(dummy_input)
        # 简化估算: 遍历卷积层
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                out_h = dummy_input.shape[2]
                out_w = dummy_input.shape[3]
                kernel_h, kernel_w = module.kernel_size
                in_ch = module.in_channels
                out_ch = module.out_channels
                flops += 2 * out_h * out_w * in_ch * out_ch * kernel_h * kernel_w

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'flops': flops
        }

    def measure_inference_speed(self, img):
        """测量单帧推理耗时"""
        # GPU计时
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                _ = self.model(img)
            torch.cuda.synchronize()
            gpu_time = (time.time() - start) * 1000  # ms
        else:
            gpu_time = 0.0

        # CPU计时
        cpu_model = self.model.cpu()
        cpu_img = img.cpu()
        start = time.time()
        with torch.no_grad():
            _ = cpu_model(cpu_img)
        cpu_time = (time.time() - start) * 1000

        self.model.to(self.device)
        return gpu_time, cpu_time

    def detect(self, image):
        """单图推理检测"""
        h, w = image.shape[:2]

        # 预处理
        img_resized = cv2.resize(image, (self.config['imgsz'], self.config['imgsz']))
        img_tensor = torch.from_numpy(img_resized).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            cls_scores, bbox_preds = self.model(img_tensor)

        # 后处理(简化版)
        detections = self._postprocess(cls_scores, bbox_preds, (h, w))
        return detections

    def _postprocess(self, cls_scores, bbox_preds, orig_size):
        """后处理(NMS)"""
        all_detections = []

        for cls_score, bbox_pred in zip(cls_scores, bbox_preds):
            cls_score = cls_score.sigmoid()
            bbox_pred = bbox_pred.squeeze(0)

            h, w = bbox_pred.shape[1], bbox_pred.shape[2]

            for y in range(h):
                for x in range(w):
                    scores = cls_score[0, :, y, x]
                    max_score, cls_id = scores.max(), scores.argmax()

                    if max_score > self.config['conf_thres']:
                        # 获取边界框
                        l, t, r, b = bbox_pred[0, :, y, x]
                        l, t, r, b = l.item(), t.item(), r.item(), b.item()

                        # 转换回原图坐标
                        scale_x = orig_size[1] / self.config['imgsz']
                        scale_y = orig_size[0] / self.config['imgsz']

                        cx = (x + 0.5) * (self.config['imgsz'] / w)
                        cy = (y + 0.5) * (self.config['imgsz'] / h)

                        detections.append({
                            'bbox': [cx - l * scale_x, cy - t * scale_y,
                                     cx + r * scale_x, cy + b * scale_y],
                            'confidence': max_score.item(),
                            'class_id': cls_id.item()
                        })

        # 简化NMS
        detections = self._nms(detections, self.config['iou_thres'])
        return detections

    def _nms(self, detections, iou_thres):
        """非极大值抑制"""
        if not detections:
            return []

        boxes = torch.tensor([d['bbox'] for d in detections])
        scores = torch.tensor([d['confidence'] for d in detections])

        # 简化NMS实现
        order = scores.argsort(descending=True)
        keep = []

        while order.numel() > 0:
            i = order[0].item()
            keep.append(i)

            if order.numel() == 1:
                break

            remaining = order[1:]
            ious = self._compute_iou(boxes[i].unsqueeze(0), boxes[remaining])
            mask = ious.squeeze(0) <= iou_thres
            order = remaining[mask]

        return [detections[i] for i in keep]

    def _compute_iou(self, box1, box2):
        """计算IoU"""
        x1 = torch.max(box1[:, 0], box2[:, 0])
        y1 = torch.max(box1[:, 1], box2[:, 1])
        x2 = torch.min(box1[:, 2], box2[:, 2])
        y2 = torch.min(box1[:, 3], box2[:, 3])

        inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union = area1 + area2 - inter

        return inter / (union + 1e-6)

    def evaluate(self, test_img_dir, test_label_dir):
        """
        完整评估
        :return: 量化指标字典
        """
        test_img_dir = Path(test_img_dir)
        test_label_dir = Path(test_label_dir)

        img_files = sorted([
            f for f in test_img_dir.iterdir()
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
        ])

        print(f"[INFO] 测试图片数量: {len(img_files)}")

        # 统计容器
        all_tp = 0
        all_fp = 0
        all_fn = 0
        small_target_tp = 0
        small_target_fn = 0
        gpu_times = []
        cpu_times = []

        for img_path in img_files:
            # 读取图片
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]

            # 测量推理速度
            img_tensor = torch.from_numpy(cv2.resize(img, (self.config['imgsz'], self.config['imgsz']))).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
            gpu_time, cpu_time = self.measure_inference_speed(img_tensor)
            gpu_times.append(gpu_time)
            cpu_times.append(cpu_time)

            # 检测
            detections = self.detect(img)

            # 读取GT标签
            label_path = test_label_dir / f"{img_path.stem}.txt"
            gt_bboxes = []
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            xc, yc, bw, bh = float(parts[1]), float(parts[2]), \
                                              float(parts[3]), float(parts[4])
                            # YOLO归一化坐标转像素坐标
                            px, py, pw, ph = xc * w, yc * h, bw * w, bh * h
                            gt_bboxes.append({
                                'bbox': [px - pw / 2, py - ph / 2, px + pw / 2, py + ph / 2],
                                'is_small': pw < 30 or ph < 30  # < 30x30像素
                            })

            # 计算TP/FP/FN
            matched_gt = set()

            for det in detections:
                det_bbox = torch.tensor(det['bbox']).unsqueeze(0)
                best_iou = 0
                best_gt_idx = -1

                for gt_idx, gt in enumerate(gt_bboxes):
                    gt_bbox = torch.tensor(gt['bbox']).unsqueeze(0)
                    iou = self._compute_iou(det_bbox, gt_bbox).item()
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= 0.5 and best_gt_idx not in matched_gt:
                    all_tp += 1
                    matched_gt.add(best_gt_idx)
                    if gt_bboxes[best_gt_idx]['is_small']:
                        small_target_tp += 1
                else:
                    all_fp += 1

            all_fn += len(gt_bboxes) - len(matched_gt)
            for gt_idx, gt in enumerate(gt_bboxes):
                if gt_idx not in matched_gt and gt['is_small']:
                    small_target_fn += 1

        # 计算指标
        precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
        recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
        mAP50 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        small_recall = small_target_tp / (small_target_tp + small_target_fn) \
            if (small_target_tp + small_target_fn) > 0 else 0

        # 模型参数统计
        param_info = self._count_parameters()

        self.metrics = {
            'mAP@0.5': round(mAP50, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'small_target_recall_30x30': round(small_recall, 4),
            'total_params': param_info['total_params'],
            'trainable_params': param_info['trainable_params'],
            'flops': param_info['flops'],
            'avg_gpu_time_ms': round(np.mean(gpu_times), 2),
            'avg_cpu_time_ms': round(np.mean(cpu_times), 2),
            'gpu_time_std_ms': round(np.std(gpu_times), 2),
            'cpu_time_std_ms': round(np.std(cpu_times), 2),
            'test_samples': len(img_files),
            'true_positives': all_tp,
            'false_positives': all_fp,
            'false_negatives': all_fn
        }

        return self.metrics


def main():
    parser = argparse.ArgumentParser(description="YOLOv8模型测试脚本")
    parser.add_argument('--model_path', type=str,
                        default='Hotel_Exp/output/yolo_train_log/best.pt')
    parser.add_argument('--dataset_yaml', type=str,
                        default='Hotel_Exp/dataset/det/hotel_det.yaml')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf_thres', type=float, default=0.25)
    parser.add_argument('--iou_thres', type=float, default=0.45)
    parser.add_argument('--output', type=str,
                        default='Hotel_Exp/output/ablation_results')

    args = parser.parse_args()

    # 读取数据集配置
    with open(args.dataset_yaml, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)

    base_path = Path(dataset_config['path'])

    config = {
        'imgsz': args.imgsz,
        'conf_thres': args.conf_thres,
        'iou_thres': args.iou_thres,
        'num_classes': dataset_config.get('nc', 1),
        'class_names': dataset_config.get('names', {0: 'person'})
    }

    # 加载模型并测试
    tester = YOLOv8Tester(args.model_path, config)
    metrics = tester.evaluate(
        test_img_dir=str(base_path / dataset_config.get('test', 'test/images')),
        test_label_dir=str(base_path / 'test' / 'labels')
    )

    # 输出结果
    print("\n" + "=" * 60)
    print("[测试结果] 量化指标报告")
    print("=" * 60)
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # 保存结果
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'test_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] 指标已保存至: {output_dir / 'test_metrics.json'}")


if __name__ == '__main__':
    main()