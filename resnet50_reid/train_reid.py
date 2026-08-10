# -*- coding: utf-8 -*-
"""
改进ResNet50行人重识别训练脚本 train_reid.py
训练流程：
  1. 预训练: Market-1501公开数据集训练
  2. 微调: 酒店专属reid数据集微调
固定参数:
  - 优化器: Adam
  - 学习率: 余弦退火
  - epochs: 120
  - 损失: L_cls + 1.0 * L_tri (margin=0.3)
"""

import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import ImprovedResNet50ReID, ReIDLoss, create_model


# ============================================================
# 重识别数据集
# ============================================================

class ReIDDataset(Dataset):
    """行人重识别数据集(兼容Market-1501和自定义格式)"""

    def __init__(self, data_root, is_train=True, img_size=(256, 128)):
        """
        :param data_root: 数据集根目录
        :param is_train: 是否训练模式
        :param img_size: 输入尺寸 (H, W)
        """
        self.data_root = Path(data_root)
        self.is_train = is_train
        self.img_size = img_size

        # 收集样本
        self.samples = []
        self.id_map = {}  # 人员ID -> 索引

        self._scan_dataset()

    def _scan_dataset(self):
        """扫描数据集目录结构"""
        # 支持Market-1501格式或自定义格式
        # 格式1: data_root/train/XXX.jpg (文件名: 摄像头ID_人员ID_帧序号.jpg)
        # 格式2: data_root/person_id/xxx.jpg
        train_dir = self.data_root / ('train' if self.is_train else 'test')

        if not train_dir.exists():
            # 格式2: 按人员ID分文件夹
            for person_dir in sorted(self.data_root.iterdir()):
                if person_dir.is_dir():
                    person_id = person_dir.name
                    if person_id not in self.id_map:
                        self.id_map[person_id] = len(self.id_map)

                    for img_file in sorted(person_dir.iterdir()):
                        if img_file.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                            self.samples.append({
                                'path': img_file,
                                'person_id': self.id_map[person_id],
                                'camera_id': 0,
                                'frame_id': 0
                            })
        else:
            # 格式1: Market-1501文件名格式
            for img_file in sorted(train_dir.iterdir()):
                if img_file.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                    # 解析文件名: 0001_c1s1_001234_01.jpg
                    stem = img_file.stem
                    parts = stem.split('_')
                    person_id = int(parts[0]) if parts else 0
                    camera_str = parts[1] if len(parts) > 1 else 'c0'
                    camera_id = int(camera_str[1]) if len(camera_str) > 1 and camera_str.startswith('c') else 0

                    if person_id not in self.id_map:
                        self.id_map[person_id] = len(self.id_map)

                    self.samples.append({
                        'path': img_file,
                        'person_id': self.id_map[person_id],
                        'camera_id': camera_id,
                        'frame_id': 0
                    })

        print(f"[INFO] 数据集加载完成: {len(self.samples)} 张图片, "
              f"{len(self.id_map)} 个人员ID")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = cv2.imread(str(sample['path']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 预处理
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]))  # (W, H)

        # 数据增强(训练时)
        if self.is_train:
            img = self._augment(img)

        # 归一化
        img = img.astype(np.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW

        return img, sample['person_id']

    def _augment(self, img):
        """训练时数据增强"""
        # 随机水平翻转
        if np.random.random() < 0.5:
            img = cv2.flip(img, 1)

        # 随机擦除
        if np.random.random() < 0.5:
            h, w = img.shape[:2]
            sh = np.random.randint(0, h // 5)
            sw = np.random.randint(0, w // 5)
            y = np.random.randint(0, h - sh)
            x = np.random.randint(0, w - sw)
            img[y:y + sh, x:x + sw] = 0

        return img


# ============================================================
# 训练器
# ============================================================

class ReIDTrainer:
    """重识别模型训练器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建输出目录
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 构建数据集
        train_dataset = ReIDDataset(
            config['data_root'], is_train=True,
            img_size=(config['img_h'], config['img_w'])
        )
        val_dataset = ReIDDataset(
            config['data_root'], is_train=False,
            img_size=(config['img_h'], config['img_w'])
        )

        self.num_classes = len(train_dataset.id_map)

        # 构建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            pin_memory=True,
            drop_last=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )

        # 构建模型
        self.model = ImprovedResNet50ReID(
            num_classes=self.num_classes,
            feat_dim=config['feat_dim']
        )
        self.model.to(self.device)

        # 加载预训练权重(微调阶段)
        if config.get('pretrained_path'):
            self._load_pretrained(config['pretrained_path'])

        # 优化器: Adam
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )

        # 余弦退火学习率调度
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=config['lrf']
        )

        # 损失函数
        self.criterion = ReIDLoss(
            margin=config['margin'],
            lambda_tri=config['lambda_tri']
        )

        # 训练历史
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'best_acc': 0.0
        }

        print(f"[INFO] 设备: {self.device}")
        print(f"[INFO] 人员ID数: {self.num_classes}")
        print(f"[INFO] 模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")

    def _load_pretrained(self, path):
        """加载预训练权重"""
        path = Path(path)
        if not path.exists():
            print(f"[WARN] 预训练权重不存在: {path}")
            return

        print(f"[INFO] 加载预训练权重: {path}")
        state_dict = torch.load(path, map_location='cpu', weights_only=True)

        # 过滤分类头(因为类别数可能不同)
        model_dict = self.model.state_dict()
        loaded = 0
        skipped = 0

        for k, v in state_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    model_dict[k] = v
                    loaded += 1
                else:
                    skipped += 1

        self.model.load_state_dict(model_dict)
        print(f"[INFO] 加载 {loaded} 个参数, 跳过 {skipped} 个(尺寸不匹配)")

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in self.train_loader:
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            # 前向传播
            cls_logits, features = self.model(imgs, training=True)

            # 计算损失
            loss, loss_dict = self.criterion(cls_logits, features, labels)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            _, preds = cls_logits.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        return total_loss / len(self.train_loader), correct / max(total, 1)

    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                cls_logits, features = self.model(imgs, training=True)
                loss, _ = self.criterion(cls_logits, features, labels)

                total_loss += loss.item()
                _, preds = cls_logits.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

        return total_loss / len(self.val_loader), correct / max(total, 1)

    def train(self):
        """完整训练流程"""
        print("\n" + "=" * 60)
        print(f"[开始训练] 阶段: {self.config.get('stage', 'finetune')}")
        print("=" * 60)

        start_time = time.time()

        for epoch in range(1, self.config['epochs'] + 1):
            epoch_start = time.time()

            # 训练
            train_loss, train_acc = self.train_epoch()

            # 验证
            val_loss, val_acc = self.validate()

            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录历史
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # 保存最优模型
            is_best = val_acc > self.history['best_acc']
            if is_best:
                self.history['best_acc'] = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'id_map': ReIDDataset(self.config['data_root']).id_map
                }, self.output_dir / 'best_reid.pt')

            # 日志
            epoch_time = time.time() - epoch_start
            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch [{epoch}/{self.config['epochs']}] "
                    f"| Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
                    f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
                    f"| LR: {current_lr:.6f} "
                    f"| Time: {epoch_time:.1f}s"
                    f"{' ★BEST' if is_best else ''}"
                )

        total_time = time.time() - start_time

        # 保存训练历史
        with open(self.output_dir / 'reid_training_history.json', 'w', encoding='utf-8') as f:
            json.dump({
                'config': self.config,
                'history': self.history,
                'total_time_seconds': total_time,
                'best_val_acc': self.history['best_acc']
            }, f, indent=2, ensure_ascii=False)

        print(f"\n[完成] 训练总耗时: {total_time:.1f}s")
        print(f"[完成] 最优验证准确率: {self.history['best_acc']:.4f}")
        print(f"[完成] 模型权重: {self.output_dir / 'best_reid.pt'}")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="改进ResNet50重识别训练脚本")
    parser.add_argument('--stage', type=str, default='pretrain',
                        choices=['pretrain', 'finetune'],
                        help='训练阶段: pretrain(预训练) / finetune(微调)')
    parser.add_argument('--data_root', type=str, required=True,
                        help='数据集根目录')
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='预训练权重路径(微调阶段使用)')
    parser.add_argument('--output_dir', type=str,
                        default='Hotel_Exp/output/reid_train_log')
    parser.add_argument('--img_h', type=int, default=256)
    parser.add_argument('--img_w', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--feat_dim', type=int, default=2048)
    parser.add_argument('--margin', type=float, default=0.3)
    parser.add_argument('--lambda_tri', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    config = {
        'stage': args.stage,
        'data_root': args.data_root,
        'pretrained_path': args.pretrained_path,
        'output_dir': args.output_dir,
        'img_h': args.img_h,
        'img_w': args.img_w,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'lrf': args.lrf,
        'weight_decay': args.weight_decay,
        'feat_dim': args.feat_dim,
        'margin': args.margin,
        'lambda_tri': args.lambda_tri,
        'num_workers': args.num_workers
    }

    trainer = ReIDTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()