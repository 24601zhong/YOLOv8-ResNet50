# 模型存档

> 自动整理时间: 2026-08-12

## 命名规范

`{模型类型}_{训练数据}_{关键指标名}-{值}_{子指标}-{值}_ep{最佳轮次}.{扩展名}`

---

## YOLO 检测模型 (Person Detection)

### 1. YOLOv8n_v4_HotelDet_mAP50-0696_mAP50-95-0462_ep80.pt
| 项目 | 值 |
|---|---|
| **源文件** | `train_output/yolo_log/hotel_det_mix_bifpn/weights/best.pt` |
| **模型** | YOLOv8n (v4 优化版) |
| **大小** | 6.0 MB |
| **训练数据** | Hotel Detection Dataset (单类 person) |
| **训练配置** | SGD, batch=16, imgsz=416, cos_lr, rect=True, cache=disk |
| **总轮次** | 80 epochs |
| **mAP50** | **0.6959** |
| **mAP50-95** | **0.4624** |
| **Precision** | 0.8004 |
| **Recall** | 0.5959 |
| **box_loss** | 1.0443 |
| **状态** | ✅ 训练完成 |

### 2. YOLOv8n_v1_HotelDet_mAP50-0715_ep40.pt
| 项目 | 值 |
|---|---|
| **源文件** | `train_output/yolo_log/hotel_det_mix_bifpn/weights/best_v1_epoch40_mAP715.pt` |
| **模型** | YOLOv8n (v1 基础版) |
| **大小** | 2.4 MB |
| **训练数据** | Hotel Detection Dataset (单类 person) |
| **总轮次** | 40 epochs |
| **mAP50** | **0.715** |
| **状态** | ✅ 训练完成 |

---

## ReID 重识别模型 (Person Re-Identification)

### 3. ResNet50_Market1501_TriLoss-0447_TrainAcc-0999_ep118.pth
| 项目 | 值 |
|---|---|
| **源文件** | `resnet50_reid_train/train_output/reid_log/best_market1501_improved.pth` |
| **模型** | ResNet50 (含 CBAM + BNNeck) |
| **大小** | 99 MB |
| **训练数据** | Market-1501 (751 IDs) |
| **训练配置** | Adam, batch=64, 120 epochs |
| **总轮次** | 120 epochs |
| **最佳轮次** | Epoch 118 |
| **Triplet Loss** | **-0.4469** (最佳) |
| **Train Accuracy** | **0.9991** |
| **总参数** | 25,705,636 |
| **状态** | ✅ 训练完成 |

### 4. ResNet50_MOT17Finetune_Rank1-0562_mAP-0763_Rank5-0997_ep4.pth
| 项目 | 值 |
|---|---|
| **源文件** | `resnet50_reid_train/train_output/mot17_finetune_log/best_mot17_finetune.pth` |
| **模型** | ResNet50 (从 Market-1501 预训练微调) |
| **大小** | 95 MB |
| **训练数据** | MOT17 ReID Clean (223 IDs) |
| **预训练权重** | Market-1501 Improved (模型 #3) |
| **训练配置** | lr=1e-4, batch=16, 计划80轮/实际16轮 |
| **总轮次** | 16 epochs (early stop) |
| **最佳轮次** | Epoch 4 |
| **Rank-1** | **0.5615** |
| **Rank-5** | **0.9970** |
| **mAP** | **0.7632** |
| **Val Loss** | 0.3948 |
| **总参数** | 24,624,292 |
| **状态** | ✅ 训练完成 |

### 5. ResNet50_Combined974_Rank1-0392_mAP-0302_ep20.pth ⚠️
| 项目 | 值 |
|---|---|
| **源文件** | `resnet50_reid_train/train_output/combined_v2_log/checkpoint_epoch20.pth` |
| **模型** | ResNet50 V2 (GeM Pooling + ArcFace + LabelSmoothing) |
| **大小** | 300 MB (含优化器状态) |
| **训练数据** | Market-1501 (751 IDs) + MOT17 (223 IDs) = **974 IDs** |
| **训练配置** | AdamW, PK采样(16×4), 120 epochs warmup+cosine |
| **总轮次** | 29 epochs (Run 4) |
| **最佳轮次** | Epoch 20 (checkpoint) |
| **MOT17 Rank-1** | **0.3921** |
| **MOT17 mAP** | **0.3019** |
| **状态** | ⚠️ 未完成 — 训练中断于 epoch 29/120 |

> **⚠️ 注意**: Combined V2 训练尚未完成（计划 120 epochs，Run 4 训练到 29 轮中断）。
> Run 5 正在训练中（PID 4820，当前 epoch 7）。
> 此前 Run 4 的最佳模型（epoch 25, Rank-1=**0.4526**, mAP=**0.4147**）
> 已被 Run 5 覆盖丢失。此 checkpoint (epoch 20) 是目前可用的最佳 Combined V2 模型。
> 等 Run 5 训练完成后，会有更好的模型替换此文件。

---

## 文件总览

| # | 文件名 | 大小 | 类型 | 关键指标 |
|---|---|---|---|---|
| 1 | YOLOv8n_v4_HotelDet_mAP50-0696_mAP50-95-0462_ep80.pt | 6.0 MB | 检测 | mAP50=0.696 |
| 2 | YOLOv8n_v1_HotelDet_mAP50-0715_ep40.pt | 2.4 MB | 检测 | mAP50=0.715 |
| 3 | ResNet50_Market1501_TriLoss-0447_TrainAcc-0999_ep118.pth | 99 MB | ReID | TriLoss=-0.447 |
| 4 | ResNet50_MOT17Finetune_Rank1-0562_mAP-0763_Rank5-0997_ep4.pth | 95 MB | ReID | Rank-1=0.562 |
| 5 | ResNet50_Combined974_Rank1-0392_mAP-0302_ep20.pth | 300 MB | ReID | Rank-1=0.392 |
