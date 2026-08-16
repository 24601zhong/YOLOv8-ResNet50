# 模型存档

> 自动整理时间: 2026-08-13

> 🏆 **最新最佳模型 (2026-08-16)**
> - **ReID V3**（IBN-Net 两阶段）：MOT17 Rank-1 = **0.9875**（目标 > 0.9，✅ 达成）
> - **YOLOv8m V6**：部署口径 mAP50 = **0.8218**（+MOT20 密集人群，比 V5 +0.33）
> - **人脸识别 V1**（IResNet50 + ArcFace）：LFW = **99.38%**（验收 99.5%，差 0.12%）
> - YOLOv8s V5：mAP50 = 0.7652（混合口径）

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

### 8. YOLOv8m_V6_HotelDet_mAP50-0822_deploy_ep30.pt 🏆
| 项目 | 值 |
|---|---|
| **源文件** | `train_output/yolo_log/hotel_det_v6/weights/best.pt` |
| **模型** | YOLOv8m |
| **大小** | 52.0 MB |
| **训练数据** | Hotel Detection Dataset v6 (单类 person: COCO 8k + MOT17室内 1.2k + **MOT20 7.1k** + Hotel 8k) |
| **训练配置** | AdamW lr=1e-3, batch=8, workers=4, imgsz=640, close_mosaic=10 |
| **总轮次** | 30 epochs (epoch 30 仍在上行未平台) |
| **部署口径 mAP50** | **0.8218** (Hotel+MOT17+MOT20, 无 COCO) |
| **部署口径 mAP50-95** | 0.5042 |
| **混合口径 mAP50** | 0.7400 (COCO+Hotel+MOT17) |
| **分源 mAP50** | mot20=0.846 / mot17=0.778 / coco=0.736 / hotel=0.726 |
| **Precision / Recall** | 0.845 / 0.720 (部署口径) |
| **状态** | ✅ 训练完成 |

> **注**: V6 相比 V5 (0.489→0.822 部署口径) 靠「+MOT20 密集人群 + m模型 + AdamW」，
> MOT20 单源 0.499→0.846。代价是通用 COCO 场景退化 (0.842→0.736)，混合口径 0.796→0.740。
> 部署口径 recall 仍 0.72，若继续冲 0.9 需提分辨率(960)或续训。

### 6. YOLOv8s_V5_HotelDet_mAP50-0765_ep60.pt
| 项目 | 值 |
|---|---|
| **源文件** | `train_output/yolo_log/hotel_det_v5/weights/best.pt` |
| **模型** | YOLOv8s (V5 均衡数据集版) |
| **大小** | 22.5 MB |
| **训练数据** | Hotel Detection Dataset v5 (单类 person, 均衡采样) |
| **训练配置** | imgsz=640, 60 epochs |
| **总轮次** | 60 epochs |
| **mAP50** | **0.7652** |
| **mAP50-95** | **0.5284** |
| **Precision** | 0.6749 |
| **Recall** | 0.8056 |
| **状态** | ✅ 训练完成 (无 NaN) |

---

## ReID 重识别模型 (Person Re-Identification)

### 7. IBNet_V3_MOT17_Rank1-0987_mAP-0846_ep40.pth 🏆
| 项目 | 值 |
|---|---|
| **源文件** | `resnet50_reid_train/train_output/mot17_v3/best_mot17_v3.pth` |
| **模型** | IBNetResNet50-a + CBAM + Dilated + GeM + BNNeck + ArcFace |
| **大小** | 296 MB (含优化器状态) |
| **训练数据** | Stage1: Market-1501 (751 IDs) → Stage2: MOT17 全7序列 (297 IDs) |
| **输入分辨率** | 384×128 |
| **训练配置** | 两阶段: AdamW + PK采样 + LabelSmoothingCE + BatchHardTriplet |
| **总轮次** | Stage1: 80 ep / Stage2: 60 ep |
| **最佳轮次** | Stage2 Epoch 40 |
| **Rank-1** | **0.9875** ✅ (目标 > 0.9) |
| **Rank-5 / Rank-10** | 0.9931 / 0.9962 |
| **mAP** | **0.8463** |
| **总参数** | 25,705,637 |
| **状态** | ✅ 训练完成 (独立评估已确认) |

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

## 人脸识别模型 (Face Embedding)

### 9. IResNet50_ArcFace_CASIAWebFace_LFW-09938_ep20.pth
| 项目 | 值 |
|---|---|
| **源文件** | `face_arcface/output/face_embed_iresnet50.pth`（由 `last.pt` 导出，去 ArcFace 头 + 优化器） |
| **模型** | IResNet50 (512 维 embedding) + ArcFace (scale=30, margin=0.3) |
| **大小** | 174.7 MB（仅 backbone state_dict） |
| **训练数据** | CASIA-WebFace (10,572 IDs / ~49 万张 112×112) |
| **输入分辨率** | 112×112 |
| **训练配置** | SGD lr=0.1 + warmup2 + cosine, AMP fp16, LabelSmoothingCE(0.1), batch=128 |
| **总轮次** | 20 epochs |
| **LFW** | **99.38%**（验收 99.5%，差 0.12%）|
| **CFP-FF / CFP-FP** | 99.46% / 94.96% |
| **AgeDB-30 / CPLFW** | 93.98% / 89.73% |
| **总参数** | 43.63 M |
| **状态** | ✅ 训练完成（LFW 差验收 0.12%，末段仍上行，可续训补足）|

> **推理接口**: `face_arcface/face_embed.py` → `FaceEmbedder` 类（人脸图 → 512 维 L2 归一化向量 + 余弦相似度）。
> 用于「YOLO 框人 → `saved_models/yolov8n-face.pt` 框脸 → face_embed 认住客」跨天比对，与全身 ReID (V3) 融合。
> 完整 checkpoint（含 ArcFace 头 + 优化器，可续训）在 `face_arcface/output/last.pt`。

---

## 文件总览

| # | 文件名 | 大小 | 类型 | 关键指标 |
|---|---|---|---|---|
| 1 | YOLOv8n_v4_HotelDet_mAP50-0696_mAP50-95-0462_ep80.pt | 6.0 MB | 检测 | mAP50=0.696 |
| 2 | YOLOv8n_v1_HotelDet_mAP50-0715_ep40.pt | 2.4 MB | 检测 | mAP50=0.715 |
| 3 | ResNet50_Market1501_TriLoss-0447_TrainAcc-0999_ep118.pth | 99 MB | ReID | TriLoss=-0.447 |
| 4 | ResNet50_MOT17Finetune_Rank1-0562_mAP-0763_Rank5-0997_ep4.pth | 95 MB | ReID | Rank-1=0.562 |
| 5 | ResNet50_Combined974_Rank1-0392_mAP-0302_ep20.pth | 300 MB | ReID | Rank-1=0.392 |
| 6 | YOLOv8s_V5_HotelDet_mAP50-0765_ep60.pt | 22.5 MB | 检测 | mAP50=0.765 |
| 7 | IBNet_V3_MOT17_Rank1-0987_mAP-0846_ep40.pth 🏆 | 296 MB | ReID | Rank-1=0.9875 |
| 8 | YOLOv8m_V6_HotelDet_mAP50-0822_deploy_ep30.pt 🏆 | 52 MB | 检测 | 部署mAP50=0.822 |
| 9 | IResNet50_ArcFace_CASIAWebFace_LFW-09938_ep20.pth | 175 MB | 人脸 | LFW=99.38% |
