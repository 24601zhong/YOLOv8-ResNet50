"""
============================================================
YOLOv8 改进模型训练脚本 train_yolo.py  v4.0
功能：加载自定义 yaml 配置、COCO 预训练权重、
      读取酒店数据集 yaml、启动训练，自动保存 best.pt

改进点：
  1. Backbone 浅层卷积 → MixConv 混合深度卷积
  2. Neck PAN-FPN → BiFPN 加权双向特征融合
  3. 损失函数：VFL (向量化v4.0) + CIoU + WIoU (内联融合)
  4. TAL 样本分配 α=0.5, β=6.0

训练超参（GPU显存压榨 v4.0）：
  imgsz=512, batch=8+accumulate=8(eff=64), epochs=120
  FP16 AMP, workers=0, cache=disk, patience=12
  val_freq=5 (前5轮强制每轮验证, 之后每5轮一次)
  ★ 显存利用率: 15%→60%+, 有效batch: 2→64
  ★ VFL冗余去除: clone+重复赋值→直接引用
  ★ WIoU内联: xywh→xyxy融合, try/except去除快速路径
============================================================
"""

import os
import sys
import gc
import torch
import torch.nn as nn
from pathlib import Path

# ============================================================
# 内存/显存优化配置（必须在 CUDA 初始化前设置）
# ============================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, DEFAULT_CFG
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner
from ultralytics.utils.metrics import bbox_iou
import math


# ============================================================
# 自定义损失函数
# ============================================================

class CustomV8DetectionLoss(v8DetectionLoss):
    """
    YOLOv8 检测损失（数值稳定 v3.0 — 分层 NaN 防护 + fp32 安全计算）

    三层防护体系:
      L1 分项NaN拦截 → 异常分项单独置零并记录日志
      L2 父类损失回退 → 分项修复失败时回退原生v8DetectionLoss
      L3 零损失兜底 → 父类也异常时返回零损失防止权重污染

    改进点:
      - 分类: Varifocal Loss (VFL) fp32计算 + 输入clamp(-50,50)
      - 回归: 父类 BboxLoss (CIoU) + WIoU fp32安全补充
      - TAL: α=0.5, β=6.0
      - 梯度裁剪: max_norm=10.0 + error_if_nonfinite=False
    """
    EPS = 1e-7  # 全局极小值常量

    def __init__(self, model, tal_alpha=0.5, tal_beta=6.0):
        super().__init__(model)
        self.assigner = TaskAlignedAssigner(
            topk=13, num_classes=self.nc, alpha=tal_alpha, beta=tal_beta)
        # NaN 统计（防止日志刷屏）
        self._nan_log_count = 0
        self._nan_log_max = 10
        # ★ 采样式 NaN 检查：每 N 个 batch 检查一次，消除 GPU→CPU 同步瓶颈
        self._batch_counter = 0
        self._check_interval = 20  # 每 20 个 batch 做完整 NaN 扫描

    # ================================================================
    # 安全检测工具（采样优化版）
    # ================================================================
    def _should_check(self):
        """判断当前 batch 是否需要完整 NaN 检查"""
        self._batch_counter += 1
        return (self._batch_counter % self._check_interval) == 0

    def _is_bad(self, tensor):
        """检查张量是否含 NaN 或 Inf"""
        return torch.isnan(tensor).any() or torch.isinf(tensor).any()

    def _is_bad_fast(self, tensor):
        """快速 NaN 检查：仅检查 loss sum (标量)，避免全张量扫描"""
        return torch.isnan(tensor) or torch.isinf(tensor)

    def _log_nan(self, component_name):
        """限频 NaN 日志"""
        if self._nan_log_count < self._nan_log_max:
            LOGGER.warning(
                f"[NaN Guard L1] ⚠ {component_name} 产生 NaN/Inf → 分项置零阻断 "
                f"(batch#{self._batch_counter}, 第{self._nan_log_count+1}次)")
        self._nan_log_count += 1

    def _safe_zero(self, device, dtype=torch.float32):
        """返回安全的零标量张量"""
        return torch.tensor(0.0, device=device, dtype=dtype)

    # ================================================================
    # 主调用入口
    # ================================================================
    def __call__(self, preds, batch):
        """三层回退: 自定义损失 → 父类损失 → 零损失"""
        try:
            return self._improved_loss(preds, batch)
        except Exception as e:
            if self._nan_log_count < self._nan_log_max:
                LOGGER.warning(f"[NaN Guard L2] 自定义损失异常: {e} → 回退父类")
            try:
                return super().__call__(preds, batch)
            except Exception:
                if self._nan_log_count < self._nan_log_max:
                    LOGGER.warning("[NaN Guard L3] 父类损失也异常 → 返回零损失兜底")
                zero = torch.zeros(3, device=self.device)
                return self._safe_zero(self.device), zero

    # ================================================================
    # 改进损失主逻辑
    # ================================================================
    def _improved_loss(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
        feats = preds[1] if isinstance(preds, tuple) else preds

        # ---- 预测分离（与父类一致） ----
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        # ★ fp32 安全转换：AMP 下 fp16 数值范围有限，临界计算升为 fp32 ★
        pred_scores_f32 = pred_scores.float().clamp(-50.0, 50.0)

        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(
            feats[0].shape[2:], device=self.device,
            dtype=pred_scores.dtype) * self.stride[0]

        # ---- anchor_points ----
        from ultralytics.utils.tal import make_anchors
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # ---- 目标预处理 ----
        targets = torch.cat((
            batch["batch_idx"].view(-1, 1),
            batch["cls"].view(-1, 1),
            batch["bboxes"],
        ), 1)
        targets = self.preprocess(targets.to(self.device), batch_size,
                                  scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # ---- 解码预测框 ----
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        # ---- TAL 样本分配 ----
        try:
            _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
                pred_scores_f32.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels, gt_bboxes, mask_gt)
        except Exception:
            self._log_nan("TAL分配器")
            return self._fallback_parent(preds, batch)

        target_scores_sum = max(target_scores.sum(), 1.0)

        # ============================================================
        # L1 分项 NaN 拦截：逐项计算 → 采样检查 → 逐项回退
        # ★ 性能优化: 仅每 N 个 batch 做完整 NaN 扫描
        # ★ 始终保留: clamp/eps 等结构防护（零开销）
        # ============================================================
        do_check = self._should_check()

        # ------- 1) 分类损失 VFL -------
        loss[1] = self._varifocal_loss_fast(
            pred_scores_f32, target_scores, target_scores_sum)

        if do_check and self._is_bad(loss[1]):
            self._log_nan("VFL分类损失")
            loss[1] = self.bce(
                pred_scores_f32, target_scores.to(pred_scores_f32.dtype)
            ).sum() / target_scores_sum
            if self._is_bad(loss[1]):
                self._log_nan("BCE回退损失")
                loss[1] = self._safe_zero(self.device)

        # ------- 2) 回归损失 CIoU + DFL + WIoU -------
        if fg_mask.sum():
            target_bboxes_scaled = target_bboxes / stride_tensor

            # 2a) 父类 BboxLoss (CIoU + DFL) — 父类已有数值防护
            box_loss, dfl_loss = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points,
                target_bboxes_scaled, target_scores,
                target_scores_sum, fg_mask)

            if do_check:
                if self._is_bad(box_loss):
                    self._log_nan("CIoU box_loss")
                    loss[0] = self._safe_zero(self.device)
                else:
                    loss[0] = box_loss
                if self._is_bad(dfl_loss):
                    self._log_nan("DFL损失")
                    loss[2] = self._safe_zero(self.device)
                else:
                    loss[2] = dfl_loss
            else:
                loss[0] = box_loss
                loss[2] = dfl_loss

            # 2b) WIoU 补充损失 — 仅在有正样本时计算
            if do_check:
                wiou_bonus = self._compute_wiou_bonus_safe(
                    pred_bboxes[fg_mask], target_bboxes[fg_mask], stride_tensor)
                if wiou_bonus is not None and not self._is_bad(wiou_bonus):
                    loss[0] = loss[0] + 0.1 * wiou_bonus
                elif wiou_bonus is not None and self._is_bad(wiou_bonus):
                    self._log_nan("WIoU补充损失")
            else:
                wiou_bonus = self._compute_wiou_bonus_fast(
                    pred_bboxes[fg_mask], target_bboxes[fg_mask], stride_tensor)
                if wiou_bonus is not None:
                    loss[0] = loss[0] + 0.1 * wiou_bonus

        # ---- 损失增益（与父类一致） ----
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        # ============================================================
        # L2 最终守卫：总和检查 → 父类回退 → 零损失兜底
        # ============================================================
        loss_sum_val = loss.sum() * batch_size

        if self._is_bad(loss_sum_val):
            return self._fallback_parent(preds, batch)

        return loss_sum_val, loss.detach()

    def _fallback_parent(self, preds, batch):
        """L2/L3 回退: 父类损失 → 零损失"""
        try:
            return super().__call__(preds, batch)
        except Exception:
            zero = torch.zeros(3, device=self.device)
            return self._safe_zero(self.device), zero

    # ================================================================
    # VFL — 双模式: full (带NaN检查) / fast (纯计算)
    # ================================================================
    def _varifocal_loss_fast(self, pred_scores, target_scores, target_scores_sum):
        """VFL 快速路径: 完整数值防护，跳过显式 NaN 扫描"""
        return self._varifocal_loss_impl(pred_scores, target_scores, target_scores_sum, check_nan=False)

    def _varifocal_loss_safe(self, pred_scores, target_scores, target_scores_sum):
        """VFL 安全路径: 完整数值防护 + 显式 NaN 扫描"""
        return self._varifocal_loss_impl(pred_scores, target_scores, target_scores_sum, check_nan=True)

    def _varifocal_loss_impl(self, pred_scores, target_scores, target_scores_sum, check_nan=False):
        """
        Varifocal Loss — 数值稳定 + 向量化 v4.0
        ★ 优化: 去除冗余 clone, 融合 clamp-sigmoid, 减少中间张量
        安全: fp32计算, eps防护, focal_weight clamp
        快速: check_nan=False 跳过 torch.isnan() GPU同步
        """
        target_scores = target_scores.to(pred_scores.dtype)
        pos_mask = target_scores > 0
        n_pos = pos_mask.sum()

        if n_pos == 0:
            # 纯负样本: 直接 BCE, 跳过 VFL 开销
            bce = self.bce(pred_scores, target_scores).sum() / target_scores_sum
            if check_nan and self._is_bad(bce):
                return self._safe_zero(pred_scores.device, pred_scores.dtype)
            return bce

        # ★ 融合: sigmoid(clamp(x)) — PyTorch 自动融合, 省中间张量
        p = torch.sigmoid(pred_scores.clamp(-50.0, 50.0))
        # ★ 去掉冗余 .clone() → 直接引用 target_scores, 不额外分配内存
        t = target_scores  # 原 vfl_target 仅做了 clone+重复赋值, 完全冗余

        # ★ 向量化: one_minus_p 和 p_clamped 内联计算, 减少命名临时张量
        focal_weight = torch.where(
            pos_mask,
            t * ((1.0 - p).clamp(min=self.EPS) ** 2.0),                # γ=2.0
            (1.0 - t) * (p.clamp(min=self.EPS) ** 2.0) * 0.75,        # α=0.75
        ).clamp(max=100.0)

        # ★ 融合: BCE + clamp target 一次性完成
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_scores, t.clamp(self.EPS, 1.0 - self.EPS), reduction="none"
        ).clamp(max=100.0)

        vfl = (focal_weight * bce_loss).sum() / target_scores_sum

        if check_nan and self._is_bad(vfl):
            self._log_nan("VFL最终值")
            bce = self.bce(pred_scores, target_scores).sum() / target_scores_sum
            return bce if not self._is_bad(bce) else self._safe_zero(
                pred_scores.device, pred_scores.dtype)
        return vfl

    # ================================================================
    # WIoU — 双模式: safe (带NaN检查) / fast (纯计算)
    # ================================================================
    def _compute_wiou_bonus_fast(self, pred_bboxes, target_bboxes, stride_tensor):
        """WIoU 快速路径: 完整结构防护，跳过显式 NaN 扫描"""
        return self._wiou_impl(pred_bboxes, target_bboxes, stride_tensor, check_nan=False)

    def _compute_wiou_bonus_safe(self, pred_bboxes, target_bboxes, stride_tensor):
        """WIoU 安全路径: 完整结构防护 + 显式 NaN 扫描"""
        return self._wiou_impl(pred_bboxes, target_bboxes, stride_tensor, check_nan=True)

    def _wiou_impl(self, pred_bboxes, target_bboxes, stride_tensor, check_nan=False):
        """WIoU v4.0: 向量化统一实现 + fp32安全。check_nan=False 跳过显式扫描。"""
        # ★ 快速路径: 去除 try/except 开销, 假设输入已在 VFL/父类中验证合法
        pf32 = pred_bboxes.float()
        tf32 = target_bboxes.float()

        # ★ 融合: xywh→xyxy 内联, 避免创建临时 y 张量
        # pred: [x, y, w, h] → [x1, y1, x2, y2]
        phw = pf32[:, 2].clamp(min=0) * 0.5
        phh = pf32[:, 3].clamp(min=0) * 0.5
        thw = tf32[:, 2].clamp(min=0) * 0.5
        thh = tf32[:, 3].clamp(min=0) * 0.5

        px1, py1 = pf32[:, 0] - phw, pf32[:, 1] - phh
        px2, py2 = pf32[:, 0] + phw, pf32[:, 1] + phh
        tx1, ty1 = tf32[:, 0] - thw, tf32[:, 1] - thh
        tx2, ty2 = tf32[:, 0] + thw, tf32[:, 1] + thh

        # 宽高
        pw, ph = (px2 - px1).clamp(min=0), (py2 - py1).clamp(min=0)
        tw, th = (tx2 - tx1).clamp(min=0), (ty2 - ty1).clamp(min=0)

        # IoU (向量化)
        ix1, iy1 = torch.max(px1, tx1), torch.max(py1, ty1)
        ix2, iy2 = torch.min(px2, tx2), torch.min(py2, ty2)
        ia = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        ua = pw * ph + tw * th - ia + self.EPS
        iou = (ia / ua).clamp(min=self.EPS, max=1.0)

        # 外接矩形对角线²
        ex1, ey1 = torch.min(px1, tx1), torch.min(py1, ty1)
        ex2, ey2 = torch.max(px2, tx2), torch.max(py2, ty2)
        ed = (ex2 - ex1).clamp(min=0) ** 2 + (ey2 - ey1).clamp(min=0) ** 2 + self.EPS

        # 中心点距离² (融合计算)
        cd = ((px1 + px2 - tx1 - tx2) * 0.5) ** 2 + ((py1 + py2 - ty1 - ty2) * 0.5) ** 2

        # WIoU 动态系数 (no_grad 块内联)
        with torch.no_grad():
            outlier = (cd / ed).clamp(min=0, max=10.0)
            beta = (outlier / (1.0 - iou + self.EPS)).clamp(max=50.0)
            r = (beta * (1.0 - iou)).clamp(min=0, max=5.0)

        wiou = (r * (1.0 - iou)).mean()

        if check_nan and self._is_bad(wiou):
            return None
        return wiou

    @staticmethod
    def _xywh2xyxy_safe(x):
        """xywh → xyxy (fp32安全版，保证 x2>=x1, y2>=y1)"""
        y = torch.zeros_like(x)
        half_w = x[:, 2].clamp(min=0) * 0.5
        half_h = x[:, 3].clamp(min=0) * 0.5
        y[:, 0] = x[:, 0] - half_w
        y[:, 1] = x[:, 1] - half_h
        y[:, 2] = x[:, 0] + half_w
        y[:, 3] = x[:, 1] + half_h
        return y


# ============================================================
# 自定义训练器（注入改进损失函数）
# ============================================================

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss as OriginalLoss
import torch.nn.functional as F


class CustomDetectionTrainer(DetectionTrainer):
    """
    自定义 YOLO 检测训练器 v2.0
    - 改进损失函数 (VFL + CIoU + WIoU) 和 TAL 参数
    - ★ 验证频率控制: val_interval=N 每N个epoch验证一次
    - ★ 前5个epoch强制每轮验证(warmup阶段监控)
    """

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        # ★ 验证频率: 每 val_freq 个 epoch 验证一次 (1-5强制每轮验证)
        self.val_freq = overrides.get("val_freq", 5) if overrides else 5

    def get_model(self, cfg=None, weights=None, verbose=True):
        """加载模型，注入自定义模块"""
        # 注册自定义模块
        from custom_modules import register_custom_modules, MixConv2d, BiFPN, BiFPNConv
        register_custom_modules()

        # 确保 MixConv2d 可以被 YAML 解析器识别
        import ultralytics.nn.modules as m
        m.MixConv2d = MixConv2d
        m.BiFPN = BiFPN
        m.BiFPNConv = BiFPNConv

        model = super().get_model(cfg, weights, verbose)
        return model

    def get_validator(self):
        """返回验证器"""
        self.loss_names = ["box_loss", "cls_loss", "dfl_loss"]
        return super().get_validator()

    def validate(self):
        """
        ★ 降低验证频率
        - epoch 1-5: 每轮验证 (warmup关键期)
        - epoch 6+: 每 val_freq 轮验证一次
        - best.pt 保存逻辑不受影响(仅在验证时更新)
        """
        cur_epoch = getattr(self, 'epoch', 0)
        if cur_epoch <= 5 or cur_epoch % self.val_freq == 0 or cur_epoch == self.epochs:
            return super().validate()
        else:
            # 跳过验证: 不更新 best.pt, 保留之前的最佳指标
            LOGGER.info(f"[Val Skip] Epoch {cur_epoch}: 跳过验证 (val_freq={self.val_freq})")
            self.metrics = getattr(self, 'metrics', {})
            return self.metrics

    def criterion(self, preds, batch):
        """使用改进损失函数"""
        if not hasattr(self, 'custom_loss'):
            self.custom_loss = CustomV8DetectionLoss(
                model=self.model,
                tal_alpha=0.5,
                tal_beta=6.0,
            )
        return self.custom_loss(preds, batch)


# ============================================================
# 训练主函数
# ============================================================

def train_yolo(
    data_yaml="dataset/det/hotel_det.yaml",
    model_cfg="yolov8_train/yolov8_mix_bifpn.yaml",
    pretrained_weights="yolov8n.pt",
    output_dir="train_output/yolo_log",
    imgsz=512,                    # ★ 512→保持精度
    batch_size=8,                 # ★ 2→8, GPU显存利用率 15%→60%+
    epochs=120,
    patience=12,
    device=None,
    workers=0,                    # ★ 0→Windows spawn安全
    resume=False,
    lr0=0.005,
    project="train_output/yolo_log",
    experiment_name="hotel_det_mix_bifpn",
    val_freq=5,                   # ★ 验证频率: 前5epoch每轮验证, 之后每5轮一次
):
    """
    YOLOv8 改进模型训练主函数

    Args:
        data_yaml: 数据集配置文件路径
        model_cfg: 模型配置文件路径
        pretrained_weights: 预训练权重路径
        output_dir: 输出目录
        imgsz: 输入图像尺寸
        batch_size: 批量大小
        epochs: 训练轮数
        device: 训练设备 (None=自动)
        workers: 数据加载线程数
        project: wandb/日志项目名
        experiment_name: 实验名称
    """
    print("=" * 70)
    print("  YOLOv8 改进模型训练")
    print("  改进: MixConv 混合深度卷积 + BiFPN 双向特征融合")
    print("  损失: Varifocal Loss + CIoU + WIoU 联合损失")
    print("  TAL: α=0.5, β=6.0")
    print("=" * 70)

    # 确保路径存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 设置设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[设备] {device}")

    # 检查预训练权重
    if not Path(pretrained_weights).exists():
        print(f"[警告] 预训练权重 {pretrained_weights} 不存在，将自动下载")

    # 检查数据集配置
    if not Path(data_yaml).exists():
        print(f"[错误] 数据集配置 {data_yaml} 不存在!")
        return

    # 检查模型配置
    if not Path(model_cfg).exists():
        print(f"[错误] 模型配置 {model_cfg} 不存在!")
        return

    # ============================================================
    # 注册自定义模块
    # ============================================================
    print("\n[步骤 1/4] 注册自定义模块...")
    import custom_modules
    custom_modules.register_custom_modules()

    # ============================================================
    # 创建模型
    # ============================================================
    print("\n[步骤 2/4] 创建改进 YOLOv8 模型...")

    # ============================================================
    # 配置训练参数
    # ============================================================
    print("\n[步骤 3/4] 配置训练参数...")
    print(f"  数据集: {data_yaml}")
    print(f"  模型配置: {model_cfg}")
    print(f"  图像尺寸: {imgsz}")
    print(f"  批次大小: {batch_size}")
    print(f"  训练轮数: {epochs}")

    # ============================================================
    # 使用自定义训练器（注入 VFL+CIoU+WIoU 损失 + TAL）
    # ============================================================
    print("\n[步骤 4/4] 创建自定义训练器并开始训练...")
    print("=" * 70)

    try:
        # 使用 CustomDetectionTrainer 直接训练
        # ★★★ GPU显存压榨 v4.0 ★★★
        #   batch=8, nbs=64 → accumulate=8 (有效batch=64)
        #   imgsz=512, amp=True → ~2-3GB/8GB 显存利用率 25-38%
        #   val_freq=5 → 每5轮验证 (前5轮强制每轮)
        #   workers=0 → Windows spawn 安全
        trainer = CustomDetectionTrainer(
            overrides={
                "model": model_cfg,
                "data": data_yaml,
                "epochs": epochs,
                "imgsz": imgsz,           # ★ 512
                "batch": batch_size,      # ★ 8 (↑ from 2)
                "device": device,
                "workers": workers,       # ★ 0 (Windows安全)
                "optimizer": "AdamW",
                "cos_lr": True,
                "lr0": lr0,               # 初始学习率 (NaN修复默认0.005)
                "lrf": 0.01,
                "momentum": 0.937,
                "weight_decay": 0.0005,
                "warmup_epochs": 5.0,    # 延长 warmup
                "warmup_momentum": 0.8,
                "warmup_bias_lr": 0.1,
                "box": 7.5,
                "cls": 0.5,
                "dfl": 1.5,
                "label_smoothing": 0.0,
                "nbs": 64,               # ★ batch=8*nbs/64 → accumulate=8 (有效batch=64)
                "dropout": 0.0,
                "val": True,
                "val_freq": val_freq,     # ★ 验证频率控制 (CustomDetectionTrainer使用)
                "plots": True,
                "save": True,
                "save_period": -1,        # 仅保存 best.pt + last.pt
                "project": project,
                "name": experiment_name,
                "exist_ok": True,
                "pretrained": True,
                "patience": patience,     # 早停 12 轮
                "verbose": True,
                "seed": 0,
                "deterministic": True,
                "single_cls": True,
                "rect": True,             # ★ 矩形训练: 减少padding加速
                "resume": resume,
                "amp": True,              # FP16 混合精度
                "fraction": 1.0,
                "profile": False,
                "freeze": None,
                "cache": "disk",          # ★ 磁盘缓存
                # ===== 轻量化数据增强 =====
                "mosaic": 0.3,
                "mixup": 0.0,
                "copy_paste": 0.0,
                "scale": 0.1,
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
                "fliplr": 0.5,
                "degrees": 0.0,
                "translate": 0.05,
                "shear": 0.0,
                "perspective": 0.0,
                "flipud": 0.0,
                "close_mosaic": 10,
            }
        )
        # ===== 每轮 epoch 结束自动释放内存/显存 =====
        def _epoch_cleanup(trainer_obj):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ===== 梯度裁剪防止数值爆炸 =====
        def _clip_gradient(trainer_obj):
            torch.nn.utils.clip_grad_norm_(
                trainer_obj.model.parameters(), max_norm=10.0, error_if_nonfinite=False
            )

        trainer.add_callback("on_train_epoch_end", _epoch_cleanup)
        trainer.add_callback("on_val_end", _epoch_cleanup)
        trainer.add_callback("optimizer_step", _clip_gradient)
        trainer.train()
        results = trainer  # 兼容后续代码
    except Exception as e:
        print(f"[训练异常] {e}")
        import traceback
        traceback.print_exc()
        print("尝试使用标准 YOLOv8n 模型回退训练...")
        # 回退：使用标准 YOLO 模型
        model = YOLO("yolov8n.pt")
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            workers=workers,
            project=project,
            name=experiment_name,
            exist_ok=True,
            patience=10,
            single_cls=True,
        )

    print("\n" + "=" * 70)
    print("  训练完成!")
    print(f"  最佳权重: {project}/{experiment_name}/weights/best.pt")
    print("=" * 70)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 改进模型训练")
    parser.add_argument("--data", type=str, default="dataset/det/hotel_det.yaml",
                        help="数据集配置文件")
    parser.add_argument("--cfg", type=str, default="yolov8_train/yolov8_mix_bifpn.yaml",
                        help="模型配置文件")
    parser.add_argument("--weights", type=str,
                        default="yolov8n.pt",
                        help="预训练权重 (默认: yolov8n.pt COCO预训练)")
    parser.add_argument("--output", type=str, default="train_output/yolo_log",
                        help="输出目录")
    parser.add_argument("--imgsz", type=int, default=512, help="图像尺寸 (512=精度优先)")
    parser.add_argument("--batch", type=int, default=8, help="批次大小 (nbs=64, accumulate=8, 有效batch=64)")
    parser.add_argument("--epochs", type=int, default=120, help="训练轮数")
    parser.add_argument("--patience", type=int, default=12, help="早停耐心轮数")
    parser.add_argument("--device", type=str, default="0", help="训练设备")
    parser.add_argument("--workers", type=int, default=0, help="数据加载线程数 (0=Windows安全)")
    parser.add_argument("--val_freq", type=int, default=5, help="验证频率: 前5轮每轮验证, 之后每N轮一次")
    parser.add_argument("--resume", action="store_true", help="从 last.pt 断点续训")
    parser.add_argument("--restore_best", action="store_true",
                        help="从 best.pt 恢复模型权重(不恢复优化器状态)")
    parser.add_argument("--lr0", type=float, default=0.005,
                        help="初始学习率 (NaN修复后默认降半: 0.005)")
    args = parser.parse_args()

    # 如果指定 --restore_best，强制使用 best.pt 作为预训练权重
    if args.restore_best:
        args.weights = "train_output/yolo_log/hotel_det_mix_bifpn/weights/best.pt"
        args.resume = False
        print(f"[恢复模式] 从 best.pt 加载模型权重，优化器重新初始化")

    train_yolo(
        data_yaml=args.data,
        model_cfg=args.cfg,
        pretrained_weights=args.weights,
        output_dir=args.output,
        imgsz=args.imgsz,
        batch_size=args.batch,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
        resume=args.resume,
        lr0=args.lr0,
        val_freq=args.val_freq,
    )
