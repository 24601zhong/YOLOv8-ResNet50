# -*- coding: utf-8 -*-
"""
YOLOv8 自定义改进模块
  1. MixConv: 混合深度卷积 (3x3/5x5/7x7并行通道分组)
  2. BiFPN: 加权双向特征融合模块
  3. Varifocal Loss (VFL)
  4. CIoU + WIoU 联合边界框损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


# ============================================================
# 模块1: MixConv 混合深度卷积
# 通道分组并行使用3x3/5x5/7x7卷积核提取多尺度特征
# ============================================================

class MixConv(nn.Module):
    """
    混合深度卷积模块
    将输入通道分组，每组使用不同核大小的深度卷积，拼接输出
    """

    def __init__(self, in_channels, out_channels, kernel_sizes=(3, 5, 7),
                 stride=1, padding='same'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.stride = stride

        # 将输入通道均匀分配到各卷积分组
        n_groups = len(kernel_sizes)
        self.group_channels = in_channels // n_groups
        self.remainder = in_channels % n_groups

        # 为每个分组创建独立的深度+逐点卷积
        self.branches = nn.ModuleList()
        for i, ks in enumerate(kernel_sizes):
            in_ch = self.group_channels + (1 if i < self.remainder else 0)

            branch = nn.Sequential(
                # 深度卷积
                nn.Conv2d(
                    in_ch, in_ch, kernel_size=ks,
                    stride=stride, padding=ks // 2 if padding == 'same' else 0,
                    groups=in_ch, bias=False
                ),
                nn.BatchNorm2d(in_ch),
                nn.SiLU(inplace=True),
                # 逐点卷积
                nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            )
            self.branches.append(branch)

        # 通道混洗(Shuffle)
        self.channel_shuffle = True

    def forward(self, x):
        # 通道分组
        x_groups = []
        start = 0
        for i, branch in enumerate(self.branches):
            ch = self.group_channels + (1 if i < self.remainder else 0)
            x_groups.append(x[:, start:start + ch, :, :])
            start += ch

        # 各分支独立卷积
        y_groups = [branch(g) for branch, g in zip(self.branches, x_groups)]

        # 相加融合(轻量化)
        out = sum(y_groups)

        # 通道混洗
        if self.channel_shuffle:
            b, c, h, w = out.shape
            out = out.view(b, n_groups := len(self.kernel_sizes),
                           c // n_groups, h, w)
            out = out.transpose(1, 2).contiguous().view(b, c, h, w)

        return out


# ============================================================
# 模块2: BiFPN 加权双向特征融合
# ============================================================

class BiFPNLayer(nn.Module):
    """
    BiFPN单层模块
    实现自上而下 + 自下而上双向特征流通路
    使用可学习归一化融合权重
    """

    def __init__(self, in_channels_list, out_channels):
        super().__init__()

        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.num_inputs = len(in_channels_list)

        # 自适应通道对齐
        self.lateral_convs = nn.ModuleList()
        for in_ch in in_channels_list:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True)
                )
            )

        # 自上而下路径的融合权重(P3/P4/P5 -> P4/P5)
        self.fpn_weights_1 = nn.ParameterList([
            nn.Parameter(torch.ones(2, dtype=torch.float32))
            for _ in range(self.num_inputs - 1)
        ])

        # 自下而上路径的融合权重(P5/P4/P3 -> P4/P3)
        self.fpn_weights_2 = nn.ParameterList([
            nn.Parameter(torch.ones(2, dtype=torch.float32))
            for _ in range(self.num_inputs - 1)
        ])

        # 输出融合权重
        self.out_weights = nn.Parameter(
            torch.ones(self.num_inputs, dtype=torch.float32)
        )

        # 输出卷积
        self.out_convs = nn.ModuleList()
        for _ in range(self.num_inputs):
            self.out_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True)
                )
            )

    def _weighted_fuse(self, inputs, weights):
        """可学习权重融合"""
        w = F.relu(torch.stack(weights))
        w_sum = w.sum() + 1e-6
        normalized_weights = w / w_sum

        out = 0
        for inp, nw in zip(inputs, normalized_weights):
            out += nw * inp
        return out

    def forward(self, inputs):
        """
        :param inputs: 多尺度特征列表 [P3, P4, P5]
        :return: 融合后的多尺度特征
        """
        # 通道对齐
        aligned = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]

        # 自适应尺寸处理
        h, w = aligned[0].shape[2:]
        target_sizes = [(h * (2 ** i), w * (2 ** i)) for i in range(self.num_inputs)]

        # 将所有特征对齐到目标尺寸
        resized = []
        for i, feat in enumerate(aligned):
            th, tw = target_sizes[i]
            if feat.shape[2:] != (th, tw):
                feat = F.interpolate(feat, size=(th, tw), mode='nearest')
            resized.append(feat)

        # 自上而下路径
        top_down = []
        for i in range(self.num_inputs - 1):
            merged = self._weighted_fuse(
                [resized[i],
                 F.interpolate(resized[i + 1],
                               size=resized[i].shape[2:],
                               mode='nearest')],
                self.fpn_weights_1[i]
            )
            top_down.append(merged)

        # 自下而上路径
        bottom_up = []
        for i in range(self.num_inputs - 1, 0, -1):
            merged = self._weighted_fuse(
                [resized[i],
                 F.interpolate(resized[i - 1],
                               size=resized[i].shape[2:],
                               mode='nearest')],
                self.fpn_weights_2[self.num_inputs - 1 - i]
            )
            bottom_up.insert(0, merged)

        # 输出融合
        outputs = []
        for i in range(self.num_inputs):
            if i == 0:
                feat = self._weighted_fuse([resized[i], top_down[i]],
                                           self.out_weights[:2])
            elif i == self.num_inputs - 1:
                feat = self._weighted_fuse([resized[i], bottom_up[i - 1]],
                                           self.out_weights[:2])
            else:
                feat = self._weighted_fuse(
                    [resized[i], top_down[i], bottom_up[i]],
                    self.out_weights[:3]
                )
            outputs.append(self.out_convs[i](feat))

        return outputs


class BiFPN Neck(nn.Module):
    """
    BiFPN颈部模块 - 堆叠多层BiFPNLayer
    """

    def __init__(self, in_channels_list, out_channels, num_layers=3):
        super().__init__()
        self.bifpn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.bifpn_layers.append(
                BiFPNLayer(in_channels_list, out_channels)
            )
            in_channels_list = [out_channels] * len(in_channels_list)

    def forward(self, inputs):
        x = inputs
        for layer in self.bifpn_layers:
            x = layer(x)
        return x


# ============================================================
# 模块3: Varifocal Loss (VFL)
# ============================================================

class VarifocalLoss(nn.Module):
    """
    Varifocal Loss - 解决密集场景正负样本不均衡
    """

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        """
        :param pred: 预测分数 [N, C]
        :param target: 目标标签 [N, C] (one-hot or soft target)
        """
        target = target.float()
        pred_sigmoid = pred.sigmoid()

        # 正样本损失
        pos_weight = target * (1 - pred_sigmoid).pow(self.gamma)
        pos_loss = -self.alpha * pos_weight * pred_sigmoid.clamp(1e-6).log()

        # 负样本损失
        neg_weight = (1 - target) * pred_sigmoid.pow(self.gamma)
        neg_loss = -(1 - self.alpha) * neg_weight * (1 - pred_sigmoid).clamp(1e-6).log()

        loss = (pos_loss + neg_loss).mean()
        return loss


# ============================================================
# 模块4: CIoU + WIoU 联合边界框损失
# ============================================================

class CIoU_WIoU_Loss(nn.Module):
    """
    CIoU + WIoU 联合损失
    动态降低低质量预测框的梯度权重
    """

    def __init__(self, beta=0.5, eps=1e-7):
        super().__init__()
        self.beta = beta
        self.eps = eps

    def _bbox_iou(self, pred, target, xywh=True):
        """计算IoU"""
        if xywh:
            # xywh -> xyxy
            px1 = pred[:, 0] - pred[:, 2] / 2
            py1 = pred[:, 1] - pred[:, 3] / 2
            px2 = pred[:, 0] + pred[:, 2] / 2
            py2 = pred[:, 1] + pred[:, 3] / 2

            tx1 = target[:, 0] - target[:, 2] / 2
            ty1 = target[:, 1] - target[:, 3] / 2
            tx2 = target[:, 0] + target[:, 2] / 2
            ty2 = target[:, 1] + target[:, 3] / 2
        else:
            px1, py1, px2, py2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
            tx1, ty1, tx2, ty2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

        # 交集
        inter_x1 = torch.max(px1, tx1)
        inter_y1 = torch.max(py1, ty1)
        inter_x2 = torch.min(px2, tx2)
        inter_y2 = torch.min(py2, ty2)

        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

        # 并集
        pred_area = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
        target_area = (tx2 - tx1).clamp(0) * (ty2 - ty1).clamp(0)
        union_area = pred_area + target_area - inter_area

        iou = inter_area / (union_area + self.eps)

        # CIoU附加项
        cw = torch.max(px2, tx2) - torch.min(px1, tx1)
        ch = torch.max(py2, ty2) - torch.min(py1, ty1)
        c2 = cw.pow(2) + ch.pow(2) + self.eps

        # 中心点距离
        rho2 = (pred[:, 0] - target[:, 0]).pow(2) + (pred[:, 1] - target[:, 1]).pow(2)

        # 宽高比
        v = (4 / 3.14159 ** 2) * (
            torch.atan(target[:, 2] / (target[:, 3] + self.eps)) -
            torch.atan(pred[:, 2] / (pred[:, 3] + self.eps))
        ).pow(2)
        alpha = v / (1 - iou + v + self.eps)

        ciou = iou - (rho2 / c2 + v * alpha)

        # WIoU: 根据质量因子动态加权
        quality = iou.detach().clamp(0)
        wiou_weight = torch.exp(-quality / self.beta)

        loss = (1 - ciou) * wiou_weight
        return loss.mean()


# ============================================================
# 模块5: TAL动态样本分配
# ============================================================

class TALAssigner:
    """
    Task-Aligned Learning动态样本分配
    alpha=0.5, beta=6.0
    """

    def __init__(self, alpha=0.5, beta=6.0):
        self.alpha = alpha
        self.beta = beta

    def assign(self, anchor_points, gt_labels, gt_bboxes,
               cls_scores, bbox_preds):
        """
        动态分配正负样本
        """
        num_anchors = anchor_points.shape[0]
        num_gts = gt_bboxes.shape[0]

        if num_gts == 0:
            return torch.zeros(num_anchors, dtype=torch.bool), \
                   torch.full((num_anchors,), -1, dtype=torch.long)

        # 计算对齐度量
        align_metric = cls_scores[:, gt_labels].pow(self.alpha) * \
                       self._bbox_quality(anchor_points, gt_bboxes).pow(self.beta)

        # 动态Top-K分配
        topk = min(num_anchors, max(1, int(num_anchors * 0.5)))
        topk_anchors = torch.zeros(num_anchors, dtype=torch.bool)
        assigned_gt = torch.full((num_anchors,), -1, dtype=torch.long)

        for gt_idx in range(num_gts):
            metric = align_metric[:, gt_idx]
            topk_idx = metric.topk(topk).indices
            topk_anchors[topk_idx] = True
            assigned_gt[topk_idx] = gt_idx

        return topk_anchors, assigned_gt

    def _bbox_quality(self, anchor_points, gt_bboxes):
        """计算边界框质量"""
        # 简化版：使用中心距离
        ap_x = anchor_points[:, 0].unsqueeze(1)
        ap_y = anchor_points[:, 1].unsqueeze(1)

        gt_cx = gt_bboxes[:, 0].unsqueeze(0)
        gt_cy = gt_bboxes[:, 1].unsqueeze(0)
        gt_w = gt_bboxes[:, 2].unsqueeze(0)
        gt_h = gt_bboxes[:, 3].unsqueeze(0)

        # 检查anchor是否在gt内
        inside = (ap_x >= gt_cx - gt_w / 2) & (ap_x <= gt_cx + gt_w / 2) & \
                 (ap_y >= gt_cy - gt_h / 2) & (ap_y <= gt_cy + gt_h / 2)

        quality = inside.float()
        return quality


# ============================================================
# 模块6: 完整的改进YOLOv8模型构建器
# ============================================================

class YOLOv8MixBiFPN(nn.Module):
    """
    改进YOLOv8模型
    - Backbone: MixConv混合深度卷积
    - Neck: BiFPN加权双向特征融合
    - Head: 保留原生Anchor-Free解耦检测头
    """

    def __init__(self, num_classes=1):
        super().__init__()

        self.num_classes = num_classes

        # ===== Backbone: MixConv替代标准卷积 =====
        self.backbone = nn.Sequential(
            # Stem层
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            # Stage 1
            MixConv(32, 64, stride=2),
            # Stage 2
            MixConv(64, 128, stride=2),
            # Stage 3
            MixConv(128, 256, stride=2),
            # Stage 4
            MixConv(256, 512, stride=2),
        )

        # ===== Neck: BiFPN替代PAN-FPN =====
        # 多尺度特征通道数
        neck_channels = [256, 512, 512]
        self.bifpn_neck = BiFPN(
            in_channels_list=neck_channels,
            out_channels=256,
            num_layers=3
        )

        # ===== Head: Anchor-Free解耦检测头 =====
        self.cls_head = nn.ModuleList()
        self.reg_head = nn.ModuleList()

        for i in range(3):  # P3, P4, P5
            self.cls_head.append(
                nn.Sequential(
                    nn.Conv2d(256, 256, 3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(256, num_classes, 1)
                )
            )
            self.reg_head.append(
                nn.Sequential(
                    nn.Conv2d(256, 256, 3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(256, 4, 1)  # left, top, right, bottom
                )
            )

        # 损失函数
        self.cls_loss_fn = VarifocalLoss()
        self.reg_loss_fn = CIoU_WIoU_Loss()

    def forward(self, x):
        """
        :param x: 输入图像 [B, 3, H, W]
        :return: (cls_scores, bbox_preds)
        """
        # Backbone特征提取
        feats = []
        feat = x
        for layer in self.backbone:
            feat = layer(feat)
            feats.append(feat)

        # 取多尺度特征(P3, P4, P5)
        p3, p4, p5 = feats[2], feats[3], feats[4]

        # BiFPN特征融合
        fused = self.bifpn_neck([p3, p4, p5])

        # Head检测
        cls_scores = []
        bbox_preds = []
        for i, feat in enumerate(fused):
            cls_scores.append(self.cls_head[i](feat))
            bbox_preds.append(self.reg_head[i](feat))

        return cls_scores, bbox_preds

    def compute_loss(self, cls_scores, bbox_preds, targets):
        """
        计算联合损失
        :param cls_scores: 分类预测列表
        :param bbox_preds: 边界框预测列表
        :param targets: 标注
        """
        # 简化的损失计算
        cls_loss = torch.tensor(0.0, device=cls_scores[0].device)
        reg_loss = torch.tensor(0.0, device=cls_scores[0].device)

        for cls_score, bbox_pred in zip(cls_scores, bbox_preds):
            # 分类损失(VFL)
            cls_score_flat = cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            cls_loss = cls_loss + self.cls_loss_fn(cls_score_flat,
                                                   torch.zeros_like(cls_score_flat))

            # 回归损失
            bbox_pred_flat = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            reg_loss = reg_loss + self.reg_loss_fn(bbox_pred_flat,
                                                   bbox_pred_flat.detach())

        total_loss = cls_loss + 2.0 * reg_loss
        return total_loss, {'cls_loss': cls_loss, 'reg_loss': reg_loss}