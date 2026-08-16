"""
============================================================
ArcFace 损失 + 标签平滑交叉熵
从 resnet50_reid_train/model_v2.py 复制精简版 (纯 torch, 自包含),
避免 import model_v2 时拖入 model.py 的依赖链。

ArcFaceLayer: 加性角度间隔分类器 (scale=30, margin=0.3)
LabelSmoothingCrossEntropy: 标签平滑 CE
============================================================
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLayer(nn.Module):
    """ArcFace 加性角度间隔分类器
    公式: logits = s * cos(θ + m)  (正确类别)
          logits = s * cos(θ)      (其他类别)
    参考: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    """

    def __init__(self, in_features, num_classes, scale=30.0, margin=0.3):
        super(ArcFaceLayer, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.eps = 1e-7

    def forward(self, features, labels=None):
        """features: [B, D]  (IResNet 输出, 会被 L2 归一化)
           labels:   [B]     (None 时不做 margin, 用于推理)"""
        feat_norm = F.normalize(features, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)

        cos_theta = F.linear(feat_norm, w_norm)  # [B, C]

        if labels is None:
            return cos_theta * self.scale

        cos_theta = cos_theta.clamp(-1.0 + self.eps, 1.0 - self.eps)
        sin_theta = torch.sqrt(1.0 - cos_theta ** 2)
        phi = cos_theta * self.cos_m - sin_theta * self.sin_m  # cos(θ+m)

        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        logits = one_hot * phi + (1.0 - one_hot) * cos_theta
        return logits * self.scale


class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵: y_smooth = (1-ε)*one_hot + ε/C"""

    def __init__(self, epsilon=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_probs = F.log_softmax(pred, dim=-1)
        smooth_target = (1.0 - self.epsilon) * F.one_hot(target, n_classes).float()
        smooth_target = smooth_target + self.epsilon / n_classes
        loss = -(smooth_target * log_probs).sum(dim=-1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
