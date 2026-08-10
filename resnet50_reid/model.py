# -*- coding: utf-8 -*-
"""
改进ResNet50行人重识别模型 resnet50_reid/model.py
三层强制改进：
  1. 标准ResNet50 Bottleneck主干
  2. CBAM混合注意力(Layer3/Layer4后，空间注意力核5x5)
  3. 空洞卷积(Layer3膨胀率d=2, Layer4膨胀率d=4)
  4. 全局平均池化输出2048维特征向量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ============================================================
# 基础模块
# ============================================================

class ConvBNAct(nn.Module):
    """卷积+BN+激活"""

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ============================================================
# CBAM 混合注意力模块 (通道注意力 + 空间注意力)
# ============================================================

class ChannelAttention(nn.Module):
    """通道注意力"""

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid_channels = max(in_channels // reduction, 8)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class SpatialAttention(nn.Module):
    """空间注意力 - 使用5x5卷积核(降低背景噪声)"""

    def __init__(self, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(concat))
        return x * out


class CBAM(nn.Module):
    """CBAM混合注意力模块"""

    def __init__(self, in_channels, reduction=16, spatial_kernel=5):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, x):
        # 先通道注意力，后空间注意力
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# ============================================================
# 改进Bottleneck (支持空洞卷积)
# ============================================================

class ImprovedBottleneck(nn.Module):
    """改进的Bottleneck块 - 支持空洞卷积"""

    def __init__(self, in_channels, out_channels, stride=1,
                 downsample=None, dilation=1):
        super().__init__()
        # 1x1卷积降维
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 4)

        # 3x3卷积(支持空洞)
        self.conv2 = nn.Conv2d(
            out_channels // 4, out_channels // 4, 3,
            stride=stride, padding=dilation, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels // 4)

        # 1x1卷积升维
        self.conv3 = nn.Conv2d(out_channels // 4, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


# ============================================================
# 改进ResNet50重识别模型
# ============================================================

class ImprovedResNet50ReID(nn.Module):
    """
    改进ResNet50行人重识别模型
    - 主干: 标准50层残差网络
    - CBAM: Layer3/Layer4后串联(空间注意力5x5)
    - 空洞卷积: Layer3膨胀率2, Layer4膨胀率4
    - 输出: 2048维特征向量
    """

    def __init__(self, num_classes=1000, feat_dim=2048):
        """
        :param num_classes: 训练时的分类类别数(人员ID数)
        :param feat_dim: 输出特征维度
        """
        super().__init__()
        self.feat_dim = feat_dim

        # ===== 初始卷积层 =====
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # ===== Layer1: 标准Bottleneck x3 =====
        self.layer1 = self._make_layer(64, 256, 3, stride=1)

        # ===== Layer2: 标准Bottleneck x4 =====
        self.layer2 = self._make_layer(256, 512, 4, stride=2)

        # ===== Layer3: 空洞卷积(d=2) + CBAM =====
        self.layer3 = self._make_layer(512, 1024, 6, stride=2, dilation=2)
        self.cbam3 = CBAM(1024, reduction=16, spatial_kernel=5)

        # ===== Layer4: 空洞卷积(d=4) + CBAM =====
        self.layer4 = self._make_layer(1024, 2048, 3, stride=1, dilation=4)
        self.cbam4 = CBAM(2048, reduction=16, spatial_kernel=5)

        # ===== 全局平均池化 =====
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        # ===== 特征投影层 =====
        self.fc_feature = nn.Linear(2048, feat_dim)
        self.bn_feature = nn.BatchNorm1d(feat_dim)

        # ===== 分类头(训练时使用) =====
        self.fc_classifier = nn.Linear(feat_dim, num_classes)

        # 权重初始化
        self._init_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks,
                    stride=1, dilation=1):
        """构建残差层"""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = [
            ImprovedBottleneck(
                in_channels, out_channels,
                stride=stride, downsample=downsample, dilation=dilation
            )
        ]

        for _ in range(1, num_blocks):
            layers.append(
                ImprovedBottleneck(
                    out_channels, out_channels,
                    stride=1, dilation=dilation
                )
            )

        return nn.Sequential(*layers)

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, training=True):
        """
        :param x: 输入图像 [B, 3, H, W]
        :param training: 是否训练模式(True=返回分类+特征, False=仅返回特征)
        """
        # 特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        # Layer3 + CBAM
        x = self.layer3(x)
        x = self.cbam3(x)

        # Layer4 + CBAM
        x = self.layer4(x)
        x = self.cbam4(x)

        # 全局平均池化 -> 2048维特征
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)

        # 特征投影
        feature = self.fc_feature(x)
        feature = self.bn_feature(feature)

        if training:
            # 训练时返回分类预测和特征
            cls_logits = self.fc_classifier(feature)
            return cls_logits, feature
        else:
            # 推理时仅返回特征向量
            return feature

    def extract_feature(self, x):
        """特征提取接口"""
        self.eval()
        with torch.no_grad():
            feature = self.forward(x, training=False)
        return feature


# ============================================================
# 损失函数
# ============================================================

class TripletLoss(nn.Module):
    """难样本挖掘Triplet Loss (margin=0.3)"""

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, features, labels):
        """
        :param features: 特征向量 [B, D]
        :param labels: 人员ID标签 [B]
        """
        batch_size = features.size(0)

        # 计算距离矩阵
        dist_mat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, batch_size)
        dist_mat = dist_mat + dist_mat.t()
        dist_mat.addmm_(1, -2, features, features.t())
        dist_mat = torch.clamp(dist_mat, min=1e-12).sqrt()

        # 构建mask
        mask = labels.expand(batch_size, batch_size).eq(labels.expand(batch_size, batch_size).t())

        # 对每个anchor，选择 hardest positive 和 hardest negative
        dist_ap, dist_an = [], []
        for i in range(batch_size):
            # 正样本(同一ID)
            positive = dist_mat[i][mask[i]]
            if len(positive) > 1:
                positive = positive[positive > 1e-8]
                if len(positive) > 0:
                    dist_ap.append(positive.max().unsqueeze(0))

            # 负样本(不同ID)
            negative = dist_mat[i][~mask[i]]
            if len(negative) > 0:
                dist_an.append(negative.min().unsqueeze(0))

        if len(dist_ap) == 0 or len(dist_an) == 0:
            return torch.tensor(0.0, device=features.device)

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # 保证dist_ap和dist_an长度一致
        min_len = min(len(dist_ap), len(dist_an))
        dist_ap = dist_ap[:min_len]
        dist_an = dist_an[:min_len]

        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


class ReIDLoss(nn.Module):
    """联合损失: L_cls + λ * L_tri (λ=1.0)"""

    def __init__(self, margin=0.3, lambda_tri=1.0):
        super().__init__()
        self.lambda_tri = lambda_tri
        self.triplet_loss = TripletLoss(margin)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, cls_logits, features, labels):
        """
        :param cls_logits: 分类预测 [B, num_classes]
        :param features: 特征向量 [B, D]
        :param labels: 人员ID标签 [B]
        """
        # 分类交叉熵损失
        cls_loss = self.ce_loss(cls_logits, labels)

        # Triplet Loss
        tri_loss = self.triplet_loss(features, labels)

        # 联合损失
        total_loss = cls_loss + self.lambda_tri * tri_loss

        return total_loss, {
            'cls_loss': cls_loss.item(),
            'tri_loss': tri_loss.item(),
            'total_loss': total_loss.item()
        }


# ============================================================
# 余弦相似度匹配器
# ============================================================

class CosineSimilarityMatcher:
    """
    余弦相似度匹配器
    - 阈值0.85: ≥0.85判定为已登记住客, <0.85判定为异常人员
    """

    def __init__(self, threshold=0.85):
        self.threshold = threshold
        self.feature_db = None  # 特征数据库 [N, D]
        self.label_db = None   # 对应的身份标签 [N]

    def build_database(self, features, labels):
        """
        构建特征数据库
        :param features: 特征张量 [N, D]
        :param labels: 身份标签 [N]
        """
        # L2归一化
        features_norm = F.normalize(features, p=2, dim=1)
        self.feature_db = features_norm.cpu()
        self.label_db = labels.cpu()
        print(f"[INFO] 特征数据库构建完成: {len(labels)} 个身份样本")

    def match(self, query_feature):
        """
        匹配查询特征
        :param query_feature: 查询特征向量 [D]
        :return: (is_matched, similarity, matched_label)
        """
        if self.feature_db is None:
            raise ValueError("请先调用build_database构建特征数据库")

        # L2归一化查询特征
        query_norm = F.normalize(query_feature.unsqueeze(0), p=2, dim=1).cpu()

        # 计算余弦相似度
        similarities = torch.mm(query_norm, self.feature_db.t()).squeeze(0)

        # 获取最大相似度
        max_sim, max_idx = similarities.max(dim=0)
        matched_label = self.label_db[max_idx].item()

        # 阈值判定
        is_matched = max_sim.item() >= self.threshold

        return is_matched, max_sim.item(), matched_label

    def batch_match(self, query_features):
        """
        批量匹配
        :param query_features: 查询特征 [M, D]
        :return: (is_matched_array, similarities, labels)
        """
        query_norm = F.normalize(query_features, p=2, dim=1).cpu()
        similarities = torch.mm(query_norm, self.feature_db.t())

        max_sims, max_idxs = similarities.max(dim=1)
        matched_labels = self.label_db[max_idxs]
        is_matched = max_sims >= self.threshold

        return is_matched, max_sims, matched_labels


# ============================================================
# 模型工厂函数
# ============================================================

def create_model(num_classes=1000, feat_dim=2048, pretrained=False, pretrained_path=None):
    """
    创建改进ResNet50重识别模型
    :param num_classes: 人员ID类别数
    :param feat_dim: 特征维度
    :param pretrained: 是否加载预训练权重
    :param pretrained_path: 预训练权重路径
    """
    model = ImprovedResNet50ReID(num_classes=num_classes, feat_dim=feat_dim)

    if pretrained and pretrained_path:
        pretrained_path = Path(pretrained_path)
        if pretrained_path.exists():
            print(f"[INFO] 加载预训练权重: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=True)
            model_dict = model.state_dict()

            # 兼容加载
            loaded = 0
            for k, v in state_dict.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    model_dict[k] = v
                    loaded += 1

            model.load_state_dict(model_dict)
            print(f"[INFO] 成功加载 {loaded} 个预训练参数")
        else:
            print(f"[WARN] 预训练权重不存在: {pretrained_path}")

    return model


def get_feature_extractor(model, device='cpu'):
    """获取特征提取器"""
    extractor = model.to(device)
    extractor.eval()
    return extractor