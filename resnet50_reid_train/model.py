"""
============================================================
改进 ResNet50 行人重识别模型 model.py

三层固定改进：
  1. 基础主干: 标准 ResNet50 Bottleneck
  2. CBAM 注意力: Layer3、Layer4 残差块后串联，空间卷积 5×5
  3. 空洞卷积: Layer3 膨胀率 d=2, Layer4 膨胀率 d=4
  4. 移除分类 FC 层，全局平均池化输出 2048 维特征向量
============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# CBAM: Convolutional Block Attention Module
# 通道注意力 + 空间注意力（空间卷积改为 5×5）
# ============================================================
class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attn = self.sigmoid(avg_out + max_out)
        return x * attn


class SpatialAttention(nn.Module):
    """空间注意力模块（卷积核改为 5×5）"""
    def __init__(self, kernel_size=5):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 5, 7), "kernel_size must be 3, 5, or 7"
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        combined = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        attn = self.sigmoid(self.conv(combined))
        return x * attn


class CBAM(nn.Module):
    """
    CBAM 注意力模块：先通道注意力，后空间注意力
    仅串联在 Layer3、Layer4 残差块后
    """
    def __init__(self, in_channels, reduction=16, spatial_kernel=5):
        super(CBAM, self).__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# ============================================================
# 带空洞卷积的 Bottleneck 模块
# ============================================================
class BottleneckDilated(nn.Module):
    """
    ResNet Bottleneck with Dilated Convolution
    expansion = 4
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BottleneckDilated, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # 3×3 空洞卷积
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


# ============================================================
# 改进 ResNet50 完整网络
# ============================================================
class ImprovedResNet50(nn.Module):
    """
    改进 ResNet50 ReID 模型

    架构:
      conv1 -> bn1 -> relu -> maxpool
      layer1: 3×Bottleneck (stride=1)       [256, H/4, W/4]
      layer2: 4×Bottleneck (stride=2)       [512, H/8, W/8]
      layer3: 6×BottleneckDilated (d=2)     [1024, H/16, W/16] + CBAM
      layer4: 3×BottleneckDilated (d=4)     [2048, H/32, W/32] + CBAM
      gap -> 2048-dim feature vector

    输出: 2048 维特征向量
    """

    def __init__(self, num_classes=751, use_cbam=True, use_dilation=True):
        """
        Args:
            num_classes: 训练 ID 数量 (Market-1501: 751)
            use_cbam: 是否使用 CBAM 注意力
            use_dilation: 是否使用空洞卷积
        """
        super(ImprovedResNet50, self).__init__()
        self.inplanes = 64
        self.use_cbam = use_cbam
        self.use_dilation = use_dilation
        self.num_classes = num_classes

        # ========== Stem ==========
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ========== Layer1 (stride=1, 无空洞) ==========
        self.layer1 = self._make_layer(BottleneckDilated, 64, 3, stride=1, dilation=1)

        # ========== Layer2 (stride=2, 无空洞) ==========
        self.layer2 = self._make_layer(BottleneckDilated, 128, 4, stride=2, dilation=1)

        # ========== Layer3 (dilation=2) + CBAM ==========
        layer3_dilation = 2 if use_dilation else 1
        self.layer3 = self._make_layer(BottleneckDilated, 256, 6, stride=2, dilation=layer3_dilation)
        if use_cbam:
            self.cbam3 = CBAM(1024, reduction=16, spatial_kernel=5)
        else:
            self.cbam3 = nn.Identity()

        # ========== Layer4 (dilation=4) + CBAM ==========
        layer4_dilation = 4 if use_dilation else 1
        self.layer4 = self._make_layer(BottleneckDilated, 512, 3, stride=2, dilation=layer4_dilation)
        if use_cbam:
            self.cbam4 = CBAM(2048, reduction=16, spatial_kernel=5)
        else:
            self.cbam4 = nn.Identity()

        # ========== 全局平均池化 ==========
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ========== 分类头（仅训练时使用） ==========
        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)  # 无偏置
        self.classifier = nn.Linear(2048, num_classes, bias=False)

        # ========== 初始化 ==========
        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        """构建 ResNet 层"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _init_params(self):
        """参数初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # 分类器特殊初始化
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x, return_feature=False):
        """
        Args:
            x: 输入图像 [B, 3, H, W]
            return_feature: 是否仅返回特征向量（推理模式）

        Returns:
            训练模式: (分类logits, 特征向量)
            推理模式: 2048维特征向量
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layer1-2
        x = self.layer1(x)
        x = self.layer2(x)

        # Layer3 + CBAM
        x = self.layer3(x)
        x = self.cbam3(x)

        # Layer4 + CBAM
        x = self.layer4(x)
        x = self.cbam4(x)

        # 全局平均池化 -> 2048 维特征
        x = self.gap(x)
        feature = x.view(x.size(0), -1)  # [B, 2048]

        if return_feature:
            return feature

        # BN + 分类器（训练时）
        feat_bn = self.bottleneck(feature)
        logits = self.classifier(feat_bn)

        return logits, feature


# ============================================================
# 损失函数
# ============================================================
class TripletLoss(nn.Module):
    """
    难样本挖掘 Triplet Loss
    margin = 0.3
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: 特征向量 [B, feat_dim]
            targets: 标签 [B]
        """
        n = inputs.size(0)

        # 计算特征间欧氏距离矩阵
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist = dist - 2 * torch.mm(inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # 欧氏距离

        # 按 ID 分组找 hard positive / hard negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        # 预生成排除自身的 mask，避免循环内 in-place 修改影响反向传播
        eye_mask = torch.eye(n, dtype=torch.bool, device=mask.device)
        mask = mask & (~eye_mask)  # 对角线置 False（排除自身）

        # 找出每个样本的最难正样本和最难负样本
        dist_ap = []
        dist_an = []
        for i in range(n):
            pos_mask = mask[i]
            neg_mask = ~mask[i]

            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                hardest_pos = dist[i][pos_mask].max()  # 距离最大的正样本
                hardest_neg = dist[i][neg_mask].min()  # 距离最小的负样本
                dist_ap.append(hardest_pos)
                dist_an.append(hardest_neg)

        if len(dist_ap) == 0:
            return torch.tensor(0.0, device=inputs.device)

        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)

        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


class ReIDLoss(nn.Module):
    """
    ReID 联合损失
    L = L_cls + 1.0 * L_tri
    L_cls: ID 分类交叉熵
    L_tri: 难样本挖掘 Triplet Loss (margin=0.3)
    """

    def __init__(self, num_classes, margin=0.3, tri_weight=1.0):
        super(ReIDLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.tri_loss = TripletLoss(margin=margin)
        self.tri_weight = tri_weight

    def forward(self, logits, features, targets):
        """
        Args:
            logits: 分类预测 [B, num_classes]
            features: 特征向量 [B, 2048]
            targets: ID 标签 [B]
        Returns:
            total_loss, (ce_loss, tri_loss)
        """
        ce = self.ce_loss(logits, targets)
        tri = self.tri_loss(features, targets)
        total = ce + self.tri_weight * tri
        return total, (ce, tri)


# ============================================================
# 模型构建工厂函数
# ============================================================
def create_model(num_classes=751, use_cbam=True, use_dilation=True):
    """
    创建改进 ResNet50 ReID 模型

    Args:
        num_classes: 训练集 ID 数量
        use_cbam: 是否启用 CBAM 注意力
        use_dilation: 是否启用空洞卷积
    """
    model = ImprovedResNet50(
        num_classes=num_classes,
        use_cbam=use_cbam,
        use_dilation=use_dilation,
    )
    return model


def create_baseline_model(num_classes=751):
    """
    创建对照用的原生 ResNet50（无 CBAM、无空洞卷积）
    """
    model = ImprovedResNet50(
        num_classes=num_classes,
        use_cbam=False,
        use_dilation=False,
    )
    return model


if __name__ == "__main__":
    # 测试模型
    print("=" * 60)
    print("  改进 ResNet50 ReID 模型 - 结构测试")
    print("=" * 60)

    # 完整改进模型
    model = create_model(num_classes=751, use_cbam=True, use_dilation=True)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[完整改进模型] ResNet50 + CBAM + Dilated Conv")
    print(f"  总参数量:     {total_params:,}")
    print(f"  可训练参数:   {trainable_params:,}")

    # 前向传播测试
    dummy = torch.randn(4, 3, 256, 128)
    logits, features = model(dummy)
    print(f"  输入尺寸:     {dummy.shape}")
    print(f"  分类输出:     {logits.shape}")
    print(f"  特征向量:     {features.shape}")

    # 仅推理特征
    feat_only = model(dummy, return_feature=True)
    print(f"  推理特征:     {feat_only.shape}")

    # 基准模型
    baseline = create_baseline_model(num_classes=751)
    baseline_params = sum(p.numel() for p in baseline.parameters())
    print(f"\n[基准模型] 原生 ResNet50 (无改进)")
    print(f"  总参数量:     {baseline_params:,}")

    # 仅 CBAM 模型
    cbam_only = create_model(num_classes=751, use_cbam=True, use_dilation=False)
    cbam_params = sum(p.numel() for p in cbam_only.parameters())
    print(f"\n[CBAM Only 模型] ResNet50 + CBAM")
    print(f"  总参数量:     {cbam_params:,}")

    print("\n  模型结构验证完成!")
