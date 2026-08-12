"""
============================================================
改进 ResNet50 V2 行人重识别模型 model_v2.py

V2 新增改进：
  1. GeM Pooling: 可学习的广义均值池化，替代 GAP
  2. ArcFace: 加性角度间隔分类器，增强特征判别力
  3. Label Smoothing: 标签平滑交叉熵，减轻过拟合
  4. CombinedReIDLoss: LabelSmoothCE + TripletLoss 联合损失

复用 model.py 的组件: CBAM, BottleneckDilated, TripletLoss
============================================================
"""

import sys
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# 确保可以 import 同目录的 model.py
sys.path.insert(0, str(Path(__file__).parent))

from model import CBAM, BottleneckDilated, TripletLoss


# ============================================================
# GeM Pooling: Generalized Mean Pooling
# ============================================================
class GeMPooling(nn.Module):
    """
    广义均值池化 (Generalized Mean Pooling)

    通过学习参数 p，在平均池化 (p=1) 和最大池化 (p→∞) 之间自适应。
    原始 GeM 论文: Fine-tuning CNN Image Retrieval Parameters
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeMPooling, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        """x: [B, C, H, W] → [B, C]"""
        # 防止负值导致 pow 出 NaN
        x = x.clamp(min=self.eps).pow(self.p)
        # 空间维度平均
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)))
        # 取 1/p 次方
        x = x.pow(1.0 / self.p)
        return x.view(x.size(0), -1)


# ============================================================
# ArcFace: 加性角度间隔分类器
# ============================================================
class ArcFaceLayer(nn.Module):
    """
    ArcFace 加性角度间隔分类器

    将特征和分类权重 L2 归一化后计算余弦相似度 cos(θ)，
    然后在正确类别的角度上加上 margin m，增大分类难度，增强特征判别力。

    公式: logits = s * cos(θ + m)  (仅对正确类别)
          logits = s * cos(θ)      (其他类别)

    参考: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    """

    def __init__(self, in_features, num_classes, scale=30.0, margin=0.3):
        """
        Args:
            in_features: 特征维度 (2048)
            num_classes: 类别数 (974 for combined)
            scale: 特征缩放因子 s
            margin: 角度间隔 m (弧度)
        """
        super(ArcFaceLayer, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin

        # 分类权重矩阵 (类中心原型)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        # 预计算三角函数常数
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)  # cos(π - m)
        self.mm = math.sin(math.pi - margin) * margin  # sin(π - m) * m

        self.eps = 1e-7

    def forward(self, features, labels=None):
        """
        Args:
            features: [B, 2048] L2 归一化后的特征
            labels:   [B] 标签 (可选; None 时不做 margin，用于推理/评估)
        Returns:
            logits:   [B, num_classes]
        """
        # L2 归一化特征和权重
        feat_norm = F.normalize(features, p=2, dim=1)        # [B, 2048]
        w_norm = F.normalize(self.weight, p=2, dim=1)        # [C, 2048]

        # 余弦相似度: cos(θ) = feat_norm · w_norm^T
        cos_theta = F.linear(feat_norm, w_norm)              # [B, C]

        # 推理/评估模式：不做 margin，直接返回 s * cos(θ)
        if labels is None:
            return cos_theta * self.scale

        # 训练模式：对正确类别加角度 margin
        cos_theta = cos_theta.clamp(-1.0 + self.eps, 1.0 - self.eps)

        # sin(θ) = √(1 - cos²(θ))
        sin_theta = torch.sqrt(1.0 - cos_theta ** 2)

        # cos(θ + m) = cosθ·cos_m - sinθ·sin_m
        phi = cos_theta * self.cos_m - sin_theta * self.sin_m  # [B, C]

        # One-hot 标签
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()  # [B, C]

        # 仅正确类别应用 margin
        logits = one_hot * phi + (1.0 - one_hot) * cos_theta

        # 缩放
        logits = logits * self.scale

        return logits


# ============================================================
# Label Smoothing Cross Entropy
# ============================================================
class LabelSmoothingCrossEntropy(nn.Module):
    """
    标签平滑交叉熵损失

    将硬标签 one-hot 替换为 soft label:
      y_smooth = (1 - ε) * y_onehot + ε / num_classes

    减轻模型对训练标签的过度自信，提升泛化能力。
    """

    def __init__(self, epsilon=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred:   [B, C] logits
            target: [B] integer labels
        Returns:
            loss: scalar
        """
        n_classes = pred.size(-1)
        log_probs = F.log_softmax(pred, dim=-1)

        # 构建平滑标签
        smooth_target = (1.0 - self.epsilon) * F.one_hot(target, n_classes).float()
        smooth_target = smooth_target + self.epsilon / n_classes

        # 负对数似然
        loss = -(smooth_target * log_probs).sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ============================================================
# 联合损失
# ============================================================
class CombinedReIDLoss(nn.Module):
    """
    V2 联合损失

    L = LabelSmoothingCE(ArcFace_logits, labels) + λ * TripletLoss(features, labels)

    CE 计算在 ArcFace logits 上，Triplet 计算在 raw features (GeM 输出) 上。
    """

    def __init__(self, num_classes, margin=0.3, tri_weight=1.0, label_smooth=0.1):
        super(CombinedReIDLoss, self).__init__()
        self.ce_loss = LabelSmoothingCrossEntropy(epsilon=label_smooth)
        self.tri_loss = TripletLoss(margin=margin)
        self.tri_weight = tri_weight

    def forward(self, logits, features, targets):
        """
        Args:
            logits:   [B, num_classes] ArcFace logits
            features: [B, 2048] raw features (for triplet)
            targets:  [B] ID labels
        Returns:
            total_loss, (ce_loss_val, tri_loss_val)
        """
        ce = self.ce_loss(logits, targets)
        tri = self.tri_loss(features, targets)
        total = ce + self.tri_weight * tri
        return total, (ce.item(), tri.item())


# ============================================================
# 改进 ResNet50 V2 完整网络
# ============================================================
class ImprovedResNet50V2(nn.Module):
    """
    改进 ResNet50 ReID V2 模型

    架构 (与 V1 相同的主干):
      conv1 -> bn1 -> relu -> maxpool
      layer1: 3×Bottleneck (stride=1)       [256, H/4, W/4]
      layer2: 4×Bottleneck (stride=2)       [512, H/8, W/8]
      layer3: 6×BottleneckDilated (d=2)     [1024, H/16, W/16] + CBAM
      layer4: 3×BottleneckDilated (d=4)     [2048, H/32, W/32] + CBAM

    V2 变化:
      - GeM Pooling 替代 GAP
      - ArcFace 替代 Linear 分类器
      - 返回 (logits, feat, bn_feat) 三件套
          feat:    raw GeM 输出 [B, 2048] — 用于 Triplet Loss
          bn_feat: BN 输出    [B, 2048] — 用于推理/ArcFace
          logits:  ArcFace 输出 [B, num_classes] — 用于 CE Loss
    """

    def __init__(self, num_classes=974, use_cbam=True, use_dilation=True,
                 gem_p_init=3.0, arc_scale=30.0, arc_margin=0.3):
        """
        Args:
            num_classes: 训练 ID 数量
            use_cbam: 是否使用 CBAM 注意力
            use_dilation: 是否使用空洞卷积
            gem_p_init: GeM 池化初始 p 值
            arc_scale: ArcFace 缩放因子 s
            arc_margin: ArcFace 角度间隔 m
        """
        super(ImprovedResNet50V2, self).__init__()
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
        self.layer3 = self._make_layer(BottleneckDilated, 256, 6, stride=2,
                                       dilation=layer3_dilation)
        if use_cbam:
            self.cbam3 = CBAM(1024, reduction=16, spatial_kernel=5)
        else:
            self.cbam3 = nn.Identity()

        # ========== Layer4 (dilation=4) + CBAM ==========
        layer4_dilation = 4 if use_dilation else 1
        self.layer4 = self._make_layer(BottleneckDilated, 512, 3, stride=2,
                                       dilation=layer4_dilation)
        if use_cbam:
            self.cbam4 = CBAM(2048, reduction=16, spatial_kernel=5)
        else:
            self.cbam4 = nn.Identity()

        # ========== GeM 池化（替代 GAP） ==========
        self.gem_pool = GeMPooling(p=gem_p_init)

        # ========== BNNeck（无偏置） ==========
        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)

        # ========== ArcFace 分类器 ==========
        self.arcface = ArcFaceLayer(2048, num_classes,
                                    scale=arc_scale, margin=arc_margin)

        # ========== 初始化 ==========
        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        """构建 ResNet 层（与 model.py 完全一致）"""
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
        """参数初始化（与 model.py 一致）"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, labels=None, return_feature=False):
        """
        Args:
            x:              输入图像 [B, 3, H, W]
            labels:         标签 [B]（训练时需要，用于 ArcFace margin）
            return_feature: 是否仅返回推理特征向量

        Returns:
            train mode:  (logits, feat, bn_feat)
            infer mode:  bn_feat (2048-dim L2-normalized)
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

        # GeM 池化 → 2048 维特征
        feat = self.gem_pool(x)                     # [B, 2048] raw feature

        if return_feature:
            # 推理模式：返回 BN 后的 L2 归一化特征
            bn_feat = self.bottleneck(feat)
            return F.normalize(bn_feat, p=2, dim=1)

        # 训练模式
        bn_feat = self.bottleneck(feat)              # [B, 2048]
        logits = self.arcface(bn_feat, labels)       # [B, num_classes]

        return logits, feat, bn_feat


# ============================================================
# 工厂函数
# ============================================================
def create_model_v2(num_classes=974, use_cbam=True, use_dilation=True,
                    arc_scale=30.0, arc_margin=0.3, gem_p_init=3.0):
    """
    创建 Improved ResNet50 V2 模型

    Args:
        num_classes: 训练集 ID 数量 (combined: 974)
        use_cbam: 是否启用 CBAM 注意力
        use_dilation: 是否启用空洞卷积
        arc_scale: ArcFace 缩放因子
        arc_margin: ArcFace 角度间隔
        gem_p_init: GeM 池化初始 p 值
    """
    model = ImprovedResNet50V2(
        num_classes=num_classes,
        use_cbam=use_cbam,
        use_dilation=use_dilation,
        gem_p_init=gem_p_init,
        arc_scale=arc_scale,
        arc_margin=arc_margin,
    )
    return model


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Improved ResNet50 V2 - 结构测试")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 974

    # ---- 1. 创建模型 ----
    model = create_model_v2(num_classes=num_classes, use_cbam=True, use_dilation=True)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[V2 完整模型] ResNet50 + CBAM + Dilated + GeM + ArcFace")
    print(f"  总参数量:     {total_params:,}")
    print(f"  可训练参数:   {trainable_params:,}")
    print(f"  类别数:       {num_classes}")

    # ---- 2. 前向传播测试 ----
    batch_size = 4
    dummy = torch.randn(batch_size, 3, 256, 128).to(device)
    dummy_labels = torch.randint(0, num_classes, (batch_size,)).to(device)

    # 训练模式
    model.train()
    logits, feat, bn_feat = model(dummy, labels=dummy_labels)
    print(f"\n[训练模式 Forward]")
    print(f"  输入:         {dummy.shape}")
    print(f"  logits:       {logits.shape}  (ArcFace output)")
    print(f"  feat:         {feat.shape}    (raw GeM, for triplet)")
    print(f"  bn_feat:      {bn_feat.shape}  (BN output, for ArcFace input)")

    # 推理模式
    model.eval()
    with torch.no_grad():
        infer_feat = model(dummy, labels=None, return_feature=True)
    print(f"\n[推理模式 Forward]")
    print(f"  infer_feat:   {infer_feat.shape}  (L2-normalized)")
    # 验证 L2 范数 ≈ 1
    norms = infer_feat.norm(p=2, dim=1)
    print(f"  L2 norms:     min={norms.min().item():.4f}, max={norms.max().item():.4f}, mean={norms.mean().item():.4f}")

    # ---- 3. 损失函数测试 ----
    criterion = CombinedReIDLoss(num_classes=num_classes, margin=0.3,
                                 tri_weight=1.0, label_smooth=0.1)
    loss, (ce_val, tri_val) = criterion(logits, feat, dummy_labels)
    print(f"\n[损失函数] CombinedReIDLoss")
    print(f"  Total Loss:   {loss.item():.4f}")
    print(f"  CE Loss:      {ce_val:.4f}")
    print(f"  Triplet Loss: {tri_val:.4f}")

    # ---- 4. 反向传播测试 ----
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters()
                    if p.grad is not None)
    print(f"\n[反向传播]")
    print(f"  Total grad norm: {grad_norm:.2f}")

    # ---- 5. GeM p 值 ----
    print(f"\n[GeM Pooling]")
    print(f"  p = {model.gem_pool.p.item():.4f}")

    # ---- 6. 组件独立测试 ----
    print(f"\n[组件独立测试]")
    # GeM Pooling
    gem = GeMPooling(p=3.0)
    gem_input = torch.randn(2, 2048, 8, 4)
    gem_out = gem(gem_input)
    print(f"  GeMPooling: {gem_input.shape} -> {gem_out.shape} [OK]")

    # LabelSmoothing
    ls_ce = LabelSmoothingCrossEntropy(epsilon=0.1)
    ls_pred = torch.randn(8, 974)
    ls_target = torch.randint(0, 974, (8,))
    ls_loss = ls_ce(ls_pred, ls_target)
    print(f"  LabelSmoothingCE: loss={ls_loss.item():.4f} [OK]")

    print(f"\n{'=' * 60}")
    print(f"  All tests passed!")
    print(f"{'=' * 60}")
