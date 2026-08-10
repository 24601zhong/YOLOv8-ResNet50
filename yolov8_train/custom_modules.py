"""
============================================================
YOLOv8 自定义模块 custom_modules.py
包含 MixConv 混合深度卷积 + BiFPN 加权双向特征融合
============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# MixConv: 混合深度卷积 (MixConv2d)
# 通道分组并行执行 3×3 / 5×5 / 7×7 深度卷积
# 参考: MixConv: Mixed Depthwise Convolutional Kernels (BMVC 2019)
# ============================================================
class MixConv2d(nn.Module):
    """
    混合深度卷积：将输入通道分为多组，每组使用不同大小的卷积核
    3×3 / 5×5 / 7×7 并行，轻量化降参

    YAML 格式: MixConv2d, [out_channels, [kernel_sizes]]
    输入通道 c1 从上一层自动传入（由 ultralytics parse_model 处理）
    """

    def __init__(self, c1, c2, kernel_sizes=(3, 5, 7)):
        """
        Args:
            c1: 输入通道数
            c2: 输出通道数
            kernel_sizes: 并行卷积核大小列表
        """
        super(MixConv2d, self).__init__()
        self.c1 = int(c1)
        self.c2 = int(c2)
        self.kernel_sizes = kernel_sizes

        # 将输出通道均匀分配给每个卷积核尺寸
        n_kernels = len(kernel_sizes)
        self.split_out_channels = [
            self.c2 // n_kernels + (1 if i < self.c2 % n_kernels else 0)
            for i in range(n_kernels)
        ]

        # 输入通道也按比例分配
        self.split_in_channels = [
            max(1, self.c1 * oc // self.c2)
            for oc in self.split_out_channels
        ]

        # 为每个卷积核尺寸创建深度可分离卷积
        self.convs = nn.ModuleList()
        for i, ks in enumerate(kernel_sizes):
            in_ch = self.split_in_channels[i]
            out_ch = self.split_out_channels[i]
            if in_ch > 0 and out_ch > 0:
                self.convs.append(
                    nn.Sequential(
                        # Depthwise conv
                        nn.Conv2d(in_ch, in_ch, ks, stride=1, padding=ks // 2,
                                  groups=in_ch, bias=False),
                        # Pointwise conv
                        nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0, bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.SiLU(inplace=True),
                    )
                )
            else:
                self.convs.append(nn.Identity())

    def forward(self, x):
        # 按通道分割输入
        splits = []
        start = 0
        for in_ch in self.split_in_channels:
            end = min(start + in_ch, x.shape[1])
            splits.append(x[:, start:end, :, :])
            start = end

        # 每个分组通过不同尺寸的卷积
        outputs = []
        for i, conv in enumerate(self.convs):
            if i < len(splits):
                outputs.append(conv(splits[i]))
            else:
                break

        return torch.cat(outputs, dim=1)


# ============================================================
# BiFPN 模块
# 参考: EfficientDet: Scalable and Efficient Object Detection (CVPR 2020)
# ============================================================
class BiFPNConv(nn.Module):
    """BiFPN 中的基本卷积块：深度可分离卷积 + BN + SiLU"""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super(BiFPNConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                      padding=kernel_size // 2, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class BiFPNLayer(nn.Module):
    """
    单层 BiFPN 双向特征融合
    包含自顶向下和自底向上两条路径，使用可学习权重
    """

    def __init__(self, channels_list, feature_levels=(3, 4, 5)):
        """
        Args:
            channels_list: 各层特征图的通道数 [P3_ch, P4_ch, P5_ch]
            feature_levels: 特征层级
        """
        super(BiFPNLayer, self).__init__()
        self.feature_levels = feature_levels
        n_levels = len(channels_list)

        # 可学习融合权重（使用 ReLU 保证稳定性）
        # 自顶向下路径融合权重
        self.td_weights = nn.ParameterList([
            nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            for _ in range(n_levels - 1)
        ])

        # 自底向上路径融合权重
        self.bu_weights = nn.ParameterList([
            nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            for _ in range(n_levels - 1)
        ])

        # 特征对齐卷积
        self.td_convs = nn.ModuleList([
            BiFPNConv(channels_list[i], channels_list[i+1], kernel_size=3, stride=2)
            for i in range(n_levels - 1)
        ])

        self.bu_convs = nn.ModuleList([
            BiFPNConv(channels_list[i+1], channels_list[i], kernel_size=3, stride=1)
            for i in range(n_levels - 1)
        ])

        # 上采样
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    @staticmethod
    def _weighted_fusion(weight, *inputs):
        """加权特征融合"""
        w = F.relu(weight)
        w_sum = w.sum() + 1e-4
        w_norm = w / w_sum
        fused = sum(w_norm[i] * inputs[i] for i in range(len(inputs)))
        return fused

    def forward(self, features):
        """
        Args:
            features: 多尺度特征列表 [P3, P4, P5]，从低层到高层
        Returns:
            融合后的多尺度特征列表
        """
        n = len(features)

        # ======== 自顶向下路径 ========
        td_features = [features[-1]]  # 最高层直接传入
        for i in range(n - 2, -1, -1):
            # 上采样高层特征
            up_feat = self.upsample(td_features[0])  # 上采样最近的高层特征
            # 加权融合
            fused = self._weighted_fusion(
                self.td_weights[n - 2 - i],
                features[i],
                up_feat,
            )
            td_features.insert(0, fused)

        # ======== 自底向上路径 ========
        bu_features = [td_features[0]]  # 最低层直接传入
        for i in range(n - 1):
            # 下采样低层特征
            down_feat = self.td_convs[i](bu_features[-1])
            # 加权融合
            fused = self._weighted_fusion(
                self.bu_weights[i],
                td_features[i + 1],
                down_feat,
            )
            bu_features.append(fused)

        return bu_features


class BiFPN(nn.Module):
    """
    加权 BiFPN 双向特征融合 Neck
    替换原生 YOLOv8 PAN-FPN，强化 P3/P4 小目标特征

    输入：Backbone 输出的多尺度特征 [P3, P4, P5]
    输出：融合后的多尺度特征 [P3, P4, P5]
    """

    def __init__(self, channels_list, num_layers=3):
        """
        Args:
            channels_list: [P3_ch, P4_ch, P5_ch] 各层通道数
            num_layers: BiFPN 重复层数
        """
        super(BiFPN, self).__init__()
        self.layers = nn.ModuleList([
            BiFPNLayer(channels_list) for _ in range(num_layers)
        ])

        # 输入投影（统一通道数）
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 1, bias=False),
                nn.BatchNorm2d(ch),
                nn.SiLU(inplace=True),
            )
            for ch in channels_list
        ])

        # 输出卷积
        self.output_convs = nn.ModuleList([
            BiFPNConv(ch, ch, kernel_size=3)
            for ch in channels_list
        ])

    def forward(self, features):
        """
        Args:
            features: 多尺度特征列表 [P3, P4, P5]
        Returns:
            融合后特征列表 [P3, P4, P5]
        """
        # 输入投影
        proj_feats = [proj(f) for proj, f in zip(self.input_proj, features)]

        # 逐层 BiFPN
        feats = proj_feats
        for layer in self.layers:
            feats = layer(feats)

        # 输出卷积
        outputs = [conv(f) for conv, f in zip(self.output_convs, feats)]
        return outputs


# ============================================================
# 损失函数扩展
# ============================================================

class WIoULoss(nn.Module):
    """
    WIoU (Wise-IoU) 损失
    动态非单调聚焦机制，适配遮挡行人检测
    参考: Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism
    """

    def __init__(self, monotonic=False):
        super(WIoULoss, self).__init__()
        self.monotonic = monotonic

    def forward(self, pred, target):
        """
        Args:
            pred: 预测框 [x1, y1, x2, y2]
            target: 目标框 [x1, y1, x2, y2]
        Returns:
            WIoU loss
        """
        # 计算 IoU
        pred_area = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
        target_area = (target[..., 2] - target[..., 0]) * (target[..., 3] - target[..., 1])

        # 交集
        inter_x1 = torch.max(pred[..., 0], target[..., 0])
        inter_y1 = torch.max(pred[..., 1], target[..., 1])
        inter_x2 = torch.min(pred[..., 2], target[..., 2])
        inter_y2 = torch.min(pred[..., 3], target[..., 3])

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        union_area = pred_area + target_area - inter_area
        iou = inter_area / (union_area + 1e-7)

        # 外接矩形
        enclose_x1 = torch.min(pred[..., 0], target[..., 0])
        enclose_y1 = torch.min(pred[..., 1], target[..., 1])
        enclose_x2 = torch.max(pred[..., 2], target[..., 2])
        enclose_y2 = torch.max(pred[..., 3], target[..., 3])

        enclose_w = (enclose_x2 - enclose_x1).clamp(min=0)
        enclose_h = (enclose_y2 - enclose_y1).clamp(min=0)

        # 中心点距离
        pred_ctr_x = (pred[..., 0] + pred[..., 2]) / 2
        pred_ctr_y = (pred[..., 1] + pred[..., 3]) / 2
        target_ctr_x = (target[..., 0] + target[..., 2]) / 2
        target_ctr_y = (target[..., 1] + target[..., 3]) / 2

        ctr_dist = (pred_ctr_x - target_ctr_x) ** 2 + (pred_ctr_y - target_ctr_y) ** 2
        diag_dist = enclose_w ** 2 + enclose_h ** 2 + 1e-7

        # 离群度
        with torch.no_grad():
            outlier = ctr_dist / diag_dist

        # 动态聚焦系数
        if self.monotonic:
            r = outlier
        else:
            beta = outlier / (1 - iou + 1e-7)
            r = beta * (1 - iou)

        wiou = r * (1 - iou)
        return wiou.mean()


# ============================================================
# 辅助函数：将自定义模块注册到 ultralytics
# ============================================================

def register_custom_modules():
    """
    将自定义 MixConv 和 BiFPN 模块注册到 ultralytics 模块库
    关键：parse_model 在 ultralytics.nn.tasks 的 globals() 中查找模块名，
    因此必须注册到 tasks 模块的命名空间
    """
    try:
        import ultralytics.nn.tasks as t

        # 注册 MixConv2d 到 tasks 命名空间（parse_model 使用 globals() 查找）
        t.MixConv2d = MixConv2d

        # 注册 BiFPN 相关模块
        t.BiFPNConv = BiFPNConv
        t.BiFPN = BiFPN

        print("[注册] 自定义模块 MixConv2d, BiFPN, BiFPNConv 已注册到 ultralytics.nn.tasks")
    except ImportError as e:
        print(f"[警告] 无法注册自定义模块到 ultralytics: {e}")


if __name__ == "__main__":
    # 测试 MixConv
    x = torch.randn(1, 64, 80, 80)
    mixconv = MixConv2d(64, 64, kernel_sizes=(3, 5, 7))
    y = mixconv(x)
    print(f"MixConv: 输入 {x.shape} -> 输出 {y.shape}")
    print(f"MixConv 参数量: {sum(p.numel() for p in mixconv.parameters()):,}")

    # 测试 BiFPN
    p3 = torch.randn(1, 128, 80, 80)
    p4 = torch.randn(1, 256, 40, 40)
    p5 = torch.randn(1, 512, 20, 20)
    bifpn = BiFPN(channels_list=[128, 256, 512], num_layers=3)
    out_p3, out_p4, out_p5 = bifpn([p3, p4, p5])
    print(f"BiFPN: P3 {out_p3.shape}, P4 {out_p4.shape}, P5 {out_p5.shape}")
    print(f"BiFPN 参数量: {sum(p.numel() for p in bifpn.parameters()):,}")
