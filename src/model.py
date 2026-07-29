"""Backbone + Single Coord-Head with Spatial Attention.

Supported backbones:
  - resnet34: 512 channels, ~21M params, feature stride 32
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .config import USE_SE, HEATMAP_SIZE, HEATMAP_TEMPERATURE


def build_backbone(name, pretrained=True):
    """Build backbone and return (features_module, out_channels, out_stride)."""
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        features = nn.Sequential(*list(backbone.children())[:-2])
        out_channels = 512
        out_stride = 32
    elif name == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet34(weights=weights)
        features = nn.Sequential(*list(backbone.children())[:-2])
        out_channels = 512
        out_stride = 32
    elif name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)
        features = nn.Sequential(*list(backbone.children())[:-2])
        out_channels = 2048
        out_stride = 32
    elif name == "efficientnet_b4":
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b4(weights=weights)
        features = backbone.features
        out_channels = 1792
        out_stride = 32
    else:
        raise ValueError(f"Unsupported backbone: {name}")
    return features, out_channels, out_stride


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    """Spatial attention module — learns which spatial regions are important.

    Takes feature map → computes 2D attention weights (1 channel) → applies.
    Focuses the model on the fovea region without needing explicit detection.
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average and max across channel dimension
        avg_out = x.mean(dim=1, keepdim=True)   # [B, 1, H, W]
        max_out = x.max(dim=1, keepdim=True)[0]  # [B, 1, H, W]
        attn = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        attn = self.sigmoid(self.conv(attn))         # [B, 1, H, W]
        return x * attn


class CoordinateHead(nn.Module):
    """Coordinate regression head: feature → (x, y) normalized."""

    def __init__(self, in_channels, dropout_rate=0.3):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.global_pool(x)  # [B, C, 1, 1]
        x = x.flatten(1)         # [B, C]
        x = self.fc(x)           # [B, 2]
        return x


class FoveaLocalizationModel(nn.Module):
    """Single-head model: backbone → SE → SpatialAttn → coord.

    Simplified architecture: removed heatmap head, added spatial attention
    for implicit region-of-interest focus.
    """

    def __init__(self, backbone_name="resnet34", pretrained=True):
        super().__init__()
        self.backbone_name = backbone_name
        self.features, out_channels, self.stride = build_backbone(backbone_name, pretrained)

        # Channel attention (optional)
        self.use_se = USE_SE
        if self.use_se:
            self.se = SEBlock(out_channels)

        # Spatial attention (always on, lightweight ~100 params)
        self.spatial_attn = SpatialAttention(kernel_size=7)

        # Single coordinate head
        self.coord_head = CoordinateHead(out_channels)

    def forward(self, x):
        features = self.features(x)            # [B, C, 16, 16]
        if self.use_se:
            features = self.se(features)       # channel attention
        features = self.spatial_attn(features) # spatial attention
        coord = self.coord_head(features)      # [B, 2]
        return coord


class SimpleFoveaModel(nn.Module):
    """Simplest baseline: backbone → GAP → FC → Sigmoid.

    No SE, no SpatialAttn, no dropout.
    Isolates Backbone-only capacity for architecture ablation analysis.
    """

    def __init__(self, backbone_name="resnet18", pretrained=True):
        super().__init__()
        self.features, out_channels, self.stride = build_backbone(backbone_name, pretrained)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(out_channels, 2)

    def forward(self, x):
        x = self.features(x)         # [B, C, 16, 16]
        x = self.gap(x).flatten(1)   # [B, C]
        x = self.fc(x)               # [B, 2]
        return torch.sigmoid(x)


class HeatmapDecoder(nn.Module):
    """轻量热图解码器：将 Backbone 特征图上采样为空间热图。

    输入: [B, 512, 16, 16]  (ResNet34 输出，stride=32)
    输出: [B, 1, HEATMAP_SIZE, HEATMAP_SIZE] 热图

    架构：
      Conv → Upsample → Conv → Upsample → Conv → Conv1×1 → Sigmoid
    """

    def __init__(self, in_channels=512, heatmap_size=HEATMAP_SIZE):
        super().__init__()
        self.heatmap_size = heatmap_size

        # 计算需要的上采样倍数
        # 对于 input [16,16] → output [64,64]，需要 4× 上采样
        # 拆成两个 2× upsampling
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # →32×32
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # →64×64
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),  # →[B, 1, H, W]
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)


def soft_argmax(heatmap, temperature=HEATMAP_TEMPERATURE):
    """可微分的 Soft-Argmax 从热图解码坐标。

    Args:
        heatmap: [B, 1, H, W] 热图 (经过 Sigmoid)
        temperature: Softmax 温度参数，越小越接近 argmax

    Returns:
        coords: [B, 2] 归一化坐标 (x, y) ∈ [0, 1]
    """
    B, C, H, W = heatmap.shape
    # Flatten spatial dimensions
    hm_flat = heatmap.view(B, C, -1)  # [B, 1, H*W]
    hm_soft = F.softmax(hm_flat / temperature, dim=2)  # [B, 1, H*W]
    x_grid, y_grid = torch.meshgrid(
        torch.arange(W, device=heatmap.device),
        torch.arange(H, device=heatmap.device),
        indexing='xy',
    )
    # Normalize to [0, 1]
    x_grid = x_grid.float().view(-1) / (W - 1)
    y_grid = y_grid.float().view(-1) / (H - 1)
    # Weighted sum (expectation)
    pred_x = (hm_soft * x_grid[None, None, :]).sum(dim=2).squeeze(1)  # [B]
    pred_y = (hm_soft * y_grid[None, None, :]).sum(dim=2).squeeze(1)  # [B]
    return torch.stack([pred_x, pred_y], dim=1)  # [B, 2]


class HeatmapLocModel(nn.Module):
    """热图定位模型：Backbone → SE → Decoder → Soft-Argmax → (x, y)

    与坐标回归模型的区别：
      - 输出空间热图而非全局坐标
      - 热图对裁剪不敏感（局部峰值特性）
      - 适合级联裁剪架构
    """

    def __init__(self, backbone_name="resnet34", pretrained=True):
        super().__init__()
        self.features, out_channels, self.stride = build_backbone(backbone_name, pretrained)

        # SE 通道注意力
        self.use_se = USE_SE
        if self.use_se:
            self.se = SEBlock(out_channels)

        # 热图解码器
        self.decoder = HeatmapDecoder(out_channels)

    def forward(self, x):
        features = self.features(x)              # [B, C, 16, 16]
        if self.use_se:
            features = self.se(features)
        heatmap = self.decoder(features)         # [B, 1, H_hm, W_hm]
        coords = soft_argmax(heatmap)            # [B, 2]
        return coords, heatmap


class CoarseLocModel(nn.Module):
    """轻量粗定位模型 — 用于两阶段级联的 Stage 1。

    输入 256×256 下采样图，输出归一化坐标 (x, y)。
    仅需 ImageNet 归一化，无需预处理流水线。
    """

    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-2])

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)        # [B, 512, 8, 8]
        x = self.gap(x).flatten(1)  # [B, 512]
        x = self.fc(x)              # [B, 2]
        return x


# ========================================================================
#  U-Net Heatmap Model (ResNet encoder + skip connection decoder)
# ========================================================================

class ResNetEncoder(nn.Module):
    """ResNet 编码器，暴露中间特征用于跳跃连接。

    输出:
        s4: [B, C, H/32, W/32]   — 最深特征
        skips: [s3, s2, s1]      — 跳跃连接特征
            s3: [B, 256, H/16, W/16]
            s2: [B, 128, H/8,  W/8]
            s1: [B, 64,  H/4,  W/4]
    """

    def __init__(self, backbone_name="resnet34", pretrained=True):
        super().__init__()
        backbone = self._build_resnet(backbone_name, pretrained)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        if backbone_name in ("resnet18", "resnet34"):
            self.skip_channels = [256, 128, 64]
            self.out_channels = 512
        elif backbone_name == "resnet50":
            self.skip_channels = [1024, 512, 256]
            self.out_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

    @staticmethod
    def _build_resnet(name, pretrained):
        if name == "resnet18":
            w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            return models.resnet18(weights=w)
        elif name == "resnet34":
            w = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            return models.resnet34(weights=w)
        elif name == "resnet50":
            w = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            return models.resnet50(weights=w)
        else:
            raise ValueError(f"Unsupported backbone: {name}")

    def forward(self, x):
        x = self.stem(x)
        x = self.maxpool(x)
        s1 = self.layer1(x)   # H/4
        s2 = self.layer2(s1)  # H/8
        s3 = self.layer3(s2)  # H/16
        s4 = self.layer4(s3)  # H/32
        return s4, [s3, s2, s1]


class UNetHeatmapDecoder(nn.Module):
    """U-Net 风格解码器，带跳跃连接，生成热图。

    架构:
        16×16 (deep) → upsample → cat skip3(32×32) → conv → 32×32
                      → upsample → cat skip2(64×64) → conv → 64×64
                      → conv 1×1 → heatmap
    """

    def __init__(self, out_channels=512, skip_channels=None):
        super().__init__()
        if skip_channels is None:
            skip_channels = [256, 128, 64]

        # 16→32: concat skip3 (256ch)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(out_channels, skip_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(skip_channels[0]),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(skip_channels[0] * 2, skip_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(skip_channels[0]),
            nn.ReLU(inplace=True),
        )

        # 32→64: concat skip2 (128ch)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(skip_channels[0], skip_channels[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(skip_channels[1]),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(skip_channels[1] * 2, skip_channels[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(skip_channels[1]),
            nn.ReLU(inplace=True),
        )

        # Final 1×1 conv to heatmap
        self.final = nn.Sequential(
            nn.Conv2d(skip_channels[1], 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, skips):
        s3, s2 = skips[:2]  # use skip3 (32×32) and skip2 (64×64)

        # 16×16 → 32×32
        x = self.up1(x)
        x = torch.cat([x, s3], dim=1)
        x = self.conv1(x)

        # 32×32 → 64×64
        x = self.up2(x)
        x = torch.cat([x, s2], dim=1)
        x = self.conv2(x)

        heatmap = self.final(x)  # [B, 1, 64, 64]
        return heatmap


class UNetHeatmapLocModel(nn.Module):
    """U-Net 热图定位模型：ResNetEncoder → SE → UNetDecoder → Soft-Argmax

    与 HeatmapLocModel 的区别：
      - 使用跳跃连接（skip connections）保留编码器各层空间细节
      - 解码器利用多尺度特征生成更高质量热图
    """

    def __init__(self, backbone_name="resnet34", pretrained=True):
        super().__init__()
        self.encoder = ResNetEncoder(backbone_name, pretrained)

        self.use_se = USE_SE
        if self.use_se:
            self.se = SEBlock(self.encoder.out_channels)

        self.decoder = UNetHeatmapDecoder(
            out_channels=self.encoder.out_channels,
            skip_channels=self.encoder.skip_channels,
        )

    def forward(self, x):
        features, skips = self.encoder(x)          # [B, C, 16, 16], skip list
        if self.use_se:
            features = self.se(features)
        heatmap = self.decoder(features, skips)     # [B, 1, 64, 64]
        coords = soft_argmax(heatmap)               # [B, 2]
        return coords, heatmap

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
