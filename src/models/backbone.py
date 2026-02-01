"""
ResNet-50 Backbone for RT-DETR at 1920x1080

Extracts multi-scale features at strides [8, 16, 32] for object detection.
Uses torchvision's pretrained ResNet-50 with modifications for FPN-style outputs.

Feature map sizes for 1920x1080 input:
- C3 (stride 8):  512 channels, 135×240 spatial
- C4 (stride 16): 1024 channels, 67×120 spatial
- C5 (stride 32): 2048 channels, 33×60 spatial
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from typing import List, Dict


class ResNetBackbone(nn.Module):
    """
    ResNet-50 backbone for feature extraction.
    Returns multi-scale features at different strides for FPN-style architectures.
    """

    def __init__(
        self,
        pretrained: bool = True,
        frozen_stages: int = -1,
        output_stride: List[int] = [8, 16, 32],
        out_channels: List[int] = [512, 1024, 2048],
        return_idx: List[int] = [1, 2, 3]  # C3, C4, C5
    ):
        """
        Args:
            pretrained: Load ImageNet pretrained weights
            frozen_stages: Number of stages to freeze (-1 for no freezing, 0-4 to freeze)
            output_stride: Output strides for multi-scale features
            out_channels: Output channels for each scale
            return_idx: Which layers to return (1=C3, 2=C4, 3=C5)
        """
        super().__init__()
        self.frozen_stages = frozen_stages
        self.output_stride = output_stride
        self.out_channels = out_channels
        self.return_idx = return_idx

        # Load pretrained ResNet-50
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2  # Latest weights
            backbone = resnet50(weights=weights)
        else:
            backbone = resnet50(weights=None)

        # Extract ResNet layers
        # ResNet structure:
        # - conv1 + bn1 + relu + maxpool: stride 4
        # - layer1 (C2): stride 4,  256 channels
        # - layer2 (C3): stride 8,  512 channels
        # - layer3 (C4): stride 16, 1024 channels
        # - layer4 (C5): stride 32, 2048 channels

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1  # C2: stride 4
        self.layer2 = backbone.layer2  # C3: stride 8
        self.layer3 = backbone.layer3  # C4: stride 16
        self.layer4 = backbone.layer4  # C5: stride 32

        # Freeze stages if requested
        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze backbone stages for transfer learning"""
        if self.frozen_stages >= 0:
            # Freeze stem (conv1, bn1)
            self.conv1.eval()
            self.bn1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False

        # Freeze subsequent stages
        stages = [self.layer1, self.layer2, self.layer3, self.layer4]
        for i in range(self.frozen_stages):
            if i < len(stages):
                stage = stages[i]
                stage.eval()
                for param in stage.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        """Override train mode to respect frozen stages"""
        super().train(mode)
        self._freeze_stages()
        return self

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale features from input image.

        Args:
            x: Input tensor of shape [B, 3, 1080, 1920]

        Returns:
            features: Dictionary of features at different scales
                {
                    "C3": [B, 512, 135, 240],   # stride 8
                    "C4": [B, 1024, 67, 120],   # stride 16
                    "C5": [B, 2048, 33, 60]     # stride 32
                }
        """
        # Stem: stride 2
        x = self.conv1(x)      # [B, 64, 540, 960]
        x = self.bn1(x)
        x = self.relu(x)

        # Stem: stride 4 (after maxpool)
        x = self.maxpool(x)    # [B, 64, 270, 480]

        # Stage 1: stride 4
        c2 = self.layer1(x)    # [B, 256, 270, 480]

        # Stage 2: stride 8 (C3)
        c3 = self.layer2(c2)   # [B, 512, 135, 240]

        # Stage 3: stride 16 (C4)
        c4 = self.layer3(c3)   # [B, 1024, 67, 120]

        # Stage 4: stride 32 (C5)
        c5 = self.layer4(c4)   # [B, 2048, 33, 60]

        # Return requested features
        features = {}
        all_features = [c2, c3, c4, c5]
        feature_names = ["C2", "C3", "C4", "C5"]

        for idx in self.return_idx:
            features[feature_names[idx]] = all_features[idx]

        return features


class ResNetBackboneWithFPN(nn.Module):
    """
    ResNet-50 backbone with Feature Pyramid Network (FPN).
    Adds lateral connections and top-down pathway for multi-scale fusion.
    """

    def __init__(
        self,
        pretrained: bool = True,
        frozen_stages: int = -1,
        fpn_dim: int = 256
    ):
        """
        Args:
            pretrained: Load ImageNet pretrained weights
            frozen_stages: Number of stages to freeze
            fpn_dim: FPN feature dimension (typically 256)
        """
        super().__init__()

        # ResNet backbone
        self.backbone = ResNetBackbone(
            pretrained=pretrained,
            frozen_stages=frozen_stages,
            return_idx=[1, 2, 3]  # C3, C4, C5
        )

        # FPN lateral connections (1x1 conv to reduce channels to fpn_dim)
        self.lateral_c3 = nn.Conv2d(512, fpn_dim, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(1024, fpn_dim, kernel_size=1)
        self.lateral_c5 = nn.Conv2d(2048, fpn_dim, kernel_size=1)

        # FPN output convolutions (3x3 conv to smooth upsampled features)
        self.fpn_c3 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)
        self.fpn_c4 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)
        self.fpn_c5 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)

        self._init_fpn_weights()

    def _init_fpn_weights(self):
        """Initialize FPN conv layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract FPN features.

        Args:
            x: Input tensor of shape [B, 3, 1080, 1920]

        Returns:
            features: Dictionary of FPN features
                {
                    "P3": [B, 256, 135, 240],   # stride 8
                    "P4": [B, 256, 67, 120],    # stride 16
                    "P5": [B, 256, 33, 60]      # stride 32
                }
        """
        # Get backbone features
        backbone_features = self.backbone(x)
        c3 = backbone_features["C3"]  # [B, 512, 135, 240]
        c4 = backbone_features["C4"]  # [B, 1024, 67, 120]
        c5 = backbone_features["C5"]  # [B, 2048, 33, 60]

        # Lateral connections
        p5 = self.lateral_c5(c5)  # [B, 256, 33, 60]
        p4 = self.lateral_c4(c4)  # [B, 256, 67, 120]
        p3 = self.lateral_c3(c3)  # [B, 256, 135, 240]

        # Top-down pathway with lateral connections
        # P5 -> P4
        p5_upsampled = F.interpolate(
            p5, size=p4.shape[-2:], mode='nearest'
        )
        p4 = p4 + p5_upsampled

        # P4 -> P3
        p4_upsampled = F.interpolate(
            p4, size=p3.shape[-2:], mode='nearest'
        )
        p3 = p3 + p4_upsampled

        # Apply smoothing convolutions
        p3 = self.fpn_c3(p3)  # [B, 256, 135, 240]
        p4 = self.fpn_c4(p4)  # [B, 256, 67, 120]
        p5 = self.fpn_c5(p5)  # [B, 256, 33, 60]

        return {
            "P3": p3,
            "P4": p4,
            "P5": p5
        }


def build_backbone(
    backbone_type: str = "resnet50",
    pretrained: bool = True,
    frozen_stages: int = -1,
    use_fpn: bool = False
) -> nn.Module:
    """
    Build backbone network.

    Args:
        backbone_type: Type of backbone ("resnet50")
        pretrained: Load pretrained weights
        frozen_stages: Number of stages to freeze
        use_fpn: Whether to use FPN

    Returns:
        Backbone module
    """
    if backbone_type.lower() == "resnet50":
        if use_fpn:
            return ResNetBackboneWithFPN(
                pretrained=pretrained,
                frozen_stages=frozen_stages
            )
        else:
            return ResNetBackbone(
                pretrained=pretrained,
                frozen_stages=frozen_stages
            )
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")


def test_backbone():
    """
    Test backbone with 1920x1080 input.
    Validates feature map shapes for all stride levels.
    """
    print("Testing ResNet-50 Backbone for 1920x1080 Resolution")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2

    # Create dummy input
    x = torch.randn(batch_size, 3, 1080, 1920, device=device)
    print(f"Input shape: {x.shape}")

    # Test standard backbone
    print("\n1. Testing Standard ResNet-50 Backbone:")
    backbone = build_backbone(
        backbone_type="resnet50",
        pretrained=False,  # Don't download weights for testing
        use_fpn=False
    ).to(device)
    backbone.eval()

    with torch.no_grad():
        features = backbone(x)

    expected_shapes = {
        "C3": (batch_size, 512, 135, 240),
        "C4": (batch_size, 1024, 67, 120),
        "C5": (batch_size, 2048, 33, 60)
    }

    for name, feat in features.items():
        expected = expected_shapes[name]
        assert feat.shape == expected, \
            f"Shape mismatch for {name}! Expected {expected}, got {feat.shape}"
        print(f"  ✓ {name}: {feat.shape} (stride {[8, 16, 32][list(features.keys()).index(name)]})")

    # Test FPN backbone
    print("\n2. Testing ResNet-50 Backbone with FPN:")
    backbone_fpn = build_backbone(
        backbone_type="resnet50",
        pretrained=False,
        use_fpn=True
    ).to(device)
    backbone_fpn.eval()

    with torch.no_grad():
        fpn_features = backbone_fpn(x)

    expected_fpn_shapes = {
        "P3": (batch_size, 256, 135, 240),
        "P4": (batch_size, 256, 67, 120),
        "P5": (batch_size, 256, 33, 60)
    }

    for name, feat in fpn_features.items():
        expected = expected_fpn_shapes[name]
        assert feat.shape == expected, \
            f"Shape mismatch for {name}! Expected {expected}, got {feat.shape}"
        print(f"  ✓ {name}: {feat.shape} (stride {[8, 16, 32][list(fpn_features.keys()).index(name)]})")

    print("\n" + "=" * 60)
    print("✅ All backbone tests passed!")
    print("Feature extraction works correctly for 1920x1080 resolution.")


if __name__ == "__main__":
    test_backbone()
