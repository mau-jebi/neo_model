"""
RT-DETR Hybrid Encoder for 1920x1080 Resolution

The hybrid encoder consists of:
1. AIFI (Attention-based Intrascale Feature Interaction) - Efficient self-attention within each scale
2. CCFM (Cross-scale Feature Fusion Module) - Fuses features across different scales

This encoder processes multi-scale features from the backbone and outputs enhanced features
for the transformer decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import math


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention module"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, N, C] where N is sequence length
            mask: Optional attention mask [B, N, N]

        Returns:
            Output tensor [B, N, C]
        """
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [B, num_heads, N, head_dim]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        x = self.proj(x)
        x = self.dropout(x)

        return x


class AIFILayer(nn.Module):
    """
    Attention-based Intrascale Feature Interaction (AIFI) Layer.
    Efficient self-attention for processing features within a single scale.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        dropout: float = 0.0,
        activation: str = "relu"
    ):
        super().__init__()

        # Self-attention
        self.self_attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Feedforward network
        self.linear1 = nn.Linear(embed_dim, feedforward_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, embed_dim)

        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation function
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Input features [B, C, H, W]

        Returns:
            Output features [B, C, H, W]
        """
        B, C, H, W = src.shape

        # Flatten spatial dimensions for attention
        src_flat = src.flatten(2).permute(0, 2, 1)  # [B, H*W, C]

        # Self-attention with residual connection
        src2 = self.self_attn(self.norm1(src_flat))
        src_flat = src_flat + self.dropout1(src2)

        # Feedforward with residual connection
        src2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(src_flat)))))
        src_flat = src_flat + self.dropout2(src2)

        # Reshape back to spatial format
        src = src_flat.permute(0, 2, 1).reshape(B, C, H, W)

        return src


class RepC3(nn.Module):
    """
    Reparameterized C3 module for feature processing.
    Consists of multiple bottleneck blocks with residual connections.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        expansion: float = 0.5
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0)
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0)
        self.conv3 = nn.Conv2d(2 * hidden_channels, out_channels, 1, 1, 0)

        self.blocks = nn.Sequential(*[
            Bottleneck(hidden_channels, hidden_channels, expansion=1.0)
            for _ in range(num_blocks)
        ])

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.blocks(self.conv2(x))
        x = torch.cat([x1, x2], dim=1)
        x = self.bn(self.conv3(x))
        x = self.act(x)
        return x


class Bottleneck(nn.Module):
    """Bottleneck block for RepC3"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        shortcut: bool = True
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut:
            out = out + identity
        out = self.act(out)
        return out


class CCFMLayer(nn.Module):
    """
    Cross-scale Feature Fusion Module (CCFM).
    Fuses features from different scales using cross-attention.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        num_blocks: int = 3,
        expansion: float = 0.5
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Reduce channels of each scale to out_channels
        self.reduce_layers = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1, 1, 0)
            for in_ch in in_channels
        ])

        # Feature fusion blocks
        self.fusion_blocks = nn.ModuleList([
            RepC3(out_channels, out_channels, num_blocks, expansion)
            for _ in range(len(in_channels))
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: List of feature maps at different scales
                [
                    [B, C1, H1, W1],  # e.g., [B, 512, 135, 240]
                    [B, C2, H2, W2],  # e.g., [B, 1024, 67, 120]
                    [B, C3, H3, W3]   # e.g., [B, 2048, 33, 60]
                ]

        Returns:
            fused_features: List of fused feature maps
                [
                    [B, out_channels, H1, W1],
                    [B, out_channels, H2, W2],
                    [B, out_channels, H3, W3]
                ]
        """
        # Reduce channels
        reduced_features = [
            reduce(feat) for reduce, feat in zip(self.reduce_layers, features)
        ]

        # Fuse features across scales
        fused_features = []
        for i, feat in enumerate(reduced_features):
            # Upsample/downsample other scales to match current scale
            aligned_features = [feat]  # Include current scale

            for j, other_feat in enumerate(reduced_features):
                if i != j:
                    # Resize to match target spatial size
                    aligned_feat = F.interpolate(
                        other_feat,
                        size=feat.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                    aligned_features.append(aligned_feat)

            # Sum aligned features
            fused = sum(aligned_features)

            # Apply fusion block
            fused = self.fusion_blocks[i](fused)
            fused_features.append(fused)

        return fused_features


class HybridEncoder(nn.Module):
    """
    RT-DETR Hybrid Encoder combining AIFI and CCFM.
    Processes multi-scale features from backbone with efficient attention.
    """

    def __init__(
        self,
        in_channels: List[int] = [512, 1024, 2048],
        hidden_dim: int = 256,
        num_encoder_layers: int = 1,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        dropout: float = 0.0,
        expansion: float = 1.0,
        use_encoder_idx: List[int] = [2]  # Apply AIFI only to last scale
    ):
        """
        Args:
            in_channels: Input channels for each scale [C3, C4, C5]
            hidden_dim: Hidden dimension for encoder
            num_encoder_layers: Number of AIFI layers
            num_heads: Number of attention heads
            feedforward_dim: Feedforward network dimension
            dropout: Dropout rate
            expansion: Expansion ratio for RepC3
            use_encoder_idx: Indices of scales to apply AIFI (typically just C5)
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx

        # Input projection layers
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, hidden_dim, 1, 1, 0),
                nn.BatchNorm2d(hidden_dim)
            )
            for in_ch in in_channels
        ])

        # AIFI layers (applied only to selected scales)
        self.aifi_layers = nn.ModuleList([
            AIFILayer(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                dropout=dropout
            )
            for _ in range(num_encoder_layers)
        ])

        # CCFM for cross-scale fusion
        self.ccfm = CCFMLayer(
            in_channels=[hidden_dim] * len(in_channels),
            out_channels=hidden_dim,
            expansion=expansion
        )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Process multi-scale features.

        Args:
            features: List of backbone features
                [
                    [B, 512, 135, 240],   # C3 (stride 8)
                    [B, 1024, 67, 120],   # C4 (stride 16)
                    [B, 2048, 33, 60]     # C5 (stride 32)
                ]

        Returns:
            encoded_features: List of encoded features
                [
                    [B, 256, 135, 240],   # P3
                    [B, 256, 67, 120],    # P4
                    [B, 256, 33, 60]      # P5
                ]
        """
        # Project input features to hidden_dim
        proj_features = [
            proj(feat) for proj, feat in zip(self.input_proj, features)
        ]

        # Apply AIFI to selected scales (typically just the last one - C5)
        aifi_features = []
        for i, feat in enumerate(proj_features):
            if i in self.use_encoder_idx:
                # Apply AIFI layers
                for aifi_layer in self.aifi_layers:
                    feat = aifi_layer(feat)
            aifi_features.append(feat)

        # Cross-scale feature fusion
        fused_features = self.ccfm(aifi_features)

        return fused_features


def build_encoder(
    in_channels: List[int] = [512, 1024, 2048],
    hidden_dim: int = 256,
    num_encoder_layers: int = 1,
    **kwargs
) -> HybridEncoder:
    """Build hybrid encoder"""
    return HybridEncoder(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_encoder_layers,
        **kwargs
    )


def test_encoder():
    """Test encoder with 1920x1080 feature maps"""
    print("Testing RT-DETR Hybrid Encoder for 1920x1080 Resolution")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2

    # Create dummy multi-scale features (from backbone)
    features = [
        torch.randn(batch_size, 512, 135, 240, device=device),   # C3
        torch.randn(batch_size, 1024, 67, 120, device=device),   # C4
        torch.randn(batch_size, 2048, 33, 60, device=device)     # C5
    ]

    print("Input features:")
    for i, feat in enumerate(features):
        print(f"  C{i+3}: {feat.shape}")

    # Build encoder
    encoder = build_encoder(
        in_channels=[512, 1024, 2048],
        hidden_dim=256,
        num_encoder_layers=1
    ).to(device)
    encoder.eval()

    # Forward pass
    with torch.no_grad():
        encoded_features = encoder(features)

    print("\nEncoded features:")
    expected_shapes = [
        (batch_size, 256, 135, 240),
        (batch_size, 256, 67, 120),
        (batch_size, 256, 33, 60)
    ]

    for i, (feat, expected) in enumerate(zip(encoded_features, expected_shapes)):
        assert feat.shape == expected, \
            f"Shape mismatch! Expected {expected}, got {feat.shape}"
        print(f"  ✓ P{i+3}: {feat.shape}")

    print("\n" + "=" * 60)
    print("✅ Encoder test passed!")
    print("Hybrid encoder correctly processes 1920x1080 features.")


if __name__ == "__main__":
    test_encoder()
