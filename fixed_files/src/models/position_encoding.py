"""
Position Encoding for RT-DETR at 1920x1080 Resolution

CRITICAL MODULE: This handles the 16:9 aspect ratio adaptation.
RT-DETR defaults assume square inputs (640x640), but we need rectangular (1920x1080)
which produces non-square feature maps: 240×135, 120×67, 60×33

This module implements separate sine/cosine positional embeddings for height and width
dimensions to correctly encode spatial information in rectangular feature maps.
"""

import torch
import torch.nn as nn
import math


class PositionEmbeddingSine(nn.Module):
    """
    Sine/Cosine positional encoding for 2D feature maps.
    Handles rectangular (non-square) feature maps by applying separate embeddings
    for height and width dimensions.

    This is adapted from DETR and modified for RT-DETR with rectangular inputs.
    """

    def __init__(
        self,
        num_pos_feats: int = 128,
        temperature: int = 10000,
        normalize: bool = True,
        scale: float = None,
        eps: float = 1e-6
    ):
        """
        Args:
            num_pos_feats: Number of positional features (total embedding dim = 2 * num_pos_feats)
            temperature: Temperature for the sinusoidal functions
            normalize: If True, normalize positions to [0, 1]
            scale: Scaling factor (if None, uses 2*pi)
            eps: Small value to avoid division by zero
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale if scale is not None else 2 * math.pi
        self.eps = eps

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Generate positional encodings for a feature map.

        Args:
            x: Feature map of shape [B, C, H, W]
            mask: Optional mask of shape [B, H, W] where True indicates invalid positions

        Returns:
            pos: Positional encoding of shape [B, num_pos_feats*2, H, W]

        Example feature map sizes for 1920x1080 input:
            - Stride 8:  H=135, W=240  (1080/8, 1920/8)
            - Stride 16: H=67,  W=120  (1080/16, 1920/16)
            - Stride 32: H=33,  W=60   (1080/32, 1920/32)
        """
        B, C, H, W = x.shape

        # Create or use provided mask
        if mask is None:
            mask = torch.zeros(B, H, W, dtype=torch.bool, device=x.device)

        # Invert mask: 1 for valid positions, 0 for invalid
        not_mask = ~mask

        # Compute cumulative sum along height and width
        # This creates position indices from 0 to H-1 and 0 to W-1
        y_embed = not_mask.cumsum(dim=1, dtype=torch.float32)  # [B, H, W]
        x_embed = not_mask.cumsum(dim=2, dtype=torch.float32)  # [B, H, W]

        if self.normalize:
            # Normalize positions to [0, 1] range
            # This is crucial for handling different feature map sizes
            y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale

        # Create dimension indices for the sinusoidal functions
        # dim_t = temperature^(2i/d) where i is the dimension index
        dim_t = torch.arange(
            self.num_pos_feats,
            dtype=torch.float32,
            device=x.device
        )
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Apply sinusoidal encoding
        # For even dimensions: sin(pos / temperature^(2i/d))
        # For odd dimensions: cos(pos / temperature^(2i/d))

        # Height (Y) positional encoding
        pos_y = y_embed[:, :, :, None] / dim_t  # [B, H, W, num_pos_feats]
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4
        ).flatten(3)  # [B, H, W, num_pos_feats]

        # Width (X) positional encoding
        pos_x = x_embed[:, :, :, None] / dim_t  # [B, H, W, num_pos_feats]
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4
        ).flatten(3)  # [B, H, W, num_pos_feats]

        # Concatenate height and width encodings
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # [B, num_pos_feats*2, H, W]

        return pos

    def extra_repr(self) -> str:
        """String representation for debugging"""
        return (
            f"num_pos_feats={self.num_pos_feats}, "
            f"temperature={self.temperature}, "
            f"normalize={self.normalize}, "
            f"scale={self.scale}"
        )


class PositionEmbeddingLearned(nn.Module):
    """
    Learned positional embeddings for 2D feature maps.
    Alternative to sinusoidal encoding - uses learned embeddings.
    """

    def __init__(
        self,
        num_pos_feats: int = 128,
        max_h: int = 135,  # Maximum height (for stride 8: 1080/8 = 135)
        max_w: int = 240   # Maximum width (for stride 8: 1920/8 = 240)
    ):
        """
        Args:
            num_pos_feats: Number of positional features per dimension
            max_h: Maximum height of feature maps
            max_w: Maximum width of feature maps
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.row_embed = nn.Embedding(max_h, num_pos_feats)
        self.col_embed = nn.Embedding(max_w, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize embeddings"""
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Generate learned positional encodings.

        Args:
            x: Feature map of shape [B, C, H, W]
            mask: Optional mask (not used but kept for API compatibility)

        Returns:
            pos: Positional encoding of shape [B, num_pos_feats*2, H, W]
        """
        B, C, H, W = x.shape

        # Create position indices
        i = torch.arange(W, device=x.device)  # [W]
        j = torch.arange(H, device=x.device)  # [H]

        # Get embeddings
        x_emb = self.col_embed(i)  # [W, num_pos_feats]
        y_emb = self.row_embed(j)  # [H, num_pos_feats]

        # Broadcast and concatenate
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(H, 1, 1),  # [H, W, num_pos_feats]
            y_emb.unsqueeze(1).repeat(1, W, 1),  # [H, W, num_pos_feats]
        ], dim=-1)  # [H, W, num_pos_feats*2]

        # Permute to [num_pos_feats*2, H, W] and add batch dimension
        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1)  # [B, num_pos_feats*2, H, W]

        return pos


def build_position_encoding(
    hidden_dim: int = 256,
    position_embedding: str = "sine",
    temperature: int = 10000
) -> nn.Module:
    """
    Build position encoding module.

    Args:
        hidden_dim: Hidden dimension of the model
        position_embedding: Type of position embedding ("sine" or "learned")
        temperature: Temperature for sinusoidal encoding

    Returns:
        Position encoding module
    """
    num_pos_feats = hidden_dim // 2

    if position_embedding == "sine":
        return PositionEmbeddingSine(
            num_pos_feats=num_pos_feats,
            temperature=temperature,
            normalize=True
        )
    elif position_embedding == "learned":
        return PositionEmbeddingLearned(
            num_pos_feats=num_pos_feats,
            max_h=135,  # For 1080/8 = 135
            max_w=240   # For 1920/8 = 240
        )
    else:
        raise ValueError(f"Unknown position embedding: {position_embedding}")


def test_position_encoding():
    """
    Test position encoding with 1920x1080 feature maps.
    Validates that encodings work correctly for all three stride levels.
    """
    print("Testing Position Encoding for 1920x1080 Resolution")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2

    # Test both sine and learned embeddings
    for emb_type in ["sine", "learned"]:
        print(f"\nTesting {emb_type.upper()} Position Encoding:")
        pos_enc = build_position_encoding(
            hidden_dim=256,
            position_embedding=emb_type
        ).to(device)

        # Test all three feature map sizes from RT-DETR
        test_cases = [
            (135, 240, 8),   # Stride 8:  1080/8  = 135, 1920/8  = 240
            (67, 120, 16),   # Stride 16: 1080/16 = 67,  1920/16 = 120
            (33, 60, 32),    # Stride 32: 1080/32 = 33,  1920/32 = 60
        ]

        for h, w, stride in test_cases:
            # Create dummy feature map
            x = torch.randn(batch_size, 256, h, w, device=device)

            # Generate position encoding
            pos = pos_enc(x)

            # Verify shape
            expected_shape = (batch_size, 256, h, w)
            assert pos.shape == expected_shape, \
                f"Shape mismatch! Expected {expected_shape}, got {pos.shape}"

            # Verify no NaN or Inf
            assert not torch.isnan(pos).any(), "Position encoding contains NaN!"
            assert not torch.isinf(pos).any(), "Position encoding contains Inf!"

            print(f"  ✓ Stride {stride:2d}: Feature map {h:3d}×{w:3d} → Encoding {pos.shape}")

    print("\n" + "=" * 60)
    print("✅ All position encoding tests passed!")
    print("The module correctly handles rectangular (16:9) feature maps.")


if __name__ == "__main__":
    test_position_encoding()
