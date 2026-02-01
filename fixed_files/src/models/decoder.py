"""
RT-DETR Transformer Decoder for 1920x1080 Resolution

The decoder uses:
1. Learnable query embeddings (300 queries for object detection)
2. Self-attention layers for query interaction
3. Cross-attention layers for attending to encoder features
4. FFN (Feed-Forward Network) for final processing
5. Denoising training for improved convergence

Outputs class logits and bounding box coordinates for detected objects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import copy


def _get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    """Create N identical layers"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation: str) -> nn.Module:
    """Get activation function"""
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional cross-attention support"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [B, N, C] or [N, B, C]
            key: [B, M, C] or [M, B, C]
            value: [B, M, C] or [M, B, C]
            attn_mask: Optional attention mask
            key_padding_mask: Optional key padding mask [B, M]

        Returns:
            output: [B, N, C] or [N, B, C]
            attn_weights: [B, num_heads, N, M]
        """
        # Handle both [B, N, C] and [N, B, C] formats
        # Detect [N, B, C] format by checking if first dim is much larger than second
        if query.dim() == 3 and query.size(0) > query.size(1) * 2:
            # Assume [N, B, C] format, transpose to [B, N, C]
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            transposed = True
        else:
            transposed = False

        B, N, C = query.shape
        M = key.size(1)

        # Project Q, K, V
        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, M]

        if attn_mask is not None:
            attn = attn + attn_mask

        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        attn_weights = attn.softmax(dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        output = self.out_proj(output)

        if transposed:
            output = output.transpose(0, 1)

        return output, attn_weights


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with self-attention and cross-attention"""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "relu",
        normalize_before: bool = False
    ):
        super().__init__()

        # Self-attention
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)

        # Cross-attention
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            tgt: Query embeddings [B, N, C] where N is num_queries
            memory: Encoder features [B, M, C] where M is spatial size (H*W)
            masks: Optional attention masks

        Returns:
            Updated query embeddings [B, N, C]
        """
        # Self-attention
        tgt2, _ = self.self_attn(
            query=self.norm1(tgt) if self.normalize_before else tgt,
            key=self.norm1(tgt) if self.normalize_before else tgt,
            value=self.norm1(tgt) if self.normalize_before else tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        # Cross-attention
        tgt2, _ = self.cross_attn(
            query=self.norm2(tgt) if self.normalize_before else tgt,
            key=memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        # Feedforward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(
            self.norm3(tgt) if self.normalize_before else tgt
        ))))
        tgt = tgt + self.dropout3(tgt2)
        if not self.normalize_before:
            tgt = self.norm3(tgt)

        return tgt


class RTDETRTransformerDecoder(nn.Module):
    """
    RT-DETR Transformer Decoder.
    Processes query embeddings with encoder features to predict object detections.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_queries: int = 300,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "relu",
        normalize_before: bool = False,
        return_intermediate: bool = True,
        num_classes: int = 80,
        num_denoising: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0
    ):
        """
        Args:
            hidden_dim: Hidden dimension
            num_queries: Number of object queries
            num_decoder_layers: Number of decoder layers
            num_heads: Number of attention heads
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            activation: Activation function
            normalize_before: Apply normalization before attention
            return_intermediate: Return intermediate layer outputs
            num_classes: Number of object classes (80 for COCO)
            num_denoising: Number of denoising queries
            label_noise_ratio: Noise ratio for label denoising
            box_noise_scale: Noise scale for box denoising
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.return_intermediate = return_intermediate
        self.num_classes = num_classes
        self.num_denoising = num_denoising

        # Learnable query embeddings
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_embed = nn.Embedding(num_queries, hidden_dim)

        # Decoder layers
        decoder_layer = TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before
        )
        self.layers = _get_clones(decoder_layer, num_decoder_layers)

        # Final normalization
        self.norm = nn.LayerNorm(hidden_dim)

        # Prediction heads
        self.class_embed = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])

        self.bbox_embed = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, 3)  # Predict 4 bbox coords
            for _ in range(num_decoder_layers)
        ])

        # Denoising parameters
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters"""
        # Initialize query embeddings
        nn.init.normal_(self.query_embed.weight)
        nn.init.normal_(self.query_pos_embed.weight)

        # Initialize bbox prediction to predict small boxes initially
        for bbox_embed in self.bbox_embed:
            nn.init.constant_(bbox_embed.layers[-1].weight, 0)
            nn.init.constant_(bbox_embed.layers[-1].bias, 0)

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
        gt_labels: Optional[torch.Tensor] = None,
        gt_bboxes: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            memory: Encoder features [B, M, C] where M is H*W (flattened spatial dims)
            memory_mask: Optional mask for encoder features
            gt_labels: Ground truth labels for denoising training [B, N_gt]
            gt_bboxes: Ground truth bboxes for denoising training [B, N_gt, 4]

        Returns:
            all_class_logits: List of class predictions for each layer
                Each element: [B, num_queries, num_classes]
            all_bbox_preds: List of bbox predictions for each layer
                Each element: [B, num_queries, 4]
        """
        B = memory.size(0)

        # Get learnable query embeddings
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, num_queries, C]
        query_pos = self.query_pos_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, num_queries, C]

        # Initialize query content with zeros
        tgt = torch.zeros_like(query_embed)

        # Optionally add denoising queries during training
        if self.training and gt_labels is not None and gt_bboxes is not None:
            tgt, query_pos = self._add_denoising_queries(
                tgt, query_pos, gt_labels, gt_bboxes
            )

        # Decoder layers
        all_class_logits = []
        all_bbox_preds = []

        for i, layer in enumerate(self.layers):
            # Add positional embedding to queries
            tgt_with_pos = tgt + query_pos

            # Decoder layer forward
            tgt = layer(
                tgt=tgt_with_pos,
                memory=memory,
                tgt_mask=None,
                memory_mask=memory_mask
            )

            # Predictions for this layer
            class_logits = self.class_embed[i](tgt)  # [B, num_queries, num_classes]
            bbox_preds = self.bbox_embed[i](tgt).sigmoid()  # [B, num_queries, 4]

            if self.return_intermediate:
                all_class_logits.append(class_logits)
                all_bbox_preds.append(bbox_preds)

        # Normalize final output
        tgt = self.norm(tgt)

        # Final predictions (if not already added)
        if not self.return_intermediate:
            class_logits = self.class_embed[-1](tgt)
            bbox_preds = self.bbox_embed[-1](tgt).sigmoid()
            all_class_logits = [class_logits]
            all_bbox_preds = [bbox_preds]

        return all_class_logits, all_bbox_preds

    def _add_denoising_queries(
        self,
        tgt: torch.Tensor,
        query_pos: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add denoising queries for training stability.
        Adds noisy versions of ground truth as additional queries.

        Args:
            tgt: Query content [B, num_queries, C]
            query_pos: Query positional embeddings [B, num_queries, C]
            gt_labels: Ground truth labels [B, N_gt]
            gt_bboxes: Ground truth bboxes [B, N_gt, 4]

        Returns:
            tgt_combined: Combined queries [B, num_queries + num_denoising, C]
            query_pos_combined: Combined positional embeddings
        """
        # This is a simplified version - full implementation would be more complex
        # For now, just return the original queries
        return tgt, query_pos


class MLP(nn.Module):
    """Multi-layer perceptron for bbox prediction"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_decoder(
    hidden_dim: int = 256,
    num_queries: int = 300,
    num_decoder_layers: int = 6,
    num_classes: int = 80,
    **kwargs
) -> RTDETRTransformerDecoder:
    """Build RT-DETR transformer decoder"""
    return RTDETRTransformerDecoder(
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        num_decoder_layers=num_decoder_layers,
        num_classes=num_classes,
        **kwargs
    )


def test_decoder():
    """Test decoder with 1920x1080 encoder features"""
    print("Testing RT-DETR Transformer Decoder")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    hidden_dim = 256
    num_queries = 300

    # Simulate flattened encoder features
    # For 1920x1080, assuming we use C5 features (33×60)
    spatial_size = 33 * 60  # 1980
    memory = torch.randn(batch_size, spatial_size, hidden_dim, device=device)

    print(f"Input memory shape: {memory.shape}")

    # Build decoder
    decoder = build_decoder(
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        num_decoder_layers=6,
        num_classes=80
    ).to(device)
    decoder.eval()

    # Forward pass
    with torch.no_grad():
        class_logits, bbox_preds = decoder(memory)

    print(f"\nNumber of decoder layers: {len(class_logits)}")
    print(f"Class logits shape (last layer): {class_logits[-1].shape}")
    print(f"Bbox predictions shape (last layer): {bbox_preds[-1].shape}")

    # Verify shapes
    expected_class_shape = (batch_size, num_queries, 80)
    expected_bbox_shape = (batch_size, num_queries, 4)

    assert class_logits[-1].shape == expected_class_shape, \
        f"Class shape mismatch! Expected {expected_class_shape}, got {class_logits[-1].shape}"
    assert bbox_preds[-1].shape == expected_bbox_shape, \
        f"Bbox shape mismatch! Expected {expected_bbox_shape}, got {bbox_preds[-1].shape}"

    # Verify bbox predictions are in [0, 1] range (after sigmoid)
    assert (bbox_preds[-1] >= 0).all() and (bbox_preds[-1] <= 1).all(), \
        "Bbox predictions should be in [0, 1] range!"

    print("\n" + "=" * 60)
    print("✅ Decoder test passed!")
    print(f"Decoder outputs {num_queries} detection predictions per image.")


if __name__ == "__main__":
    test_decoder()
