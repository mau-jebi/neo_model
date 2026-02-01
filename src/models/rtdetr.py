"""
RT-DETR Main Model for 1920x1080 Resolution

Integrates all components:
1. ResNet-50 Backbone - Multi-scale feature extraction
2. Hybrid Encoder - AIFI + CCFM for feature enhancement
3. Transformer Decoder - Query-based object detection
4. Prediction Heads - Classification and bounding box regression

This is the complete end-to-end model for training and inference.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import torch.nn.functional as F

from .backbone import build_backbone
from .encoder import build_encoder
from .decoder import build_decoder
from .criterion import build_criterion
from .matcher import HungarianMatcher, box_cxcywh_to_xyxy


class RTDETR(nn.Module):
    """
    RT-DETR: Real-Time Detection Transformer
    Adapted for 1920x1080 (16:9) resolution for Jebi AI Engine
    """

    def __init__(
        self,
        num_classes: int = 80,
        # Backbone config
        backbone_type: str = "resnet50",
        backbone_pretrained: bool = True,
        frozen_stages: int = -1,
        # Encoder config
        encoder_in_channels: List[int] = [512, 1024, 2048],
        encoder_hidden_dim: int = 256,
        num_encoder_layers: int = 1,
        # Decoder config
        num_queries: int = 300,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        # Training config
        aux_loss: bool = True
    ):
        """
        Args:
            num_classes: Number of object classes (80 for COCO)
            backbone_type: Backbone architecture
            backbone_pretrained: Load pretrained backbone weights
            frozen_stages: Number of backbone stages to freeze
            encoder_in_channels: Input channels for encoder (from backbone)
            encoder_hidden_dim: Hidden dimension for encoder
            num_encoder_layers: Number of encoder layers
            num_queries: Number of object queries
            num_decoder_layers: Number of decoder layers
            num_heads: Number of attention heads
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            aux_loss: Whether to use auxiliary losses from intermediate layers
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.aux_loss = aux_loss

        # Build backbone
        self.backbone = build_backbone(
            backbone_type=backbone_type,
            pretrained=backbone_pretrained,
            frozen_stages=frozen_stages,
            use_fpn=False
        )

        # Build encoder
        self.encoder = build_encoder(
            in_channels=encoder_in_channels,
            hidden_dim=encoder_hidden_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            feedforward_dim=dim_feedforward,
            dropout=dropout
        )

        # Build decoder
        self.decoder = build_decoder(
            hidden_dim=encoder_hidden_dim,
            num_queries=num_queries,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_classes=num_classes,
            return_intermediate=aux_loss
        )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training or inference.

        Args:
            images: Input images [B, 3, 1080, 1920]

        Returns:
            outputs: Dictionary with:
                - "pred_logits": [B, num_queries, num_classes]
                - "pred_boxes": [B, num_queries, 4]
                - "aux_outputs": List of intermediate predictions (if aux_loss=True)
        """
        # Extract backbone features
        features = self.backbone(images)
        # features is a dict: {"C3": [B, 512, 135, 240],
        #                       "C4": [B, 1024, 67, 120],
        #                       "C5": [B, 2048, 33, 60]}

        # Convert dict to list for encoder
        feature_list = [features["C3"], features["C4"], features["C5"]]

        # Encode features
        encoded_features = self.encoder(feature_list)
        # encoded_features: List of [B, 256, H, W] for each scale

        # Use the last (smallest) scale for decoder
        # For 1920x1080: [B, 256, 33, 60]
        memory = encoded_features[-1]

        # Flatten spatial dimensions for decoder
        B, C, H, W = memory.shape
        memory = memory.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # Decode to get predictions
        all_class_logits, all_bbox_preds = self.decoder(memory)

        # Prepare output
        outputs = {
            "pred_logits": all_class_logits[-1],  # [B, num_queries, num_classes]
            "pred_boxes": all_bbox_preds[-1]      # [B, num_queries, 4]
        }

        # Add auxiliary outputs for intermediate layers
        if self.aux_loss and len(all_class_logits) > 1:
            outputs["aux_outputs"] = [
                {"pred_logits": logits, "pred_boxes": boxes}
                for logits, boxes in zip(all_class_logits[:-1], all_bbox_preds[:-1])
            ]

        return outputs

    @torch.no_grad()
    def inference(
        self,
        images: torch.Tensor,
        conf_threshold: float = 0.3,
        nms_threshold: float = 0.7,
        max_detections: int = 300
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Inference mode with post-processing.

        Args:
            images: Input images [B, 3, 1080, 1920]
            conf_threshold: Confidence threshold for filtering
            nms_threshold: NMS IoU threshold
            max_detections: Maximum number of detections per image

        Returns:
            results: List of dictionaries (one per image):
                - "labels": [N] class labels
                - "boxes": [N, 4] bounding boxes in xyxy format (pixel coords)
                - "scores": [N] confidence scores
        """
        self.eval()

        # Forward pass
        outputs = self.forward(images)
        pred_logits = outputs["pred_logits"]  # [B, num_queries, num_classes]
        pred_boxes = outputs["pred_boxes"]    # [B, num_queries, 4] in cxcywh normalized

        # Get image dimensions
        H, W = images.shape[-2:]

        # Process each image
        results = []
        for logits, boxes in zip(pred_logits, pred_boxes):
            # Get class probabilities and scores
            scores = logits.sigmoid()  # [num_queries, num_classes]
            max_scores, labels = scores.max(dim=-1)  # [num_queries]

            # Filter by confidence
            keep = max_scores > conf_threshold
            scores = max_scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]

            # Convert boxes to pixel coordinates (xyxy format)
            boxes = box_cxcywh_to_xyxy(boxes)
            boxes[:, [0, 2]] *= W  # Scale width
            boxes[:, [1, 3]] *= H  # Scale height

            # Apply NMS per class
            keep_nms = []
            for class_id in labels.unique():
                class_mask = labels == class_id
                class_boxes = boxes[class_mask]
                class_scores = scores[class_mask]
                class_indices = torch.where(class_mask)[0]

                # NMS
                keep_idx = self._nms(class_boxes, class_scores, nms_threshold)
                keep_nms.append(class_indices[keep_idx])

            if len(keep_nms) > 0:
                keep_nms = torch.cat(keep_nms)

                # Limit to max detections
                if len(keep_nms) > max_detections:
                    top_scores, top_idx = scores[keep_nms].topk(max_detections)
                    keep_nms = keep_nms[top_idx]

                results.append({
                    "labels": labels[keep_nms],
                    "boxes": boxes[keep_nms],
                    "scores": scores[keep_nms]
                })
            else:
                # No detections
                results.append({
                    "labels": torch.empty(0, dtype=torch.long, device=images.device),
                    "boxes": torch.empty(0, 4, device=images.device),
                    "scores": torch.empty(0, device=images.device)
                })

        return results

    @staticmethod
    def _nms(boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Non-Maximum Suppression

        Args:
            boxes: [N, 4] in xyxy format
            scores: [N] confidence scores
            threshold: IoU threshold

        Returns:
            keep: Indices of kept boxes
        """
        if boxes.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=boxes.device)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        _, order = scores.sort(descending=True)

        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order[0])
                break

            i = order[0]
            keep.append(i)

            # Compute IoU with remaining boxes
            xx1 = torch.maximum(x1[i], x1[order[1:]])
            yy1 = torch.maximum(y1[i], y1[order[1:]])
            xx2 = torch.minimum(x2[i], x2[order[1:]])
            yy2 = torch.minimum(y2[i], y2[order[1:]])

            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # Keep boxes with IoU below threshold
            idx = (iou <= threshold).nonzero(as_tuple=False).squeeze()
            if idx.numel() == 0:
                break
            order = order[idx + 1]

        return torch.tensor(keep, dtype=torch.long, device=boxes.device)

    def load_pretrained_weights(self, checkpoint_path: str, strict: bool = False):
        """
        Load pretrained weights from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            strict: Whether to strictly enforce weight matching
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Load weights
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)

        if not strict:
            print(f"Loaded pretrained weights from {checkpoint_path}")
            if missing_keys:
                print(f"  Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"  Unexpected keys: {len(unexpected_keys)}")


def build_rtdetr(
    num_classes: int = 80,
    pretrained: bool = False,
    checkpoint_path: str = None,
    **kwargs
) -> RTDETR:
    """
    Build RT-DETR model.

    Args:
        num_classes: Number of object classes
        pretrained: Load pretrained weights
        checkpoint_path: Path to pretrained checkpoint
        **kwargs: Additional model arguments

    Returns:
        model: RT-DETR model
    """
    model = RTDETR(num_classes=num_classes, **kwargs)

    if pretrained and checkpoint_path:
        model.load_pretrained_weights(checkpoint_path, strict=False)

    return model


def test_rtdetr():
    """Test complete RT-DETR model with 1920x1080 input"""
    print("Testing Complete RT-DETR Model for 1920x1080 Resolution")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2

    # Create model
    model = build_rtdetr(
        num_classes=80,
        backbone_pretrained=False,  # Don't download weights for testing
        num_queries=300,
        num_decoder_layers=6,
        aux_loss=True
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Create dummy input
    images = torch.randn(batch_size, 3, 1080, 1920, device=device)
    print(f"\nInput shape: {images.shape}")

    # Training mode forward pass
    print("\n1. Training Mode Forward Pass:")
    model.train()
    outputs = model(images)

    print(f"  ✓ pred_logits: {outputs['pred_logits'].shape}")
    print(f"  ✓ pred_boxes: {outputs['pred_boxes'].shape}")
    if "aux_outputs" in outputs:
        print(f"  ✓ aux_outputs: {len(outputs['aux_outputs'])} intermediate layers")

    # Verify shapes
    assert outputs["pred_logits"].shape == (batch_size, 300, 80)
    assert outputs["pred_boxes"].shape == (batch_size, 300, 4)

    # Inference mode
    print("\n2. Inference Mode with Post-processing:")
    results = model.inference(images, conf_threshold=0.5)

    print(f"  ✓ Batch size: {len(results)}")
    for i, result in enumerate(results):
        print(f"  Image {i}: {len(result['labels'])} detections")
        if len(result['labels']) > 0:
            print(f"    - Labels: {result['labels'][:5].tolist()}...")
            print(f"    - Scores: {result['scores'][:5].tolist()}...")

    print("\n" + "=" * 60)
    print("✅ RT-DETR model test passed!")
    print("The complete model successfully processes 1920x1080 images.")
    print("\nKey Features:")
    print("  ✓ Handles 16:9 aspect ratio correctly")
    print("  ✓ Multi-scale feature extraction (strides 8, 16, 32)")
    print("  ✓ Transformer-based detection with 300 queries")
    print("  ✓ Auxiliary losses from intermediate layers")
    print("  ✓ Post-processing with NMS for inference")


if __name__ == "__main__":
    test_rtdetr()
