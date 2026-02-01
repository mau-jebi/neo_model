"""
RT-DETR Loss Criterion

Implements the training loss for RT-DETR consisting of:
1. Varifocal Loss (VFL) for classification
2. L1 Loss for bounding box regression
3. GIoU Loss for bounding box quality

The loss is computed using the Hungarian matching between predictions and ground truth.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from .matcher import HungarianMatcher, box_cxcywh_to_xyxy, box_iou


class VarifocalLoss(nn.Module):
    """
    Varifocal Loss for object detection.
    Focuses more on positive samples and high-quality samples.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Weighting factor in [0, 1] to balance positive/negative examples
            gamma: Focusing parameter for modulating loss
            reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        target_score: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted class probabilities [N, num_classes]
            target: Target class labels [N] (class indices)
            target_score: Target quality scores [N] (e.g., IoU with GT)

        Returns:
            loss: Varifocal loss
        """
        pred_sigmoid = pred.sigmoid()
        target_one_hot = F.one_hot(target, num_classes=pred.shape[-1]).float()

        if target_score is not None:
            # Use target scores (e.g., IoU) as quality measure
            target_one_hot = target_one_hot * target_score.unsqueeze(-1)

        # Compute focal weight
        focal_weight = target_one_hot * (target_one_hot - pred_sigmoid).abs().pow(self.gamma) + \
                       (1 - target_one_hot) * pred_sigmoid.abs().pow(self.gamma)

        # Compute binary cross entropy
        bce = F.binary_cross_entropy_with_logits(pred, target_one_hot, reduction='none')

        # Apply focal weight
        loss = focal_weight * bce

        # Apply alpha weighting
        alpha_t = target_one_hot * self.alpha + (1 - target_one_hot) * (1 - self.alpha)
        loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class GIoULoss(nn.Module):
    """Generalized IoU Loss"""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_boxes: Predicted boxes [N, 4] in xyxy format
            target_boxes: Target boxes [N, 4] in xyxy format

        Returns:
            loss: GIoU loss
        """
        # Compute IoU and GIoU
        iou, giou = box_iou(pred_boxes, target_boxes)

        # Extract diagonal (matching pairs)
        giou = giou.diagonal()

        # GIoU loss
        loss = 1 - giou

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class RTDETRCriterion(nn.Module):
    """
    Complete loss criterion for RT-DETR training.
    Combines classification and bounding box losses.
    """

    def __init__(
        self,
        num_classes: int = 80,
        matcher: HungarianMatcher = None,
        weight_dict: Dict[str, float] = None,
        losses: List[str] = None,
        alpha: float = 0.75,
        gamma: float = 2.0
    ):
        """
        Args:
            num_classes: Number of object classes
            matcher: Hungarian matcher for bipartite matching
            weight_dict: Dictionary of loss weights {"loss_vfl": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
            losses: List of losses to compute ["vfl", "boxes"]
            alpha: Focal loss alpha
            gamma: Focal loss gamma
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher if matcher is not None else HungarianMatcher()

        if weight_dict is None:
            weight_dict = {"loss_vfl": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
        self.weight_dict = weight_dict

        if losses is None:
            losses = ["vfl", "boxes"]
        self.losses = losses

        # Loss modules
        self.vfl = VarifocalLoss(alpha=alpha, gamma=gamma, reduction='sum')
        self.giou_loss = GIoULoss(reduction='sum')

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses.

        Args:
            outputs: Dictionary with:
                - "pred_logits": [B, num_queries, num_classes]
                - "pred_boxes": [B, num_queries, 4] in cxcywh normalized format
                - "aux_outputs": Optional list of intermediate predictions
            targets: List of dictionaries (one per image):
                - "labels": [N_gt] class labels
                - "boxes": [N_gt, 4] boxes in cxcywh normalized format

        Returns:
            losses: Dictionary of losses {"loss_vfl": ..., "loss_bbox": ..., "loss_giou": ...}
        """
        # Exclude auxiliary outputs from matching
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Perform Hungarian matching
        indices = self.matcher(outputs_without_aux, targets)

        # Compute number of target boxes for normalization
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute losses
        losses = {}
        if "vfl" in self.losses:
            losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        if "boxes" in self.losses:
            losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))

        # Compute auxiliary losses (for intermediate decoder layers)
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices_aux = self.matcher(aux_outputs, targets)
                l_dict = {}
                if "vfl" in self.losses:
                    l_dict.update(self.loss_labels(aux_outputs, targets, indices_aux, num_boxes))
                if "boxes" in self.losses:
                    l_dict.update(self.loss_boxes(aux_outputs, targets, indices_aux, num_boxes))

                # Add to losses with suffix
                for k, v in l_dict.items():
                    losses[f"{k}_{i}"] = v

        return losses

    def loss_labels(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[tuple],
        num_boxes: float
    ) -> Dict[str, torch.Tensor]:
        """Compute classification loss (Varifocal Loss)"""
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]  # [B, num_queries, num_classes]

        # Get matched predictions and targets
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        # Create target tensor (all queries are background by default)
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device
        )  # [B, num_queries]

        # Set matched queries to their target class
        target_classes[idx] = target_classes_o

        # Compute target scores (IoU with ground truth)
        target_scores = torch.zeros_like(src_logits)
        if len(target_classes_o) > 0:
            # Compute IoU between matched predictions and targets
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat([t["boxes"][J] for t, (_, J) in zip(targets, indices)], dim=0)

            # Convert to xyxy for IoU computation
            src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
            target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)

            iou, _ = box_iou(src_boxes_xyxy, target_boxes_xyxy)
            iou = iou.diagonal()

            # Set scores for matched queries
            target_scores[idx, target_classes_o] = iou.detach()

        # Filter out background class
        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]  # Remove background class

        # Compute Varifocal Loss
        loss_vfl = self.vfl(
            src_logits.flatten(0, 1),
            target_classes.flatten(0, 1),
            target_scores.flatten(0, 1).sum(-1)
        ) / num_boxes

        return {"loss_vfl": loss_vfl}

    def loss_boxes(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[tuple],
        num_boxes: float
    ) -> Dict[str, torch.Tensor]:
        """Compute bounding box losses (L1 + GIoU)"""
        assert "pred_boxes" in outputs

        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]  # [num_matched, 4]
        target_boxes = torch.cat([t["boxes"][J] for t, (_, J) in zip(targets, indices)], dim=0)

        # L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='sum') / num_boxes

        # GIoU loss
        src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
        target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
        loss_giou = self.giou_loss(src_boxes_xyxy, target_boxes_xyxy) / num_boxes

        losses = {
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou
        }

        return losses

    @staticmethod
    def _get_src_permutation_idx(indices):
        """Get source permutation indices for gathering matched predictions"""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


def build_criterion(
    num_classes: int = 80,
    matcher: HungarianMatcher = None,
    weight_dict: Dict[str, float] = None,
    **kwargs
) -> RTDETRCriterion:
    """Build loss criterion"""
    return RTDETRCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        **kwargs
    )


def test_criterion():
    """Test loss criterion"""
    print("Testing RT-DETR Loss Criterion")
    print("=" * 60)

    batch_size = 2
    num_queries = 300
    num_classes = 80

    # Create dummy outputs
    outputs = {
        "pred_logits": torch.randn(batch_size, num_queries, num_classes),
        "pred_boxes": torch.rand(batch_size, num_queries, 4)
    }

    # Create dummy targets
    targets = [
        {
            "labels": torch.randint(0, num_classes, (5,)),
            "boxes": torch.rand(5, 4)
        },
        {
            "labels": torch.randint(0, num_classes, (3,)),
            "boxes": torch.rand(3, 4)
        }
    ]

    print(f"Batch size: {batch_size}")
    print(f"Number of queries: {num_queries}")
    print(f"Ground truth objects: {[len(t['labels']) for t in targets]}")

    # Build criterion
    criterion = build_criterion(num_classes=num_classes)

    # Compute losses
    losses = criterion(outputs, targets)

    print("\nLosses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    # Verify all expected losses are present
    expected_losses = ["loss_vfl", "loss_bbox", "loss_giou"]
    for loss_name in expected_losses:
        assert loss_name in losses, f"Missing loss: {loss_name}"
        assert not torch.isnan(losses[loss_name]), f"NaN in {loss_name}"
        assert not torch.isinf(losses[loss_name]), f"Inf in {loss_name}"

    print("\n" + "=" * 60)
    print("✅ Criterion test passed!")
    print("All losses computed successfully without NaN or Inf.")


if __name__ == "__main__":
    test_criterion()
