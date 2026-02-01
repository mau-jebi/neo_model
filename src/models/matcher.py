"""
Hungarian Matcher for RT-DETR

Implements bipartite matching between model predictions and ground truth objects
using the Hungarian algorithm (scipy.optimize.linear_sum_assignment).

The matching cost combines:
1. Classification cost (focal loss)
2. Bounding box L1 distance cost
3. GIoU cost

This ensures each prediction is matched to at most one ground truth object.
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple
import torch.nn.functional as F


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from (center_x, center_y, width, height) to (x1, y1, x2, y2) format.

    Args:
        boxes: Tensor of shape [..., 4] in cxcywh format

    Returns:
        boxes: Tensor of shape [..., 4] in xyxy format
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from (x1, y1, x2, y2) to (center_x, center_y, width, height) format.

    Args:
        boxes: Tensor of shape [..., 4] in xyxy format

    Returns:
        boxes: Tensor of shape [..., 4] in cxcywh format
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute IoU and GIoU between two sets of boxes.

    Args:
        boxes1: Tensor of shape [N, 4] in xyxy format
        boxes2: Tensor of shape [M, 4] in xyxy format

    Returns:
        iou: Tensor of shape [N, M] with IoU values
        giou: Tensor of shape [N, M] with GIoU values
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    # Union
    union = area1[:, None] + area2 - inter

    # IoU
    iou = inter / (union + 1e-6)

    # GIoU - compute enclosing box
    lt_enclosing = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb_enclosing = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_enclosing = (rb_enclosing - lt_enclosing).clamp(min=0)
    area_enclosing = wh_enclosing[:, :, 0] * wh_enclosing[:, :, 1]

    giou = iou - (area_enclosing - union) / (area_enclosing + 1e-6)

    return iou, giou


class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher for bipartite matching between predictions and ground truth.

    This module computes an assignment between the predictions and the targets.
    For efficiency, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the
    best predictions, while the others are un-matched (and thus treated as "background").
    """

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        alpha: float = 0.25,
        gamma: float = 2.0
    ):
        """
        Args:
            cost_class: Weight for classification cost (focal loss)
            cost_bbox: Weight for L1 bbox cost
            cost_giou: Weight for GIoU cost
            alpha: Focal loss alpha parameter
            gamma: Focal loss gamma parameter
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.alpha = alpha
        self.gamma = gamma

        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, \
            "At least one cost must be non-zero"

    @torch.no_grad()
    def forward(
        self,
        outputs: dict,
        targets: List[dict]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform bipartite matching.

        Args:
            outputs: Dictionary with:
                - "pred_logits": [B, num_queries, num_classes]
                - "pred_boxes": [B, num_queries, 4] in cxcywh format normalized to [0, 1]
            targets: List of dictionaries (one per image) with:
                - "labels": [N_gt] class labels
                - "boxes": [N_gt, 4] bboxes in cxcywh format normalized to [0, 1]

        Returns:
            indices: List of (index_i, index_j) tuples where:
                - index_i: Indices of selected predictions (length N_gt)
                - index_j: Indices of ground truth objects (length N_gt)
        """
        batch_size, num_queries = outputs["pred_logits"].shape[:2]

        # Flatten batch dimension for vectorized computation
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [B*num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [B*num_queries, 4]

        # Concatenate all target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])  # [sum(N_gt)]
        tgt_bbox = torch.cat([v["boxes"] for v in targets])  # [sum(N_gt), 4]

        # Classification cost (focal loss cost)
        # Gather predicted probabilities for target classes
        out_prob = out_prob[:, tgt_ids]  # [B*num_queries, sum(N_gt)]

        # Focal loss cost
        neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * \
                         (-(1 - out_prob + 1e-8).log())
        pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * \
                         (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class - neg_cost_class

        # Bounding box L1 cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # [B*num_queries, sum(N_gt)]

        # GIoU cost
        out_bbox_xyxy = box_cxcywh_to_xyxy(out_bbox)
        tgt_bbox_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
        _, cost_giou = box_iou(out_bbox_xyxy, tgt_bbox_xyxy)
        cost_giou = -cost_giou  # Negate because we want to maximize GIoU

        # Final cost matrix
        C = self.cost_class * cost_class + \
            self.cost_bbox * cost_bbox + \
            self.cost_giou * cost_giou
        C = C.view(batch_size, num_queries, -1).cpu()  # [B, num_queries, sum(N_gt)]

        # Split cost matrix by image and perform Hungarian matching
        sizes = [len(v["boxes"]) for v in targets]
        indices = []
        offset = 0

        for i, c in enumerate(C.split(sizes, -1)):
            # c has shape [num_queries, N_gt_i]
            c = c[i]  # Select i-th batch element

            if sizes[i] > 0:
                # Perform Hungarian algorithm
                idx_i, idx_j = linear_sum_assignment(c)
                indices.append((
                    torch.as_tensor(idx_i, dtype=torch.int64),
                    torch.as_tensor(idx_j, dtype=torch.int64)
                ))
            else:
                # No ground truth objects in this image
                indices.append((
                    torch.as_tensor([], dtype=torch.int64),
                    torch.as_tensor([], dtype=torch.int64)
                ))

            offset += sizes[i]

        return indices


def build_matcher(
    cost_class: float = 2.0,
    cost_bbox: float = 5.0,
    cost_giou: float = 2.0
) -> HungarianMatcher:
    """Build Hungarian matcher"""
    return HungarianMatcher(
        cost_class=cost_class,
        cost_bbox=cost_bbox,
        cost_giou=cost_giou
    )


def test_matcher():
    """Test Hungarian matcher"""
    print("Testing Hungarian Matcher")
    print("=" * 60)

    batch_size = 2
    num_queries = 300
    num_classes = 80

    # Create dummy outputs
    outputs = {
        "pred_logits": torch.randn(batch_size, num_queries, num_classes),
        "pred_boxes": torch.rand(batch_size, num_queries, 4)  # Random boxes in [0, 1]
    }

    # Create dummy targets (2 images with different numbers of objects)
    targets = [
        {
            "labels": torch.randint(0, num_classes, (5,)),  # 5 objects in first image
            "boxes": torch.rand(5, 4)  # 5 random boxes
        },
        {
            "labels": torch.randint(0, num_classes, (3,)),  # 3 objects in second image
            "boxes": torch.rand(3, 4)  # 3 random boxes
        }
    ]

    print(f"Number of queries: {num_queries}")
    print(f"Ground truth objects per image: {[len(t['labels']) for t in targets]}")

    # Build matcher and perform matching
    matcher = build_matcher()
    indices = matcher(outputs, targets)

    print(f"\nMatching results:")
    for i, (idx_i, idx_j) in enumerate(indices):
        print(f"  Image {i}: Matched {len(idx_i)} pairs")
        print(f"    Prediction indices: {idx_i[:5]}...")  # Show first 5
        print(f"    Target indices: {idx_j[:5]}...")

    # Verify matching properties
    for i, (idx_i, idx_j) in enumerate(indices):
        expected_matches = len(targets[i]["labels"])
        assert len(idx_i) == expected_matches, \
            f"Image {i}: Expected {expected_matches} matches, got {len(idx_i)}"
        assert len(idx_j) == expected_matches, \
            f"Image {i}: Target indices length mismatch"

        # Verify indices are unique (bijective matching)
        assert len(set(idx_i.tolist())) == len(idx_i), \
            f"Image {i}: Duplicate prediction indices found!"
        assert len(set(idx_j.tolist())) == len(idx_j), \
            f"Image {i}: Duplicate target indices found!"

    print("\n" + "=" * 60)
    print("✅ Matcher test passed!")
    print("Hungarian matching correctly assigns predictions to targets.")


if __name__ == "__main__":
    test_matcher()
