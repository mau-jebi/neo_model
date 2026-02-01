"""Model evaluation with COCO metrics.

Evaluates object detection model on validation set using COCO metrics.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List
from tqdm import tqdm
import numpy as np

from ..utils.metrics import COCOMetrics, convert_predictions_to_coco_format
from ..data.data_utils import box_cxcywh_to_xyxy


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    conf_threshold: float = 0.01,
    nms_threshold: float = 0.7,
    max_detections: int = 300,
    label_to_cat_id: Optional[Dict[int, int]] = None
) -> Dict[str, float]:
    """Evaluate model on validation set with COCO metrics.

    Args:
        model: Model to evaluate
        dataloader: Validation dataloader
        device: Device to run evaluation on
        conf_threshold: Confidence threshold for predictions (use low threshold for COCO eval)
        nms_threshold: NMS IoU threshold
        max_detections: Maximum detections per image
        label_to_cat_id: Mapping from continuous labels to COCO category IDs

    Returns:
        Dictionary of COCO metrics
    """
    model.eval()

    # Get COCO ground truth from dataset
    dataset = dataloader.dataset
    coco_gt = dataset.coco

    # If label mapping not provided, use dataset's mapping
    if label_to_cat_id is None:
        label_to_cat_id = dataset.label_to_cat_id

    # Collect predictions
    all_predictions = []

    print("Running evaluation...")
    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch['images'].to(device)
        image_ids = batch['image_ids']

        # Forward pass
        outputs = model(images)

        # Extract predictions
        # RT-DETR outputs: {'pred_logits': (B, N, C), 'pred_boxes': (B, N, 4)}
        pred_logits = outputs['pred_logits']  # (B, N, num_classes)
        pred_boxes = outputs['pred_boxes']    # (B, N, 4) in cxcywh format

        batch_size = pred_logits.shape[0]

        for i in range(batch_size):
            logits = pred_logits[i]  # (N, num_classes)
            boxes = pred_boxes[i]    # (N, 4)
            image_id = image_ids[i]

            # Get scores and labels
            scores = logits.sigmoid().max(dim=-1)
            pred_scores = scores.values
            pred_labels = scores.indices

            # Filter by confidence
            keep = pred_scores > conf_threshold
            pred_scores = pred_scores[keep]
            pred_labels = pred_labels[keep]
            boxes = boxes[keep]

            if len(pred_scores) == 0:
                continue

            # Convert boxes from cxcywh (normalized) to xyxy (pixel coordinates)
            orig_size = batch['orig_sizes'][i]
            h, w = orig_size[0].item(), orig_size[1].item()

            # Denormalize boxes
            boxes = boxes.clone()
            boxes[:, [0, 2]] *= w
            boxes[:, [1, 3]] *= h

            # Convert cxcywh to xyxy
            boxes = box_cxcywh_to_xyxy(boxes)

            # Clip boxes to image bounds
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h)

            # Apply NMS (simple per-class NMS)
            keep_indices = []
            for label in pred_labels.unique():
                label_mask = pred_labels == label
                label_boxes = boxes[label_mask]
                label_scores = pred_scores[label_mask]

                # NMS
                keep_nms = torch.ops.torchvision.nms(
                    label_boxes,
                    label_scores,
                    nms_threshold
                )

                # Get original indices
                label_indices = torch.where(label_mask)[0]
                keep_indices.extend(label_indices[keep_nms].tolist())

            keep_indices = torch.tensor(keep_indices, dtype=torch.long, device=device)

            pred_scores = pred_scores[keep_indices]
            pred_labels = pred_labels[keep_indices]
            boxes = boxes[keep_indices]

            # Keep top-k predictions
            if len(pred_scores) > max_detections:
                top_k = torch.topk(pred_scores, max_detections)
                pred_scores = pred_scores[top_k.indices]
                pred_labels = pred_labels[top_k.indices]
                boxes = boxes[top_k.indices]

            # Store prediction for this image
            all_predictions.append({
                'image_id': image_id,
                'boxes': boxes,
                'scores': pred_scores,
                'labels': pred_labels
            })

    # Convert to COCO format
    coco_predictions = convert_predictions_to_coco_format(
        all_predictions,
        label_to_cat_id
    )

    print(f"Total predictions: {len(coco_predictions)}")

    # Compute COCO metrics
    metrics_calculator = COCOMetrics(coco_gt)
    metrics = metrics_calculator.compute_metrics(coco_predictions)

    # Print metrics
    print("\n" + "="*60)
    print("COCO Evaluation Results:")
    print("="*60)
    print(COCOMetrics.format_metrics(metrics))
    print("="*60)

    return metrics


def evaluate_single_image(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    conf_threshold: float = 0.3,
    nms_threshold: float = 0.7,
    max_detections: int = 100
) -> Dict[str, torch.Tensor]:
    """Evaluate model on single image.

    Args:
        model: Model to use
        image: Input image tensor of shape (C, H, W) or (1, C, H, W)
        device: Device to run on
        conf_threshold: Confidence threshold
        nms_threshold: NMS threshold
        max_detections: Maximum detections

    Returns:
        Dictionary with 'boxes', 'scores', 'labels'
    """
    model.eval()

    # Add batch dimension if needed
    if image.dim() == 3:
        image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)

    # Extract predictions
    pred_logits = outputs['pred_logits'][0]  # (N, num_classes)
    pred_boxes = outputs['pred_boxes'][0]    # (N, 4)

    # Get scores and labels
    scores = pred_logits.sigmoid().max(dim=-1)
    pred_scores = scores.values
    pred_labels = scores.indices

    # Filter by confidence
    keep = pred_scores > conf_threshold
    pred_scores = pred_scores[keep]
    pred_labels = pred_labels[keep]
    pred_boxes = pred_boxes[keep]

    if len(pred_scores) == 0:
        return {
            'boxes': torch.zeros((0, 4), device=device),
            'scores': torch.zeros(0, device=device),
            'labels': torch.zeros(0, dtype=torch.long, device=device)
        }

    # Convert boxes from cxcywh (normalized) to xyxy (normalized)
    pred_boxes = box_cxcywh_to_xyxy(pred_boxes)

    # Apply NMS
    keep_indices = []
    for label in pred_labels.unique():
        label_mask = pred_labels == label
        label_boxes = pred_boxes[label_mask]
        label_scores = pred_scores[label_mask]

        keep_nms = torch.ops.torchvision.nms(
            label_boxes,
            label_scores,
            nms_threshold
        )

        label_indices = torch.where(label_mask)[0]
        keep_indices.extend(label_indices[keep_nms].tolist())

    keep_indices = torch.tensor(keep_indices, dtype=torch.long, device=device)

    pred_scores = pred_scores[keep_indices]
    pred_labels = pred_labels[keep_indices]
    pred_boxes = pred_boxes[keep_indices]

    # Keep top-k
    if len(pred_scores) > max_detections:
        top_k = torch.topk(pred_scores, max_detections)
        pred_scores = pred_scores[top_k.indices]
        pred_labels = pred_labels[top_k.indices]
        pred_boxes = pred_boxes[top_k.indices]

    return {
        'boxes': pred_boxes,
        'scores': pred_scores,
        'labels': pred_labels
    }
