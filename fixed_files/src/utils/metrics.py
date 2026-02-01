"""Metrics computation for object detection evaluation.

Provides COCO metrics (AP, AP50, AP75, etc.) using pycocotools.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import tempfile
from pathlib import Path


class COCOMetrics:
    """COCO evaluation metrics calculator.

    Computes standard COCO metrics:
    - AP (Average Precision averaged over IoU thresholds 0.5:0.95)
    - AP50 (AP at IoU=0.5)
    - AP75 (AP at IoU=0.75)
    - AP_small (AP for small objects: area < 32^2)
    - AP_medium (AP for medium objects: 32^2 < area < 96^2)
    - AP_large (AP for large objects: area > 96^2)
    """

    def __init__(self, coco_gt: COCO):
        """Initialize metrics calculator.

        Args:
            coco_gt: COCO ground truth object
        """
        self.coco_gt = coco_gt

    def compute_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute COCO metrics from predictions.

        Args:
            predictions: List of prediction dictionaries with keys:
                - image_id: Image ID
                - category_id: Category ID (COCO format)
                - bbox: Bounding box in [x, y, w, h] format
                - score: Confidence score

        Returns:
            Dictionary of metrics
        """
        if len(predictions) == 0:
            print("Warning: No predictions to evaluate")
            return {
                'AP': 0.0,
                'AP50': 0.0,
                'AP75': 0.0,
                'AP_small': 0.0,
                'AP_medium': 0.0,
                'AP_large': 0.0,
                'AR1': 0.0,
                'AR10': 0.0,
                'AR100': 0.0,
            }

        # Load predictions into COCO format
        coco_dt = self.coco_gt.loadRes(predictions)

        # Run COCO evaluation
        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract metrics
        metrics = {
            'AP': coco_eval.stats[0],        # AP @ IoU=0.50:0.95
            'AP50': coco_eval.stats[1],      # AP @ IoU=0.50
            'AP75': coco_eval.stats[2],      # AP @ IoU=0.75
            'AP_small': coco_eval.stats[3],  # AP for small objects
            'AP_medium': coco_eval.stats[4], # AP for medium objects
            'AP_large': coco_eval.stats[5],  # AP for large objects
            'AR1': coco_eval.stats[6],       # AR with 1 detection per image
            'AR10': coco_eval.stats[7],      # AR with 10 detections per image
            'AR100': coco_eval.stats[8],     # AR with 100 detections per image
        }

        return metrics

    @staticmethod
    def format_metrics(metrics: Dict[str, float]) -> str:
        """Format metrics as string for logging.

        Args:
            metrics: Dictionary of metrics

        Returns:
            Formatted string
        """
        return (
            f"AP: {metrics['AP']:.4f} | "
            f"AP50: {metrics['AP50']:.4f} | "
            f"AP75: {metrics['AP75']:.4f} | "
            f"AP_s: {metrics['AP_small']:.4f} | "
            f"AP_m: {metrics['AP_medium']:.4f} | "
            f"AP_l: {metrics['AP_large']:.4f}"
        )


def convert_predictions_to_coco_format(
    predictions: List[Dict[str, Any]],
    label_to_cat_id: Dict[int, int]
) -> List[Dict[str, Any]]:
    """Convert model predictions to COCO format.

    Args:
        predictions: List of prediction dictionaries with keys:
            - image_id: Image ID
            - boxes: Boxes in xyxy format, shape (N, 4)
            - scores: Confidence scores, shape (N,)
            - labels: Class labels (continuous 0-based), shape (N,)
        label_to_cat_id: Mapping from continuous labels to COCO category IDs

    Returns:
        List of COCO format predictions
    """
    coco_predictions = []

    for pred in predictions:
        image_id = pred['image_id']
        boxes = pred['boxes']
        scores = pred['scores']
        labels = pred['labels']

        # Convert tensors to numpy
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        # Convert each detection to COCO format
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box

            # Convert xyxy to xywh
            x, y = float(x1), float(y1)
            w, h = float(x2 - x1), float(y2 - y1)

            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue

            # Convert label to COCO category ID
            cat_id = label_to_cat_id.get(int(label), int(label) + 1)

            coco_predictions.append({
                'image_id': int(image_id),
                'category_id': int(cat_id),
                'bbox': [x, y, w, h],
                'score': float(score)
            })

    return coco_predictions


class AveragePrecisionMeter:
    """Simple Average Precision meter for quick evaluation without COCO API."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.predictions = []
        self.targets = []

    def update(self, pred_boxes, pred_scores, pred_labels, target_boxes, target_labels):
        """Update with batch of predictions and targets.

        Args:
            pred_boxes: Predicted boxes, shape (N, 4)
            pred_scores: Prediction scores, shape (N,)
            pred_labels: Predicted labels, shape (N,)
            target_boxes: Ground truth boxes, shape (M, 4)
            target_labels: Ground truth labels, shape (M,)
        """
        self.predictions.append({
            'boxes': pred_boxes,
            'scores': pred_scores,
            'labels': pred_labels
        })

        self.targets.append({
            'boxes': target_boxes,
            'labels': target_labels
        })

    def compute_ap(self, iou_threshold: float = 0.5) -> float:
        """Compute Average Precision at given IoU threshold.

        Args:
            iou_threshold: IoU threshold for positive matches

        Returns:
            Average Precision
        """
        # This is a simplified implementation
        # For production, use COCO metrics
        from .data_utils import box_iou

        all_scores = []
        all_matched = []

        for pred, target in zip(self.predictions, self.targets):
            if len(pred['boxes']) == 0:
                continue

            if len(target['boxes']) == 0:
                # No ground truth, all predictions are false positives
                all_scores.extend(pred['scores'].tolist())
                all_matched.extend([False] * len(pred['boxes']))
                continue

            # Compute IoU matrix
            ious = box_iou(pred['boxes'], target['boxes'])

            # Match predictions to targets
            for i in range(len(pred['boxes'])):
                score = pred['scores'][i]
                label = pred['labels'][i]

                # Find best matching target
                best_iou = 0
                best_match = -1
                for j in range(len(target['boxes'])):
                    if target['labels'][j] == label and ious[i, j] > best_iou:
                        best_iou = ious[i, j]
                        best_match = j

                all_scores.append(score)
                all_matched.append(best_iou >= iou_threshold)

        if len(all_scores) == 0:
            return 0.0

        # Sort by score
        indices = np.argsort(all_scores)[::-1]
        all_matched = np.array(all_matched)[indices]

        # Compute precision and recall
        tp = np.cumsum(all_matched)
        fp = np.cumsum(~all_matched)

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (len(self.targets) + 1e-6)

        # Compute AP (area under PR curve)
        ap = 0.0
        for i in range(1, len(precision)):
            ap += (recall[i] - recall[i-1]) * precision[i]

        return ap


def calculate_map(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    iou_thresholds: List[float] = None
) -> Dict[str, float]:
    """Calculate mean Average Precision.

    Args:
        predictions: List of predictions
        targets: List of ground truth targets
        iou_thresholds: IoU thresholds to evaluate (default: [0.5, 0.75, 0.95])

    Returns:
        Dictionary with mAP metrics
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75, 0.95]

    meter = AveragePrecisionMeter()

    for pred, target in zip(predictions, targets):
        meter.update(
            pred['boxes'],
            pred['scores'],
            pred['labels'],
            target['boxes'],
            target['labels']
        )

    metrics = {}
    for thresh in iou_thresholds:
        ap = meter.compute_ap(thresh)
        metrics[f'AP{int(thresh*100)}'] = ap

    # Compute mean AP
    metrics['mAP'] = np.mean(list(metrics.values()))

    return metrics
