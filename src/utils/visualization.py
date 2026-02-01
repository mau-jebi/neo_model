"""Visualization utilities for object detection.

Provides functions to draw bounding boxes, labels, and create detection visualizations.
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


# COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Color palette for visualization (BGR format for OpenCV)
COLOR_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128), (128, 0, 255), (0, 128, 255),
]


def get_color(class_id: int) -> Tuple[int, int, int]:
    """Get color for class ID.

    Args:
        class_id: Class ID

    Returns:
        BGR color tuple
    """
    return COLOR_PALETTE[class_id % len(COLOR_PALETTE)]


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: Optional[np.ndarray] = None,
    scores: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    line_thickness: int = 2,
    font_scale: float = 0.5
) -> np.ndarray:
    """Draw bounding boxes on image with labels and scores.

    Args:
        image: Input image (H, W, 3) in RGB or BGR format
        boxes: Bounding boxes in xyxy format, shape (N, 4)
        labels: Class labels, shape (N,)
        scores: Confidence scores, shape (N,)
        class_names: List of class names
        line_thickness: Thickness of bounding box lines
        font_scale: Font scale for text

    Returns:
        Image with drawn bounding boxes
    """
    image = image.copy()

    if class_names is None:
        class_names = COCO_CLASSES

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int)

        # Get label and color
        if labels is not None:
            label_id = int(labels[i])
            color = get_color(label_id)
            class_name = class_names[label_id] if label_id < len(class_names) else f"class_{label_id}"
        else:
            color = (0, 255, 0)
            class_name = ""

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

        # Prepare label text
        if scores is not None:
            score = scores[i]
            label_text = f"{class_name} {score:.2f}"
        else:
            label_text = class_name

        # Draw label background
        if label_text:
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )

            cv2.rectangle(
                image,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )

            # Draw label text
            cv2.putText(
                image,
                label_text,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

    return image


def visualize_predictions(
    image: np.ndarray,
    predictions: Dict[str, np.ndarray],
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = False
) -> np.ndarray:
    """Visualize model predictions on image.

    Args:
        image: Input image (H, W, 3) in RGB format
        predictions: Dictionary with 'boxes', 'scores', 'labels'
        class_names: List of class names
        save_path: Path to save visualization
        show: Whether to display image

    Returns:
        Visualization image
    """
    # Convert tensors to numpy
    boxes = predictions['boxes']
    scores = predictions['scores']
    labels = predictions['labels']

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Draw boxes
    vis_image = draw_bounding_boxes(
        image, boxes, labels, scores, class_names
    )

    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert RGB to BGR for saving with cv2
        vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path), vis_image_bgr)

    # Show if requested
    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(vis_image)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return vis_image


def visualize_ground_truth(
    image: np.ndarray,
    targets: Dict[str, np.ndarray],
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = False
) -> np.ndarray:
    """Visualize ground truth annotations on image.

    Args:
        image: Input image (H, W, 3) in RGB format
        targets: Dictionary with 'boxes', 'labels'
        class_names: List of class names
        save_path: Path to save visualization
        show: Whether to display image

    Returns:
        Visualization image
    """
    boxes = targets['boxes']
    labels = targets['labels']

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    vis_image = draw_bounding_boxes(
        image, boxes, labels, None, class_names
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path), vis_image_bgr)

    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(vis_image)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return vis_image


def create_comparison_visualization(
    image: np.ndarray,
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = False
) -> np.ndarray:
    """Create side-by-side comparison of predictions and ground truth.

    Args:
        image: Input image (H, W, 3) in RGB format
        predictions: Prediction dictionary
        targets: Ground truth dictionary
        class_names: List of class names
        save_path: Path to save visualization
        show: Whether to display image

    Returns:
        Comparison visualization
    """
    # Create visualizations
    pred_vis = visualize_predictions(image.copy(), predictions, class_names)
    gt_vis = visualize_ground_truth(image.copy(), targets, class_names)

    # Concatenate horizontally
    comparison = np.concatenate([pred_vis, gt_vis], axis=1)

    # Add titles
    h, w = comparison.shape[:2]
    title_height = 50
    titled_image = np.ones((h + title_height, w, 3), dtype=np.uint8) * 255
    titled_image[title_height:] = comparison

    # Add text titles
    cv2.putText(
        titled_image, "Predictions", (w//4 - 80, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2
    )
    cv2.putText(
        titled_image, "Ground Truth", (3*w//4 - 100, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), cv2.cvtColor(titled_image, cv2.COLOR_RGB2BGR))

    if show:
        plt.figure(figsize=(20, 10))
        plt.imshow(titled_image)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return titled_image


def plot_training_curves(
    log_file: str,
    save_path: Optional[str] = None,
    show: bool = False
):
    """Plot training curves from log file.

    Args:
        log_file: Path to training log file
        save_path: Path to save plot
        show: Whether to display plot
    """
    # This is a placeholder - implement based on actual log format
    # In practice, you'd parse the log file and extract metrics

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot loss
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)

    # Plot AP
    axes[0, 1].set_title('Average Precision')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AP')
    axes[0, 1].grid(True)

    # Plot learning rate
    axes[1, 0].set_title('Learning Rate')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('LR')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True)

    # Plot AR
    axes[1, 1].set_title('Average Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AR')
    axes[1, 1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    if show:
        plt.show()

    plt.close()
