"""Standalone evaluation script for RT-DETR model.

Evaluates a trained model on COCO validation set and reports metrics.

Usage:
    python scripts/evaluate.py --config configs/rtdetr_r50_1920x1080.yml --checkpoint checkpoints/best_checkpoint.pth
"""

import argparse
import sys
from pathlib import Path
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.misc import set_seed, get_device
from src.utils.checkpoint import CheckpointManager
from src.models.rtdetr import build_rtdetr
from src.data.coco_dataset import build_coco_dataloader
from src.engine.evaluator import evaluate_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate RT-DETR model")

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--conf_threshold',
        type=float,
        default=0.01,
        help='Confidence threshold (use low for COCO eval)'
    )
    parser.add_argument(
        '--nms_threshold',
        type=float,
        default=0.7,
        help='NMS IoU threshold'
    )
    parser.add_argument(
        '--max_detections',
        type=int,
        default=300,
        help='Maximum detections per image'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda:0, cuda:1, cpu, etc.)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )

    return parser.parse_args()


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Set random seed
    set_seed(config.seed)

    # Get device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device(
            config.hardware.get('device', None),
            config.hardware.get('gpu_ids', [0])[0]
        )

    print(f"Using device: {device}")

    # Build model
    print("\nBuilding model...")
    model = build_rtdetr(
        num_classes=config.data.num_classes,
        backbone_name=config.model.backbone.type,
        pretrained=False,  # Will load from checkpoint
        hidden_dim=config.model.encoder.hidden_dim,
        num_queries=config.model.decoder.num_queries,
        num_decoder_layers=config.model.decoder.num_decoder_layers,
        num_encoder_layers=config.model.encoder.num_encoder_layers,
        num_heads=config.model.decoder.num_heads,
        dim_feedforward=config.model.decoder.dim_feedforward,
        dropout=config.model.decoder.get('dropout', 0.0),
        activation=config.model.decoder.get('activation', 'relu'),
        num_denoising=config.model.decoder.get('num_denoising', 100),
        label_noise_ratio=config.model.decoder.get('label_noise_ratio', 0.5),
        box_noise_scale=config.model.decoder.get('box_noise_scale', 1.0),
    )

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print("✓ Checkpoint loaded successfully")

    # Print checkpoint info
    if 'epoch' in checkpoint:
        print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        if 'AP' in metrics:
            print(f"  Checkpoint AP: {metrics['AP']:.4f}")

    # Override batch size if specified
    if args.batch_size is not None:
        config.data.val.batch_size = args.batch_size

    # Build dataloader
    print("\nBuilding validation dataloader...")
    val_loader = build_coco_dataloader(config, split='val')
    print(f"Validation batches: {len(val_loader)}")
    print(f"Total images: {len(val_loader.dataset)}")

    # Run evaluation
    print("\n" + "="*60)
    print("Starting evaluation")
    print("="*60)
    print(f"Configuration: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input resolution: {config.input.width}×{config.input.height}")
    print(f"Confidence threshold: {args.conf_threshold}")
    print(f"NMS threshold: {args.nms_threshold}")
    print(f"Max detections: {args.max_detections}")
    print("="*60 + "\n")

    metrics = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
        max_detections=args.max_detections,
        label_to_cat_id=None
    )

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Average Precision (AP) @ IoU=0.50:0.95: {metrics['AP']:.4f}")
    print(f"Average Precision (AP) @ IoU=0.50:      {metrics['AP50']:.4f}")
    print(f"Average Precision (AP) @ IoU=0.75:      {metrics['AP75']:.4f}")
    print(f"AP (small objects):                      {metrics['AP_small']:.4f}")
    print(f"AP (medium objects):                     {metrics['AP_medium']:.4f}")
    print(f"AP (large objects):                      {metrics['AP_large']:.4f}")
    print(f"Average Recall (AR) @ max 1 det:        {metrics['AR1']:.4f}")
    print(f"Average Recall (AR) @ max 10 dets:      {metrics['AR10']:.4f}")
    print(f"Average Recall (AR) @ max 100 dets:     {metrics['AR100']:.4f}")
    print("="*60)

    # Save results to file
    results_dir = Path(config.checkpoint.save_dir) / "evaluation_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_name = Path(args.checkpoint).stem
    results_file = results_dir / f"{checkpoint_name}_results.txt"

    with open(results_file, 'w') as f:
        f.write("COCO Evaluation Results\n")
        f.write("="*60 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Configuration: {args.config}\n")
        f.write(f"Input resolution: {config.input.width}×{config.input.height}\n")
        f.write("\n")
        f.write("Metrics:\n")
        f.write("-"*60 + "\n")
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name:15s}: {metric_value:.4f}\n")
        f.write("="*60 + "\n")

    print(f"\nResults saved to: {results_file}")

    print("\nEvaluation completed successfully!")


if __name__ == '__main__':
    main()
