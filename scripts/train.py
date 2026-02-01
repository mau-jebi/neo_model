"""Main training script for RT-DETR at 1920×1080 resolution.

Usage:
    python scripts/train.py --config configs/rtdetr_r50_1920x1080.yml
    python scripts/train.py --config configs/rtdetr_r50_1920x1080.yml --resume checkpoints/last_checkpoint.pth
"""

import argparse
import sys
from pathlib import Path
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.misc import set_seed, get_device, count_parameters
from src.models.rtdetr import build_rtdetr
from src.models.criterion import build_criterion
from src.data.coco_dataset import build_coco_dataloader
from src.engine.optimizer import build_optimizer
from src.engine.scheduler import build_scheduler
from src.engine.trainer import build_trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train RT-DETR model")

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (overfit on small batch)'
    )
    parser.add_argument(
        '--overfit_batches',
        type=int,
        default=0,
        help='Number of batches to overfit on (for testing)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda:0, cuda:1, cpu, etc.)'
    )

    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Override debug settings
    if args.debug:
        config.debug.enabled = True
        config.debug.overfit_batches = args.overfit_batches or 5

    if args.overfit_batches > 0:
        config.debug.overfit_batches = args.overfit_batches

    # Set random seed
    set_seed(config.seed, config.get('deterministic', False))

    # Get device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device(
            config.hardware.get('device', None),
            config.hardware.get('gpu_ids', [0])[0]
        )

    print(f"Using device: {device}")

    # Enable cudnn benchmark
    if device.type == 'cuda' and config.hardware.get('benchmark', True):
        torch.backends.cudnn.benchmark = True
        print("Enabled cudnn.benchmark for faster training")

    # Build model
    print("\nBuilding model...")
    model = build_rtdetr(
        num_classes=config.data.num_classes,
        backbone_name=config.model.backbone.type,
        pretrained=config.model.backbone.get('pretrained', True),
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
    model = model.to(device)

    # Print model info
    num_params = count_parameters(model, trainable_only=True)
    print(f"Model parameters: {num_params:,} (trainable)")

    # Build criterion
    print("\nBuilding criterion...")
    criterion = build_criterion(config)
    criterion = criterion.to(device)

    # Build dataloaders
    print("\nBuilding dataloaders...")
    train_loader = build_coco_dataloader(config, split='train')
    val_loader = build_coco_dataloader(config, split='val')

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Build optimizer
    print("\nBuilding optimizer...")
    optimizer = build_optimizer(config, model)

    # Build scheduler
    print("Building scheduler...")
    scheduler = build_scheduler(config, optimizer)

    # Build trainer
    print("\nBuilding trainer...")
    trainer = build_trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader
    )

    # Load pretrained weights if specified
    if config.training.get('pretrained_weights'):
        from src.utils.checkpoint import load_pretrained_weights
        print(f"\nLoading pretrained weights from {config.training.pretrained_weights}")
        load_pretrained_weights(model, config.training.pretrained_weights, strict=False)

    # Start training
    print("\n" + "="*60)
    print("Starting training")
    print("="*60)
    print(f"Configuration: {args.config}")
    print(f"Input resolution: {config.input.width}×{config.input.height}")
    print(f"Batch size: {config.data.train.batch_size}")
    print(f"Accumulation steps: {config.training.accumulate_grad_batches}")
    print(f"Effective batch size: {config.data.train.batch_size * config.training.accumulate_grad_batches}")
    print(f"Training epochs: {config.training.epochs}")
    print(f"Learning rate: {config.training.optimizer.lr}")
    print(f"Mixed precision: {config.training.use_amp}")
    print(f"Checkpoints: {config.checkpoint.save_dir}")
    print(f"Logs: {config.logging.log_dir}")
    print("="*60 + "\n")

    # Resume from checkpoint if specified
    resume_from = args.resume or config.training.get('resume_from')

    try:
        trainer.train(resume_from=resume_from)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint before exit...")

        # Save final checkpoint
        trainer.checkpoint_manager.save_checkpoint(
            epoch=trainer.current_epoch,
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            metrics={},
            extra_state={'interrupted': True}
        )

        print("Checkpoint saved. You can resume training with:")
        print(f"  python scripts/train.py --config {args.config} --resume checkpoints/last_checkpoint.pth")

    except Exception as e:
        print(f"\n\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()
