#!/usr/bin/env python3
"""
Automated Training Pipeline for RT-DETR 1920x1080

End-to-end pipeline that:
1. Downloads labeled dataset from S3
2. Trains the RT-DETR model
3. Exports trained model to ONNX format
4. Uploads all artifacts back to S3

Usage:
    python scripts/automated_pipeline.py --config configs/pipeline_config.yml

    # With custom run ID
    python scripts/automated_pipeline.py --config configs/pipeline_config.yml --run-id my_experiment_001

    # Skip training (export existing checkpoint)
    python scripts/automated_pipeline.py --config configs/pipeline_config.yml --skip-training --checkpoint path/to/checkpoint.pth

Environment Variables:
    AWS_ACCESS_KEY_ID: AWS access key
    AWS_SECRET_ACCESS_KEY: AWS secret key
    AWS_DEFAULT_REGION: AWS region (optional, defaults to config value)
"""

import argparse
import logging
import os
import sys
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import yaml
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config, ConfigDict, merge_configs
from src.utils.s3_utils import (
    S3Client,
    download_coco_dataset_from_s3,
    upload_trained_model_to_s3,
    verify_s3_dataset_structure
)
from src.deployment.onnx_export import export_checkpoint_to_onnx, benchmark_onnx_model


# Setup logging
def setup_logging(log_level: str, log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger('pipeline')
    logger.setLevel(getattr(logging, log_level.upper()))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def load_pipeline_config(config_path: str) -> ConfigDict:
    """Load and process pipeline configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Process variable substitutions (e.g., ${base_dir})
    if 'local' in config:
        base_dir = config['local'].get('base_dir', '/tmp/neo_model_pipeline')
        for key, value in config['local'].items():
            if isinstance(value, str) and '${base_dir}' in value:
                config['local'][key] = value.replace('${base_dir}', base_dir)

    return ConfigDict(config)


def generate_run_id() -> str:
    """Generate a unique run ID based on timestamp."""
    return datetime.utcnow().strftime('%Y%m%d_%H%M%S')


class TrainingPipeline:
    """
    Orchestrates the complete training pipeline.

    Stages:
    1. Setup - Initialize directories, logging, S3 client
    2. Download - Fetch dataset from S3
    3. Train - Run model training
    4. Export - Convert model to ONNX
    5. Upload - Push artifacts to S3
    """

    def __init__(self, config: ConfigDict, run_id: str, logger: logging.Logger):
        self.config = config
        self.run_id = run_id
        self.logger = logger

        # Initialize paths
        self.base_dir = Path(config.local.base_dir)
        self.data_dir = Path(config.local.data_dir)
        self.output_dir = Path(config.local.output_dir) / run_id

        # Track pipeline state
        self.state = {
            'run_id': run_id,
            'start_time': datetime.utcnow().isoformat(),
            'stages_completed': [],
            'artifacts': {},
            'metrics': {}
        }

        # Initialize S3 client
        self.s3_client = None

    def setup(self) -> bool:
        """Stage 1: Setup directories and S3 connection."""
        self.logger.info("=" * 60)
        self.logger.info("STAGE 1: Setup")
        self.logger.info("=" * 60)

        # Create directories
        self.logger.info(f"Creating directories...")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.output_dir / 'onnx').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        (self.output_dir / 'config').mkdir(exist_ok=True)

        self.logger.info(f"  Base directory: {self.base_dir}")
        self.logger.info(f"  Data directory: {self.data_dir}")
        self.logger.info(f"  Output directory: {self.output_dir}")

        # Initialize S3 client
        self.logger.info("Initializing S3 client...")
        try:
            self.s3_client = S3Client(
                aws_region=self.config.s3_input.get('region', 'us-east-1'),
                endpoint_url=self.config.s3_input.get('endpoint_url')
            )

            if not self.s3_client.validate_credentials():
                self.logger.error("S3 credentials validation failed")
                return False

            self.logger.info("  S3 client initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize S3 client: {e}")
            return False

        self.state['stages_completed'].append('setup')
        return True

    def download_data(self) -> bool:
        """Stage 2: Download dataset from S3."""
        self.logger.info("=" * 60)
        self.logger.info("STAGE 2: Download Dataset from S3")
        self.logger.info("=" * 60)

        bucket = self.config.s3_input.bucket
        data_prefix = self.config.s3_input.data_prefix

        self.logger.info(f"Source: s3://{bucket}/{data_prefix}")
        self.logger.info(f"Destination: {self.data_dir}")

        # Verify dataset structure if requested
        if self.config.pipeline_options.get('verify_dataset', True):
            self.logger.info("Verifying S3 dataset structure...")
            verification = verify_s3_dataset_structure(
                self.s3_client, bucket, data_prefix
            )

            if not verification['valid']:
                self.logger.error("Dataset structure verification failed:")
                for error in verification['errors']:
                    self.logger.error(f"  - {error}")
                return False

            if verification['warnings']:
                for warning in verification['warnings']:
                    self.logger.warning(f"  - {warning}")

            self.logger.info(f"  Annotation files: {verification['stats']['annotation_files']}")
            self.logger.info("  Dataset structure verified")

        # Download dataset
        self.logger.info("Downloading dataset (this may take a while)...")
        try:
            success = download_coco_dataset_from_s3(
                self.s3_client,
                bucket=bucket,
                data_prefix=data_prefix,
                local_data_dir=str(self.data_dir),
                splits=['train', 'val']
            )

            if not success:
                self.logger.error("Dataset download failed")
                return False

            self.logger.info("Dataset downloaded successfully")
            self.state['artifacts']['data_dir'] = str(self.data_dir)

        except Exception as e:
            self.logger.error(f"Dataset download failed: {e}")
            return False

        self.state['stages_completed'].append('download_data')
        return True

    def train_model(self, resume_checkpoint: Optional[str] = None) -> Optional[str]:
        """
        Stage 3: Train the model.

        Returns:
            Path to best checkpoint, or None if training failed
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 3: Model Training")
        self.logger.info("=" * 60)

        # Load training config
        training_config_path = Path(__file__).parent.parent / self.config.training.config_file
        self.logger.info(f"Loading training config: {training_config_path}")

        try:
            training_config = load_config(str(training_config_path))
        except Exception as e:
            self.logger.error(f"Failed to load training config: {e}")
            return None

        # Apply overrides from pipeline config
        if 'overrides' in self.config.training:
            overrides = self.config.training.overrides
            self.logger.info("Applying training overrides:")

            if 'epochs' in overrides:
                training_config.training.epochs = overrides.epochs
                self.logger.info(f"  epochs: {overrides.epochs}")

            if 'batch_size' in overrides:
                training_config.data.train.batch_size = overrides.batch_size
                training_config.data.val.batch_size = overrides.batch_size
                self.logger.info(f"  batch_size: {overrides.batch_size}")

            if 'lr' in overrides:
                training_config.training.optimizer.lr = overrides.lr
                self.logger.info(f"  learning_rate: {overrides.lr}")

            if 'use_amp' in overrides:
                training_config.training.use_amp = overrides.use_amp
                self.logger.info(f"  use_amp: {overrides.use_amp}")

            if 'accumulate_grad_batches' in overrides:
                training_config.training.accumulate_grad_batches = overrides.accumulate_grad_batches
                self.logger.info(f"  accumulate_grad_batches: {overrides.accumulate_grad_batches}")

        # Update paths in training config
        training_config.data.train.root = str(self.data_dir / 'images' / 'train2017')
        training_config.data.train.annotation = str(self.data_dir / 'annotations' / 'instances_train2017.json')
        training_config.data.val.root = str(self.data_dir / 'images' / 'val2017')
        training_config.data.val.annotation = str(self.data_dir / 'annotations' / 'instances_val2017.json')
        training_config.checkpoint.save_dir = str(self.output_dir / 'checkpoints')
        training_config.logging.log_dir = str(self.output_dir / 'logs')

        # Save modified training config
        config_save_path = self.output_dir / 'config' / 'training_config.yml'
        with open(config_save_path, 'w') as f:
            yaml.dump(dict(training_config), f, default_flow_style=False)
        self.logger.info(f"Saved training config to: {config_save_path}")

        # Import training components
        from src.utils.misc import set_seed, get_device, count_parameters
        from src.models.rtdetr import build_rtdetr
        from src.models.criterion import build_criterion
        from src.data.coco_dataset import build_coco_dataloader
        from src.engine.optimizer import build_optimizer
        from src.engine.scheduler import build_scheduler
        from src.engine.trainer import build_trainer

        # Set random seed
        set_seed(training_config.seed, training_config.get('deterministic', False))

        # Get device
        device_str = self.config.hardware.get('device', 'cuda')
        device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {device}")

        # Enable cudnn benchmark
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        # Build model
        self.logger.info("Building model...")
        model = build_rtdetr(
            num_classes=training_config.data.num_classes,
            backbone_type=training_config.model.backbone.type,
            backbone_pretrained=training_config.model.backbone.get('pretrained', True),
            encoder_hidden_dim=training_config.model.encoder.hidden_dim,
            num_queries=training_config.model.decoder.num_queries,
            num_decoder_layers=training_config.model.decoder.num_decoder_layers,
            num_encoder_layers=training_config.model.encoder.num_encoder_layers,
            num_heads=training_config.model.decoder.num_heads,
            dim_feedforward=training_config.model.decoder.dim_feedforward,
            dropout=training_config.model.decoder.get('dropout', 0.0),
        )
        model = model.to(device)

        num_params = count_parameters(model, trainable_only=True)
        self.logger.info(f"Model parameters: {num_params:,} (trainable)")

        # Build criterion
        self.logger.info("Building criterion...")
        criterion = build_criterion(
            num_classes=training_config.data.num_classes,
            weight_dict=training_config.model.criterion.weight_dict,
            alpha=training_config.model.criterion.get("alpha", 0.75),
            gamma=training_config.model.criterion.get("gamma", 2.0)
        )
        criterion = criterion.to(device)

        # Build dataloaders
        self.logger.info("Building dataloaders...")
        train_loader = build_coco_dataloader(training_config, split='train')
        val_loader = build_coco_dataloader(training_config, split='val')

        self.logger.info(f"Training batches: {len(train_loader)}")
        self.logger.info(f"Validation batches: {len(val_loader)}")

        # Build optimizer and scheduler
        optimizer = build_optimizer(training_config, model)
        scheduler = build_scheduler(training_config, optimizer)

        # Build trainer
        trainer = build_trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            config=training_config,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader
        )

        # Start training
        self.logger.info("Starting training...")
        self.logger.info(f"  Epochs: {training_config.training.epochs}")
        self.logger.info(f"  Batch size: {training_config.data.train.batch_size}")
        self.logger.info(f"  Mixed precision: {training_config.training.use_amp}")

        try:
            trainer.train(resume_from=resume_checkpoint)
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None

        # Find best checkpoint
        checkpoint_dir = self.output_dir / 'checkpoints'
        best_checkpoint = checkpoint_dir / 'best_checkpoint.pth'

        if not best_checkpoint.exists():
            # Fall back to last checkpoint
            best_checkpoint = checkpoint_dir / 'last_checkpoint.pth'

        if best_checkpoint.exists():
            self.logger.info(f"Training completed. Best checkpoint: {best_checkpoint}")
            self.state['artifacts']['best_checkpoint'] = str(best_checkpoint)
            self.state['stages_completed'].append('train_model')
            return str(best_checkpoint)
        else:
            self.logger.error("No checkpoint found after training")
            return None

    def export_onnx(self, checkpoint_path: str) -> Optional[str]:
        """
        Stage 4: Export model to ONNX format.

        Args:
            checkpoint_path: Path to trained checkpoint

        Returns:
            Path to ONNX model, or None if export failed
        """
        if not self.config.onnx_export.get('enabled', True):
            self.logger.info("ONNX export disabled in config, skipping...")
            return None

        self.logger.info("=" * 60)
        self.logger.info("STAGE 4: ONNX Export")
        self.logger.info("=" * 60)

        output_name = self.config.onnx_export.get('output_name', 'model.onnx')
        onnx_path = self.output_dir / 'onnx' / output_name

        self.logger.info(f"Checkpoint: {checkpoint_path}")
        self.logger.info(f"Output: {onnx_path}")

        # Load training config to get model parameters
        training_config_path = Path(__file__).parent.parent / self.config.training.config_file
        training_config = load_config(str(training_config_path))

        try:
            metadata = export_checkpoint_to_onnx(
                checkpoint_path=checkpoint_path,
                output_path=str(onnx_path),
                input_height=training_config.input.height,
                input_width=training_config.input.width,
                num_classes=training_config.data.num_classes,
                num_queries=training_config.model.decoder.num_queries,
                opset_version=self.config.onnx_export.get('opset_version', 17),
                dynamic_batch=self.config.onnx_export.get('dynamic_batch', True),
                simplify=self.config.onnx_export.get('simplify', True),
                verify=self.config.onnx_export.get('verify', True),
                fp16=self.config.onnx_export.get('fp16', False)
            )

            self.logger.info(f"ONNX export successful: {onnx_path}")
            self.logger.info(f"  Model size: {metadata['model_size_mb']:.2f} MB")

            if metadata.get('verification', {}).get('passed'):
                self.logger.info("  Verification: PASSED")
            else:
                self.logger.warning("  Verification: FAILED")

            # Run benchmark if requested
            if self.config.onnx_export.get('benchmark', True):
                self.logger.info("Running ONNX benchmark...")
                bench_results = benchmark_onnx_model(
                    str(onnx_path),
                    input_shape=(1, 3, training_config.input.height, training_config.input.width),
                    use_cuda=torch.cuda.is_available()
                )
                self.logger.info(f"  Throughput: {bench_results['throughput_fps']:.1f} FPS")
                self.logger.info(f"  Mean latency: {bench_results['mean_latency_ms']:.2f} ms")
                metadata['benchmark'] = bench_results

            self.state['artifacts']['onnx_model'] = str(onnx_path)
            self.state['metrics']['onnx_export'] = metadata
            self.state['stages_completed'].append('export_onnx')

            return str(onnx_path)

        except Exception as e:
            self.logger.error(f"ONNX export failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def upload_artifacts(self, checkpoint_path: str, onnx_path: Optional[str] = None) -> bool:
        """Stage 5: Upload artifacts to S3."""
        self.logger.info("=" * 60)
        self.logger.info("STAGE 5: Upload Artifacts to S3")
        self.logger.info("=" * 60)

        bucket = self.config.s3_output.bucket
        output_prefix = f"{self.config.s3_output.output_prefix}/{self.run_id}"

        self.logger.info(f"Destination: s3://{bucket}/{output_prefix}")

        # Prepare metadata
        metadata = {
            'run_id': self.run_id,
            'pipeline_config': dict(self.config),
            'state': self.state
        }

        # Upload artifacts
        try:
            uploaded = upload_trained_model_to_s3(
                s3_client=self.s3_client,
                bucket=bucket,
                output_prefix=output_prefix,
                checkpoint_path=checkpoint_path,
                onnx_path=onnx_path,
                config_path=str(self.output_dir / 'config' / 'training_config.yml'),
                training_logs_dir=str(self.output_dir / 'logs'),
                metadata=metadata
            )

            self.logger.info("Uploaded artifacts:")
            for artifact_type, s3_uri in uploaded.items():
                self.logger.info(f"  {artifact_type}: {s3_uri}")

            self.state['artifacts']['s3_outputs'] = uploaded
            self.state['stages_completed'].append('upload_artifacts')

            return True

        except Exception as e:
            self.logger.error(f"Upload failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def cleanup(self):
        """Cleanup local files if configured."""
        if self.config.local.get('cleanup_after_upload', False):
            self.logger.info("Cleaning up local files...")
            try:
                if self.data_dir.exists():
                    shutil.rmtree(self.data_dir)
                    self.logger.info(f"  Removed: {self.data_dir}")
            except Exception as e:
                self.logger.warning(f"Cleanup failed: {e}")

    def save_state(self):
        """Save pipeline state to file."""
        self.state['end_time'] = datetime.utcnow().isoformat()
        state_path = self.output_dir / 'pipeline_state.json'
        with open(state_path, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)
        self.logger.info(f"Pipeline state saved to: {state_path}")

    def run(
        self,
        skip_download: bool = False,
        skip_training: bool = False,
        checkpoint_path: Optional[str] = None
    ) -> bool:
        """
        Run the complete pipeline.

        Args:
            skip_download: Skip data download (use existing local data)
            skip_training: Skip training (export existing checkpoint)
            checkpoint_path: Path to existing checkpoint (for skip_training)

        Returns:
            True if pipeline completed successfully
        """
        self.logger.info("=" * 60)
        self.logger.info(f"RT-DETR AUTOMATED TRAINING PIPELINE")
        self.logger.info(f"Run ID: {self.run_id}")
        self.logger.info("=" * 60)

        try:
            # Stage 1: Setup
            if not self.setup():
                return False

            # Stage 2: Download data
            if not skip_download:
                if not self.download_data():
                    return False
            else:
                self.logger.info("Skipping data download (--skip-download)")

            # Stage 3: Train model
            if not skip_training:
                resume_checkpoint = self.config.training.get('resume_checkpoint')
                checkpoint_path = self.train_model(resume_checkpoint)
                if not checkpoint_path:
                    return False
            else:
                self.logger.info("Skipping training (--skip-training)")
                if not checkpoint_path:
                    self.logger.error("--checkpoint is required when using --skip-training")
                    return False

            # Stage 4: Export to ONNX
            onnx_path = self.export_onnx(checkpoint_path)

            # Stage 5: Upload to S3
            if not self.upload_artifacts(checkpoint_path, onnx_path):
                return False

            # Cleanup
            self.cleanup()

            # Save final state
            self.save_state()

            self.logger.info("=" * 60)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 60)
            self.logger.info(f"Run ID: {self.run_id}")
            self.logger.info(f"Artifacts uploaded to: s3://{self.config.s3_output.bucket}/{self.config.s3_output.output_prefix}/{self.run_id}/")

            return True

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            self.save_state()
            return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Automated RT-DETR Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python scripts/automated_pipeline.py --config configs/pipeline_config.yml

  # With custom run ID
  python scripts/automated_pipeline.py --config configs/pipeline_config.yml --run-id experiment_001

  # Skip data download (use existing local data)
  python scripts/automated_pipeline.py --config configs/pipeline_config.yml --skip-download

  # Skip training and export existing checkpoint
  python scripts/automated_pipeline.py --config configs/pipeline_config.yml --skip-training --checkpoint checkpoints/best.pth
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to pipeline configuration file'
    )
    parser.add_argument(
        '--run-id',
        type=str,
        default=None,
        help='Custom run ID (default: auto-generated timestamp)'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip data download (use existing local data)'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training (export existing checkpoint)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to existing checkpoint (required with --skip-training)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load pipeline config
    config = load_pipeline_config(args.config)

    # Generate or use provided run ID
    if args.run_id:
        run_id = args.run_id
    elif config.pipeline_options.get('auto_run_id', True):
        run_id = generate_run_id()
    else:
        run_id = config.pipeline_options.get('run_id', generate_run_id())

    # Setup logging
    log_file = None
    if config.logging.get('log_to_file', True):
        log_dir = Path(config.local.output_dir) / run_id
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = str(log_dir / config.logging.get('log_file', 'pipeline.log'))

    logger = setup_logging(
        log_level=args.log_level or config.logging.get('level', 'INFO'),
        log_file=log_file
    )

    # Create and run pipeline
    pipeline = TrainingPipeline(config, run_id, logger)

    success = pipeline.run(
        skip_download=args.skip_download,
        skip_training=args.skip_training,
        checkpoint_path=args.checkpoint
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
