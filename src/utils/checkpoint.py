"""Checkpoint management for saving and loading model states.

Handles saving/loading of model, optimizer, scheduler states with metadata.
Supports best checkpoint tracking and resuming training.
"""

import torch
from torch import nn
from torch.optim import Optimizer
from pathlib import Path
from typing import Dict, Any, Optional
import json


class CheckpointManager:
    """Manages model checkpoints during training.

    Args:
        save_dir: Directory to save checkpoints
        save_best: Whether to save best checkpoint
        save_last: Whether to save last checkpoint
        monitor: Metric to monitor for best checkpoint
        mode: 'max' or 'min' for best metric
    """

    def __init__(
        self,
        save_dir: str,
        save_best: bool = True,
        save_last: bool = True,
        monitor: str = 'AP',
        mode: str = 'max'
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.save_best = save_best
        self.save_last = save_last
        self.monitor = monitor
        self.mode = mode

        # Track best metric
        if mode == 'max':
            self.best_metric = float('-inf')
        else:
            self.best_metric = float('inf')

        # Load metadata if exists
        self.metadata_file = self.save_dir / 'checkpoint_metadata.json'
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
                self.best_metric = metadata.get('best_metric', self.best_metric)

    def is_better(self, metric: float) -> bool:
        """Check if metric is better than current best.

        Args:
            metric: Metric value to compare

        Returns:
            True if metric is better
        """
        if self.mode == 'max':
            return metric > self.best_metric
        else:
            return metric < self.best_metric

    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Any,
        metrics: Optional[Dict[str, float]] = None,
        extra_state: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save checkpoint.

        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer to save
            scheduler: Scheduler to save
            metrics: Dictionary of metrics
            extra_state: Additional state to save
        """
        # Prepare checkpoint state
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        }

        if metrics is not None:
            checkpoint['metrics'] = metrics

        if extra_state is not None:
            checkpoint.update(extra_state)

        # Save last checkpoint
        if self.save_last:
            last_path = self.save_dir / 'last_checkpoint.pth'
            torch.save(checkpoint, last_path)
            print(f"Saved last checkpoint to {last_path}")

        # Save epoch checkpoint
        epoch_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, epoch_path)
        print(f"Saved checkpoint to {epoch_path}")

        # Save best checkpoint
        if self.save_best and metrics is not None and self.monitor in metrics:
            metric_value = metrics[self.monitor]

            if self.is_better(metric_value):
                self.best_metric = metric_value
                best_path = self.save_dir / 'best_checkpoint.pth'
                torch.save(checkpoint, best_path)
                print(f"Saved best checkpoint with {self.monitor}={metric_value:.4f} to {best_path}")

                # Update metadata
                metadata = {
                    'best_metric': self.best_metric,
                    'best_epoch': epoch,
                    'monitor': self.monitor,
                    'mode': self.mode
                }
                with open(self.metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """Load checkpoint from file.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            device: Device to load checkpoint to

        Returns:
            Dictionary with checkpoint metadata
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint from {checkpoint_path}")

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            if checkpoint['scheduler_state_dict'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        epoch = checkpoint.get('epoch', 0)
        metrics = checkpoint.get('metrics', {})

        print(f"Loaded checkpoint from epoch {epoch}")
        if metrics:
            metric_str = ', '.join([f"{k}={v:.4f}" for k, v in metrics.items()])
            print(f"Checkpoint metrics: {metric_str}")

        return checkpoint

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint.

        Returns:
            Path to latest checkpoint or None if not found
        """
        last_path = self.save_dir / 'last_checkpoint.pth'
        if last_path.exists():
            return last_path

        # Find latest epoch checkpoint
        checkpoints = list(self.save_dir.glob('checkpoint_epoch_*.pth'))
        if checkpoints:
            latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
            return latest

        return None

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint.

        Returns:
            Path to best checkpoint or None if not found
        """
        best_path = self.save_dir / 'best_checkpoint.pth'
        if best_path.exists():
            return best_path
        return None


def load_pretrained_weights(
    model: nn.Module,
    weights_path: str,
    strict: bool = False
) -> None:
    """Load pretrained weights into model.

    Args:
        model: Model to load weights into
        weights_path: Path to weights file
        strict: Whether to strictly enforce state dict keys match
    """
    weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    print(f"Loading pretrained weights from {weights_path}")

    checkpoint = torch.load(weights_path, map_location='cpu')

    # Extract state dict if checkpoint contains other info
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    if missing:
        print(f"Missing keys: {len(missing)}")
        if len(missing) <= 10:
            for key in missing:
                print(f"  - {key}")

    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
        if len(unexpected) <= 10:
            for key in unexpected:
                print(f"  - {key}")

    print("Pretrained weights loaded successfully")
