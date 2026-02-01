"""Training engine for neo_model with mixed precision and gradient accumulation.

Implements the main training loop with:
- Automatic Mixed Precision (AMP) for memory efficiency and speedup
- Gradient accumulation to simulate larger batch sizes
- Linear warmup for learning rate
- Gradient clipping for training stability
- TensorBoard logging
- Checkpoint management
"""

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from tqdm import tqdm
from pathlib import Path

from ..utils.logger import TrainingLogger
from ..utils.checkpoint import CheckpointManager
from ..utils.misc import AverageMeter, Timer
from .evaluator import evaluate_model


class Trainer:
    """Training manager for RT-DETR model.

    Handles training loop, evaluation, checkpointing, and logging.

    Args:
        model: RT-DETR model to train
        criterion: Loss criterion
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config: Configuration object
        device: Device to train on
        train_loader: Training dataloader
        val_loader: Validation dataloader
        logger: Training logger
        checkpoint_manager: Checkpoint manager
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: Any,
        config: Any,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: TrainingLogger,
        checkpoint_manager: CheckpointManager
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager

        # Training settings
        self.epochs = config.training.epochs
        self.use_amp = config.training.get('use_amp', True)
        self.accumulate_grad_batches = config.training.get('accumulate_grad_batches', 1)
        self.clip_max_norm = config.training.get('clip_max_norm', 0.1)
        self.eval_interval = config.training.get('eval_interval', 2)
        self.save_interval = config.training.get('save_interval', 5)
        self.log_interval = config.logging.get('log_interval', 50)

        # Gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0

        # Debug settings
        self.debug = config.debug.get('enabled', False)
        self.overfit_batches = config.debug.get('overfit_batches', 0)

        self.logger.info(f"Trainer initialized")
        self.logger.info(f"Training for {self.epochs} epochs")
        self.logger.info(f"Mixed precision: {self.use_amp}")
        self.logger.info(f"Gradient accumulation: {self.accumulate_grad_batches}")
        self.logger.info(f"Effective batch size: {train_loader.batch_size * self.accumulate_grad_batches}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of average losses
        """
        self.model.train()
        self.criterion.train()

        # Loss meters
        loss_meters = {
            'loss_total': AverageMeter('total'),
            'loss_vfl': AverageMeter('vfl'),
            'loss_bbox': AverageMeter('bbox'),
            'loss_giou': AverageMeter('giou'),
        }

        # Timer
        epoch_timer = Timer()
        epoch_timer.start()

        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}")

        # Zero gradients at start
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(pbar):
            # For overfitting debug mode
            if self.overfit_batches > 0 and batch_idx >= self.overfit_batches:
                break

            images = batch['images'].to(self.device)
            boxes_list = batch['boxes']
            labels_list = batch['labels']

            # Prepare targets for criterion
            targets = []
            for i in range(len(boxes_list)):
                target = {
                    'boxes': boxes_list[i].to(self.device),
                    'labels': labels_list[i].to(self.device)
                }
                targets.append(target)

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=torch.float16):
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, targets)

                # Total loss
                loss = sum(loss_dict.values())

                # Scale loss for gradient accumulation
                loss = loss / self.accumulate_grad_batches

            # Backward pass
            self.scaler.scale(loss).backward()

            # Gradient accumulation step
            if (batch_idx + 1) % self.accumulate_grad_batches == 0 or (batch_idx + 1) == len(self.train_loader):
                # Unscale gradients for clipping
                self.scaler.unscale_(self.optimizer)

                # Gradient clipping
                if self.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.clip_max_norm
                    )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Zero gradients
                self.optimizer.zero_grad(set_to_none=True)

                self.global_step += 1

            # Update loss meters
            batch_size = images.size(0)
            loss_meters['loss_total'].update(loss.item() * self.accumulate_grad_batches, batch_size)

            for loss_name, loss_value in loss_dict.items():
                if loss_name in loss_meters:
                    loss_meters[loss_name].update(loss_value.item(), batch_size)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_meters['loss_total'].avg:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

            # Log to tensorboard
            if self.global_step % self.log_interval == 0:
                for loss_name, meter in loss_meters.items():
                    self.logger.log_scalar(
                        f'train/{loss_name}',
                        meter.val,
                        self.global_step
                    )

                self.logger.log_scalar(
                    'train/learning_rate',
                    self.optimizer.param_groups[0]['lr'],
                    self.global_step
                )

        # Epoch summary
        epoch_time = epoch_timer.stop()

        avg_losses = {name: meter.avg for name, meter in loss_meters.items()}

        self.logger.info(
            f"Epoch {epoch} completed in {epoch_time:.1f}s | "
            f"Loss: {avg_losses['loss_total']:.4f} | "
            f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
        )

        return avg_losses

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate model on validation set.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of validation metrics
        """
        self.logger.info(f"Running validation for epoch {epoch}...")

        metrics = evaluate_model(
            self.model,
            self.val_loader,
            self.device,
            conf_threshold=0.01,
            nms_threshold=0.7,
            max_detections=300,
            label_to_cat_id=None  # Will use dataset's mapping
        )

        # Log metrics
        self.logger.log_metrics(metrics, epoch, prefix='val')

        return metrics

    def train(self, resume_from: Optional[str] = None) -> None:
        """Main training loop.

        Args:
            resume_from: Path to checkpoint to resume from
        """
        start_epoch = 0

        # Resume from checkpoint if specified
        if resume_from is not None:
            checkpoint = self.checkpoint_manager.load_checkpoint(
                resume_from,
                self.model,
                self.optimizer,
                self.scheduler,
                self.device
            )
            start_epoch = checkpoint['epoch'] + 1
            self.global_step = checkpoint.get('global_step', 0)
            self.logger.info(f"Resumed training from epoch {start_epoch}")

        # Training loop
        for epoch in range(start_epoch, self.epochs):
            self.current_epoch = epoch

            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            self.logger.info(f"{'='*60}")

            # Train for one epoch
            train_losses = self.train_epoch(epoch)

            # Step scheduler
            self.scheduler.step()

            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.log_lr(current_lr, epoch)

            # Validation
            should_evaluate = (
                (epoch + 1) % self.eval_interval == 0 or
                (epoch + 1) == self.epochs or
                self.overfit_batches > 0  # Always evaluate in overfit mode
            )

            if should_evaluate:
                val_metrics = self.validate(epoch)
            else:
                val_metrics = {}

            # Save checkpoint
            should_save = (
                (epoch + 1) % self.save_interval == 0 or
                (epoch + 1) == self.epochs
            )

            if should_save:
                # Prepare metrics for checkpoint
                checkpoint_metrics = {
                    **train_losses,
                    **val_metrics
                }

                extra_state = {
                    'global_step': self.global_step,
                    'config': dict(self.config)
                }

                self.checkpoint_manager.save_checkpoint(
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    metrics=checkpoint_metrics,
                    extra_state=extra_state
                )

            # Track best metric
            if val_metrics and 'AP' in val_metrics:
                if val_metrics['AP'] > self.best_metric:
                    self.best_metric = val_metrics['AP']
                    self.logger.info(f"New best AP: {self.best_metric:.4f}")

        self.logger.info(f"\n{'='*60}")
        self.logger.info("Training completed!")
        self.logger.info(f"Best AP: {self.best_metric:.4f}")
        self.logger.info(f"{'='*60}")

        # Close logger
        self.logger.close()


def build_trainer(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    config: Any,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader
) -> Trainer:
    """Build trainer from components.

    Args:
        model: Model to train
        criterion: Loss criterion
        optimizer: Optimizer
        scheduler: LR scheduler
        config: Configuration
        device: Device
        train_loader: Training dataloader
        val_loader: Validation dataloader

    Returns:
        Configured trainer
    """
    # Create logger
    logger = TrainingLogger(
        log_dir=config.logging.log_dir,
        use_tensorboard=config.logging.get('tensorboard', True)
    )

    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        save_dir=config.checkpoint.save_dir,
        save_best=config.checkpoint.get('save_best', True),
        save_last=config.checkpoint.get('save_last', True),
        monitor=config.checkpoint.get('monitor', 'AP'),
        mode=config.checkpoint.get('mode', 'max')
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
        checkpoint_manager=checkpoint_manager
    )

    return trainer
