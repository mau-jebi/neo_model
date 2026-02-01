"""Learning rate scheduler with warmup for neo_model training.

Implements multi-step LR decay with linear warmup at the beginning of training.
"""

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR, _LRScheduler
from typing import List, Any


class WarmupMultiStepLR(_LRScheduler):
    """Multi-step learning rate scheduler with warmup.

    Applies linear warmup for the first N epochs, then multi-step decay.

    Args:
        optimizer: Optimizer to schedule
        milestones: List of epoch indices for LR decay
        gamma: Multiplicative factor for LR decay
        warmup_epochs: Number of warmup epochs
        warmup_factor: Initial LR = base_lr * warmup_factor
        last_epoch: Last epoch index (for resuming)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        warmup_epochs: int = 5,
        warmup_factor: float = 0.001,
        last_epoch: int = -1
    ):
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha

            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Multi-step decay
            decay_factor = 1.0
            for milestone in self.milestones:
                if self.last_epoch >= milestone:
                    decay_factor *= self.gamma

            return [base_lr * decay_factor for base_lr in self.base_lrs]


class WarmupScheduler:
    """Wrapper for warmup with any scheduler.

    Applies linear warmup, then switches to the provided scheduler.

    Args:
        optimizer: Optimizer to schedule
        scheduler: Scheduler to use after warmup
        warmup_epochs: Number of warmup epochs
        warmup_factor: Initial LR = base_lr * warmup_factor
    """

    def __init__(
        self,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        warmup_epochs: int = 5,
        warmup_factor: float = 0.001
    ):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self.current_epoch = 0

        # Store base LRs
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, epoch: int = None):
        """Step the scheduler.

        Args:
            epoch: Current epoch (optional)
        """
        if epoch is not None:
            self.current_epoch = epoch

        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = (self.current_epoch + 1) / self.warmup_epochs
            warmup_lr_mult = self.warmup_factor * (1 - alpha) + alpha

            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * warmup_lr_mult
        else:
            # Use provided scheduler
            self.scheduler.step()

        self.current_epoch += 1

    def get_last_lr(self):
        """Get last computed learning rate."""
        if self.current_epoch <= self.warmup_epochs:
            return [group['lr'] for group in self.optimizer.param_groups]
        else:
            return self.scheduler.get_last_lr()

    def state_dict(self):
        """Get scheduler state."""
        return {
            'scheduler': self.scheduler.state_dict(),
            'current_epoch': self.current_epoch,
            'warmup_epochs': self.warmup_epochs,
            'warmup_factor': self.warmup_factor,
            'base_lrs': self.base_lrs
        }

    def load_state_dict(self, state_dict):
        """Load scheduler state."""
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.current_epoch = state_dict['current_epoch']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.warmup_factor = state_dict['warmup_factor']
        self.base_lrs = state_dict['base_lrs']


def build_scheduler(
    config: Any,
    optimizer: Optimizer,
    num_training_steps: int = None
) -> _LRScheduler:
    """Build learning rate scheduler from configuration.

    Args:
        config: Configuration object
        optimizer: Optimizer to schedule
        num_training_steps: Total number of training steps (optional, for step-based schedulers)

    Returns:
        Configured scheduler
    """
    sched_config = config.training.lr_scheduler
    warmup_epochs = config.training.get('warmup_epochs', 0)

    sched_type = sched_config.get('type', 'MultiStepLR')

    if sched_type == 'MultiStepLR':
        milestones = sched_config.get('milestones', [40, 55])
        gamma = sched_config.get('gamma', 0.1)

        if warmup_epochs > 0:
            # Use integrated warmup scheduler
            scheduler = WarmupMultiStepLR(
                optimizer,
                milestones=milestones,
                gamma=gamma,
                warmup_epochs=warmup_epochs,
                warmup_factor=0.001
            )
        else:
            scheduler = MultiStepLR(
                optimizer,
                milestones=milestones,
                gamma=gamma
            )

    elif sched_type == 'CosineAnnealingLR':
        T_max = config.training.epochs - warmup_epochs
        eta_min = sched_config.get('eta_min', 0.0)

        base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min
        )

        if warmup_epochs > 0:
            scheduler = WarmupScheduler(
                optimizer,
                base_scheduler,
                warmup_epochs=warmup_epochs
            )
        else:
            scheduler = base_scheduler

    elif sched_type == 'StepLR':
        step_size = sched_config.get('step_size', 30)
        gamma = sched_config.get('gamma', 0.1)

        base_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )

        if warmup_epochs > 0:
            scheduler = WarmupScheduler(
                optimizer,
                base_scheduler,
                warmup_epochs=warmup_epochs
            )
        else:
            scheduler = base_scheduler

    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")

    return scheduler
