"""Miscellaneous utility functions for neo_model training.

Provides helper functions for device management, random seeding, timing, and other utilities.
"""

import random
import numpy as np
import torch
import torch.distributed as dist
from typing import Optional, List, Dict, Any
import time
from pathlib import Path


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device(device: Optional[str] = None, gpu_id: int = 0) -> torch.device:
    """Get PyTorch device.

    Args:
        device: Device string ('cuda', 'cpu', or None for auto-detect)
        gpu_id: GPU ID to use if multiple GPUs available

    Returns:
        PyTorch device
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda' and torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    else:
        return torch.device('cpu')


def get_world_size() -> int:
    """Get world size for distributed training.

    Returns:
        World size (1 if not distributed)
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Get rank for distributed training.

    Returns:
        Rank (0 if not distributed)
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    """Check if current process is main process.

    Returns:
        True if main process (rank 0)
    """
    return get_rank() == 0


class Timer:
    """Simple timer for measuring execution time."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.elapsed: float = 0.0

    def start(self) -> None:
        """Start timer."""
        self.start_time = time.time()

    def stop(self) -> float:
        """Stop timer and return elapsed time.

        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            return 0.0

        self.elapsed = time.time() - self.start_time
        self.start_time = None
        return self.elapsed

    def reset(self) -> None:
        """Reset timer."""
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class AverageMeter:
    """Compute and store average and current value.

    Useful for tracking metrics during training.
    """

    def __init__(self, name: str = ''):
        self.name = name
        self.reset()

    def reset(self) -> None:
        """Reset all statistics."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Update statistics.

        Args:
            val: New value
            n: Number of samples this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f} (current: {self.val:.4f})"


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Default collate function for batching samples.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary
    """
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        values = [sample[key] for sample in batch]

        if isinstance(values[0], torch.Tensor):
            # Stack tensors if they have the same shape
            if all(v.shape == values[0].shape for v in values):
                collated[key] = torch.stack(values, dim=0)
            else:
                # Keep as list if shapes differ (e.g., variable number of boxes)
                collated[key] = values
        elif isinstance(values[0], (int, float)):
            collated[key] = torch.tensor(values)
        else:
            # Keep as list for other types
            collated[key] = values

    return collated


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count number of parameters in model.

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_model_size(model: torch.nn.Module) -> float:
    """Get model size in MB.

    Args:
        model: PyTorch model

    Returns:
        Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb


def save_checkpoint(
    state: Dict[str, Any],
    filepath: Path,
    is_best: bool = False
) -> None:
    """Save checkpoint to file.

    Args:
        state: State dictionary to save
        filepath: Path to save checkpoint
        is_best: If True, also save as best checkpoint
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    torch.save(state, filepath)

    if is_best:
        best_path = filepath.parent / 'best_checkpoint.pth'
        torch.save(state, best_path)


def load_checkpoint(
    filepath: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Load checkpoint from file.

    Args:
        filepath: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load checkpoint to

    Returns:
        Dictionary with checkpoint metadata (epoch, metrics, etc.)
    """
    if device is None:
        device = torch.device('cpu')

    checkpoint = torch.load(filepath, map_location=device)

    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])

    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint


def mkdir_if_missing(directory: Path) -> None:
    """Create directory if it doesn't exist.

    Args:
        directory: Directory path to create
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
