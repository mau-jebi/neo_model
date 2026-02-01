"""Logging utilities for training and evaluation.

Provides TensorBoard integration and console logging for neo_model training.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Union, Any
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class TrainingLogger:
    """Unified logger for training with TensorBoard and console output.

    Handles logging of metrics, losses, learning rates, and images to both
    TensorBoard and console for monitoring training progress.
    """

    def __init__(
        self,
        log_dir: Union[str, Path],
        use_tensorboard: bool = True,
        console_level: int = logging.INFO
    ):
        """Initialize training logger.

        Args:
            log_dir: Directory for log files
            use_tensorboard: Whether to enable TensorBoard logging
            console_level: Logging level for console output
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup console logger
        self.console_logger = self._setup_console_logger(console_level)

        # Setup TensorBoard
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.writer: Optional[SummaryWriter] = None

        if self.use_tensorboard:
            try:
                self.writer = SummaryWriter(log_dir=str(self.log_dir))
                self.console_logger.info(f"TensorBoard logging enabled at {self.log_dir}")
            except Exception as e:
                self.console_logger.warning(f"Failed to initialize TensorBoard: {e}")
                self.use_tensorboard = False
        elif use_tensorboard:
            self.console_logger.warning("TensorBoard requested but not available. Install with: pip install tensorboard")

    def _setup_console_logger(self, level: int) -> logging.Logger:
        """Setup console logger with formatting.

        Args:
            level: Logging level

        Returns:
            Configured logger
        """
        logger = logging.getLogger('neo_model')
        logger.setLevel(level)

        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Formatter
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

        # File handler
        log_file = self.log_dir / 'train.log'
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log scalar value.

        Args:
            tag: Name of the scalar
            value: Scalar value
            step: Global step
        """
        if self.use_tensorboard and self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int) -> None:
        """Log multiple scalars under same main tag.

        Args:
            main_tag: Parent tag for the scalars
            tag_scalar_dict: Dictionary of tag-value pairs
            step: Global step
        """
        if self.use_tensorboard and self.writer is not None:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_losses(self, loss_dict: Dict[str, float], epoch: int, step: int) -> None:
        """Log training losses.

        Args:
            loss_dict: Dictionary of loss components
            epoch: Current epoch
            step: Global step
        """
        # Log to TensorBoard
        for loss_name, loss_value in loss_dict.items():
            self.log_scalar(f'train/{loss_name}', loss_value, step)

        # Log to console (only total loss)
        if 'loss_total' in loss_dict:
            total_loss = loss_dict['loss_total']
            self.console_logger.info(
                f"Epoch {epoch} | Step {step} | Loss: {total_loss:.4f}"
            )

    def log_metrics(self, metric_dict: Dict[str, float], epoch: int, prefix: str = 'val') -> None:
        """Log evaluation metrics.

        Args:
            metric_dict: Dictionary of metrics
            epoch: Current epoch
            prefix: Prefix for metric names (e.g., 'val', 'test')
        """
        # Log to TensorBoard
        for metric_name, metric_value in metric_dict.items():
            self.log_scalar(f'{prefix}/{metric_name}', metric_value, epoch)

        # Log to console
        metric_str = ' | '.join([f"{k}: {v:.4f}" for k, v in metric_dict.items()])
        self.console_logger.info(f"Epoch {epoch} | {prefix.upper()} | {metric_str}")

    def log_lr(self, lr: float, epoch: int) -> None:
        """Log learning rate.

        Args:
            lr: Current learning rate
            epoch: Current epoch
        """
        self.log_scalar('train/learning_rate', lr, epoch)

    def log_image(self, tag: str, image: Any, step: int) -> None:
        """Log image to TensorBoard.

        Args:
            tag: Image tag
            image: Image tensor (C, H, W) or numpy array
            step: Global step
        """
        if self.use_tensorboard and self.writer is not None:
            self.writer.add_image(tag, image, step)

    def log_images(self, tag: str, images: Any, step: int) -> None:
        """Log batch of images to TensorBoard.

        Args:
            tag: Images tag
            images: Batch of images (N, C, H, W)
            step: Global step
        """
        if self.use_tensorboard and self.writer is not None:
            self.writer.add_images(tag, images, step)

    def info(self, message: str) -> None:
        """Log info message to console.

        Args:
            message: Message to log
        """
        self.console_logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message to console.

        Args:
            message: Message to log
        """
        self.console_logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message to console.

        Args:
            message: Message to log
        """
        self.console_logger.error(message)

    def debug(self, message: str) -> None:
        """Log debug message to console.

        Args:
            message: Message to log
        """
        self.console_logger.debug(message)

    def close(self) -> None:
        """Close logger and flush all buffers."""
        if self.use_tensorboard and self.writer is not None:
            self.writer.flush()
            self.writer.close()

        # Close file handlers
        for handler in self.console_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def setup_logger(
    name: str = 'neo_model',
    log_dir: Optional[Union[str, Path]] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Setup a simple console logger.

    Args:
        name: Logger name
        log_dir: Optional directory for log file
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f'{name}.log'

        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
