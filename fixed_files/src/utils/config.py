"""Configuration management for neo_model training.

Handles loading, validation, and access to training configuration from YAML files.
Ensures 16:9 aspect ratio compliance and provides convenient attribute access.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
import os


class ConfigDict(dict):
    """Dictionary with attribute-style access for convenient config usage.

    Example:
        config = ConfigDict({'model': {'hidden_dim': 256}})
        print(config.model.hidden_dim)  # 256
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")


def load_config(config_path: Union[str, Path]) -> ConfigDict:
    """Load and validate configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        ConfigDict with validated configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    if config_dict is None:
        raise ValueError(f"Empty configuration file: {config_path}")

    # Convert to ConfigDict for attribute access
    config = ConfigDict(config_dict)

    # Validate configuration
    _validate_config(config)

    return config


def _validate_config(config: ConfigDict) -> None:
    """Validate configuration values and 16:9 aspect ratio compliance.

    Args:
        config: Configuration to validate

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate required top-level sections
    required_sections = ['model', 'input', 'data', 'training']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    # Validate 16:9 aspect ratio
    width = config.input.get('width')
    height = config.input.get('height')

    if width is None or height is None:
        raise ValueError("Input width and height must be specified")

    aspect_ratio = width / height
    expected_ratio = 16 / 9
    tolerance = 0.01

    if abs(aspect_ratio - expected_ratio) > tolerance:
        raise ValueError(
            f"Input resolution {width}x{height} is not 16:9 aspect ratio. "
            f"Got {aspect_ratio:.4f}, expected {expected_ratio:.4f}"
        )

    # Validate augmentation resolutions (if multi-scale training)
    if 'augmentation' in config and 'train' in config.augmentation:
        for aug_config in config.augmentation.train:
            if aug_config.get('type') == 'RandomResize':
                sizes = aug_config.get('sizes', [])
                for size in sizes:
                    if len(size) == 2:
                        w, h = size
                        ratio = w / h
                        if abs(ratio - expected_ratio) > tolerance:
                            raise ValueError(
                                f"Augmentation size {w}x{h} is not 16:9 aspect ratio. "
                                f"Got {ratio:.4f}, expected {expected_ratio:.4f}"
                            )

    # Validate training parameters
    if config.training.epochs <= 0:
        raise ValueError(f"Training epochs must be positive, got {config.training.epochs}")

    if config.training.optimizer.lr <= 0:
        raise ValueError(f"Learning rate must be positive, got {config.training.optimizer.lr}")

    if config.data.train.batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {config.data.train.batch_size}")

    # Validate model parameters
    if config.model.decoder.num_queries <= 0:
        raise ValueError(f"Number of queries must be positive, got {config.model.decoder.num_queries}")

    if config.data.num_classes <= 0:
        raise ValueError(f"Number of classes must be positive, got {config.data.num_classes}")


def save_config(config: Union[Dict, ConfigDict], save_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary to save
        save_path: Path where to save the configuration
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert ConfigDict back to regular dict for saving
    if isinstance(config, ConfigDict):
        config_dict = dict(config)
    else:
        config_dict = config

    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def merge_configs(base_config: ConfigDict, override_config: Dict) -> ConfigDict:
    """Merge override configuration into base configuration.

    Args:
        base_config: Base configuration
        override_config: Configuration values to override

    Returns:
        Merged configuration
    """
    def _merge_dict(base: Dict, override: Dict) -> Dict:
        """Recursively merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = _merge_dict(result[key], value)
            else:
                result[key] = value
        return result

    merged = _merge_dict(dict(base_config), override_config)
    return ConfigDict(merged)


def expand_config_paths(config: ConfigDict, base_dir: Optional[Path] = None) -> ConfigDict:
    """Expand relative paths in configuration to absolute paths.

    Args:
        config: Configuration with potentially relative paths
        base_dir: Base directory for resolving relative paths (default: config file directory)

    Returns:
        Configuration with expanded paths
    """
    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)

    # Expand data paths
    if 'data' in config:
        if 'train' in config.data:
            if 'root' in config.data.train:
                config.data.train.root = str(base_dir / config.data.train.root)
            if 'annotation' in config.data.train:
                config.data.train.annotation = str(base_dir / config.data.train.annotation)

        if 'val' in config.data:
            if 'root' in config.data.val:
                config.data.val.root = str(base_dir / config.data.val.root)
            if 'annotation' in config.data.val:
                config.data.val.annotation = str(base_dir / config.data.val.annotation)

    # Expand checkpoint paths
    if 'checkpoint' in config and 'save_dir' in config.checkpoint:
        config.checkpoint.save_dir = str(base_dir / config.checkpoint.save_dir)

    # Expand logging paths
    if 'logging' in config and 'log_dir' in config.logging:
        config.logging.log_dir = str(base_dir / config.logging.log_dir)

    # Expand pretrained weights path
    if 'training' in config:
        if 'pretrained_weights' in config.training and config.training.pretrained_weights:
            config.training.pretrained_weights = str(base_dir / config.training.pretrained_weights)
        if 'resume_from' in config.training and config.training.resume_from:
            config.training.resume_from = str(base_dir / config.training.resume_from)

    return config
