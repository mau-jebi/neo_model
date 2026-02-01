"""Optimizer configuration for neo_model training.

Provides parameter grouping to exclude biases, norms, and embeddings from weight decay.
Uses fused AdamW for 10-15% speedup on CUDA.
"""

import torch
from torch import nn, optim
from typing import List, Dict, Any


def build_optimizer(config: Any, model: nn.Module) -> optim.Optimizer:
    """Build optimizer with parameter grouping.

    Separates parameters into two groups:
    1. Parameters with weight decay (weights of conv/linear layers)
    2. Parameters without weight decay (biases, norms, embeddings)

    Args:
        config: Configuration object
        model: Model to optimize

    Returns:
        Configured optimizer
    """
    opt_config = config.training.optimizer

    # Get parameters to exclude from weight decay
    no_decay = ['bias', 'norm', 'bn', 'ln', 'pos_embed', 'level_embed', 'query_embed']

    # Split parameters into groups
    params_with_decay = []
    params_without_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if parameter should have weight decay
        if any(nd in name.lower() for nd in no_decay):
            params_without_decay.append(param)
        else:
            params_with_decay.append(param)

    # Create parameter groups
    param_groups = [
        {
            'params': params_with_decay,
            'weight_decay': opt_config.get('weight_decay', 0.0001),
            'initial_lr': opt_config.lr
        },
        {
            'params': params_without_decay,
            'weight_decay': 0.0,
            'initial_lr': opt_config.lr
        }
    ]

    print(f"Optimizer: {len(params_with_decay)} params with weight decay, "
          f"{len(params_without_decay)} params without weight decay")

    # Build optimizer
    opt_type = opt_config.get('type', 'AdamW')

    if opt_type == 'AdamW':
        optimizer = optim.AdamW(
            param_groups,
            lr=opt_config.lr,
            betas=opt_config.get('betas', (0.9, 0.999)),
            eps=opt_config.get('eps', 1e-8),
            fused=torch.cuda.is_available()  # Fused kernel for 10-15% speedup
        )
    elif opt_type == 'Adam':
        optimizer = optim.Adam(
            param_groups,
            lr=opt_config.lr,
            betas=opt_config.get('betas', (0.9, 0.999)),
            eps=opt_config.get('eps', 1e-8),
            fused=torch.cuda.is_available()
        )
    elif opt_type == 'SGD':
        optimizer = optim.SGD(
            param_groups,
            lr=opt_config.lr,
            momentum=opt_config.get('momentum', 0.9),
            nesterov=opt_config.get('nesterov', True)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")

    return optimizer


def get_parameter_groups(model: nn.Module) -> Dict[str, List]:
    """Get parameter groups for logging/debugging.

    Args:
        model: Model to analyze

    Returns:
        Dictionary with parameter group information
    """
    no_decay = ['bias', 'norm', 'bn', 'ln', 'pos_embed', 'level_embed', 'query_embed']

    with_decay = []
    without_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if any(nd in name.lower() for nd in no_decay):
            without_decay.append((name, param.numel()))
        else:
            with_decay.append((name, param.numel()))

    return {
        'with_decay': with_decay,
        'without_decay': without_decay,
        'total_with_decay': sum(n for _, n in with_decay),
        'total_without_decay': sum(n for _, n in without_decay)
    }
