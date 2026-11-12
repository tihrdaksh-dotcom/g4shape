"""Small model utilities."""

import torch.nn as nn

__all__ = ["count_parameters"]


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
