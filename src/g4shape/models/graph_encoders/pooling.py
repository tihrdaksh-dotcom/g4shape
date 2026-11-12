from typing import Callable, Optional, Union

import torch
import torch.nn as nn

from torch_geometric.nn import global_add_pool, global_mean_pool


class MeanPool(nn.Module):
    def forward(self, x: torch.Tensor, batch: torch.Tensor, size: Optional[int] = None) -> torch.Tensor:
        return global_mean_pool(x, batch, size)


class AddPool(nn.Module):
    def forward(self, x: torch.Tensor, batch: torch.Tensor, size: Optional[int] = None) -> torch.Tensor:
        return global_add_pool(x, batch, size)


class WeightAndSumPool(nn.Module):
    """
    PyG equivalent of DGL ``WeightAndSum`` (used by DADGNN):
        w_n = sigmoid(Linear(x_n));  graph = sum_n w_n * x_n
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.gate_nn = nn.Linear(in_dim, 1, bias=True)

    def weights(self, x: torch.Tensor) -> torch.Tensor:
        """Per-node readout weights, [num_nodes, 1]."""
        return torch.sigmoid(self.gate_nn(x))

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        size: Optional[int] = None,
        return_weights: bool = False,
    ):
        w = self.weights(x)
        out = global_add_pool(w * x, batch, size)
        return (out, w) if return_weights else out


class _CallablePool(nn.Module):
    """Wraps a plain function ``f(x, batch[, size=])`` (e.g. a PyG global_*_pool)."""

    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, x, batch, size: Optional[int] = None):
        return self.fn(x, batch) if size is None else self.fn(x, batch, size=size)


POOL_REGISTRY = {
    "weight_and_sum": WeightAndSumPool,  # DADGNN readout
    "mean": MeanPool,
    "add": AddPool,
}


def build_pool(pool: Union[str, Callable, nn.Module], dim: int) -> nn.Module:
    if isinstance(pool, nn.Module):
        return pool
    if callable(pool):
        return _CallablePool(pool)
    key = str(pool).lower()
    if key not in POOL_REGISTRY:
        raise ValueError(f"Unknown pool '{pool}'. Available: {sorted(POOL_REGISTRY)}")
    return WeightAndSumPool(dim) if key == "weight_and_sum" else POOL_REGISTRY[key]()
