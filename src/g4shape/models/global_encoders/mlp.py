import torch
import torch.nn as nn

from src.models.stages import GlobalEncoder


class MLPEncoder(GlobalEncoder):
    """Flat-vector baseline: the topology vector as an unstructured input (registry ``mlp``)."""

    def __init__(self, global_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__(out_dim=hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(global_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, return_details: bool = False):
        out = self.net(x)
        return (out, {}) if return_details else out
