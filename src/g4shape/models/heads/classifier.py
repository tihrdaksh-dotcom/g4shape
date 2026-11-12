import torch
import torch.nn as nn

from g4shape.models.stages import Head


class BinaryMLPClassifier(Head):
    """``Linear(in→hidden) → ReLU → Dropout → Linear(hidden→1)``: one output per sample, a
    logit for classification or the (standardised) score for regression (registry name
    ``binary_mlp``).

    ``return_details=True`` also returns ``{"penultimate": [B, hidden]}``, the last
    representation before the output (post-ReLU, pre-dropout), useful for plotting.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__(in_dim=in_dim, out_dim=1)
        self.hidden = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, return_details: bool = False):
        h = self.hidden(x)
        y = self.out(self.dropout(h))  # [B, 1]
        return (y, {"penultimate": h}) if return_details else y
