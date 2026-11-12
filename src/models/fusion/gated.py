import torch
import torch.nn as nn

from src.models.stages import Fusion


class GatedFusion(Fusion):
    """Per-dimension convex mix of the two branch embeddings:
    ``fused = gate * x_global + (1 - gate) * x_graph``, ``gate = sigmoid(MLP([x_graph, x_global]))``.

    * ``norm_inputs`` (default True): LayerNorm each branch before gating. The graph
      embedding is a weighted *sum* over nodes (its scale grows with the number of G-runs)
      while the topology embedding ends in a LayerNorm; without this the gate compares
      vectors of different scales.
    * ``modality_dropout`` (default 0): during training, with this probability per sample,
      replace one branch (chosen at random) by zeros, so neither branch can be ignored and
      the model stays usable when one branch is uninformative. Off at eval.
    * ``return_details=True`` also returns ``{"gate": [B, H], "x_graph_n", "x_global_n"}``;
      ``gate.mean(1)`` is the per-sample reliance on the topology branch (0 = graph only).
    """

    def __init__(self, hidden_dim: int, norm_inputs: bool = True, modality_dropout: float = 0.0):
        super().__init__(graph_dim=hidden_dim, global_dim=hidden_dim, out_dim=hidden_dim)
        self.norm_graph = nn.LayerNorm(hidden_dim) if norm_inputs else nn.Identity()
        self.norm_global = nn.LayerNorm(hidden_dim) if norm_inputs else nn.Identity()
        self.modality_dropout = modality_dropout
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # Outputs hidden_dim for element-wise gating
            nn.Sigmoid(),
        )

    def forward(self, x_graph: torch.Tensor, x_global: torch.Tensor, return_details: bool = False):
        x_graph, x_global = self.norm_graph(x_graph), self.norm_global(x_global)
        if self.training and self.modality_dropout > 0:
            b = x_graph.size(0)
            drop = torch.rand(b, 1, device=x_graph.device) < self.modality_dropout
            which = torch.rand(b, 1, device=x_graph.device) < 0.5  # True: drop graph
            x_graph = torch.where(drop & which, torch.zeros_like(x_graph), x_graph)
            x_global = torch.where(drop & ~which, torch.zeros_like(x_global), x_global)
        gate = self.gate_net(torch.cat([x_graph, x_global], dim=1))
        fused = gate * x_global + (1.0 - gate) * x_graph
        if return_details:
            return fused, {"gate": gate, "x_graph_n": x_graph, "x_global_n": x_global}
        return fused
