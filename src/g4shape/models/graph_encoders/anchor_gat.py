from typing import Callable, Optional, Union

import torch
import torch.nn as nn

from torch_geometric.nn import global_add_pool

from src.models.graph_encoders.diffusion_gat import DiffusionGATEncoder
from src.models.graph_encoders.pooling import WeightAndSumPool


class AnchorReadout(WeightAndSumPool):
    """
    Readout for graphs with a target-bin anchor node (``features.anchor_node``):

        pooled = sum over G-run nodes of sigmoid(Linear(h)) * h   (anchor excluded)
        anchor = h of the anchor node
        'both'   -> Linear([pooled | anchor]),  'anchor' -> anchor,  'pool' -> pooled

    ``anchor_mask`` [N] bool is set by ``AnchorDiffusionGATEncoder`` before each pass.
    Returned weights (details ``pool_weights``) are 0 on anchor nodes.
    """

    def __init__(self, in_dim: int, mode: str = "both"):
        super().__init__(in_dim)
        if mode not in {"both", "anchor", "pool"}:
            raise ValueError(f"readout must be 'both', 'anchor' or 'pool', got {mode!r}")
        self.mode = mode
        self.mix = nn.Linear(2 * in_dim, in_dim) if mode == "both" else None
        self.anchor_mask: Optional[torch.Tensor] = None

    def forward(self, x, batch, size: Optional[int] = None, return_weights: bool = False):
        mask = self.anchor_mask
        w = self.weights(x) * (~mask).unsqueeze(1).to(x.dtype)
        pooled = global_add_pool(w * x, batch, size)
        anchor = x.new_zeros(pooled.shape)
        anchor[batch[mask]] = x[mask]
        if self.mode == "both":
            out = self.mix(torch.cat([pooled, anchor], dim=-1))
        else:
            out = anchor if self.mode == "anchor" else pooled
        return (out, w) if return_weights else out


class AnchorDiffusionGATEncoder(DiffusionGATEncoder):
    """
    ``diffusion_gat`` for graphs built with ``features.anchor_node=true``: identical message
    passing (every argument as in ``DiffusionGATEncoder``), but the readout uses the
    target-bin anchor node (``AnchorReadout``; ``readout`` = 'both' | 'anchor' | 'pool').

    The anchor is the node whose last input column (``is_anchor``) is 1; exactly one per
    graph is required (a ValueError otherwise, e.g. when the features have no anchor).
    Details additionally hold ``anchor_mask`` [N].
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: Optional[int] = None,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        k: int = 3,
        alpha: float = 0.1,
        num_heads: int = 4,
        merge: str = "mean",
        num_layers: int = 2,
        readout: str = "both",
        input_mlp: bool = True,
        use_edge_attr: bool = True,
        add_self_loops: bool = True,
        negative_slope: float = 0.01,
        project: Optional[bool] = None,
        layer_dropout: float = 0.0,
        residual: bool = False,
        batch_norm: bool = False,
        activation: Union[None, str, Callable] = None,
        attention: str = "gat",
        attention_dim: Optional[int] = None,
    ):
        super().__init__(
            node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim, dropout=dropout, k=k,
            alpha=alpha, num_heads=num_heads, merge=merge, num_layers=num_layers,
            pool=AnchorReadout(hidden_dim, readout), input_mlp=input_mlp,
            use_edge_attr=use_edge_attr, add_self_loops=add_self_loops,
            negative_slope=negative_slope, project=project, layer_dropout=layer_dropout,
            residual=residual, batch_norm=batch_norm, activation=activation,
            attention=attention, attention_dim=attention_dim)

    def forward(self, x, edge_index, edge_attr=None, batch=None, num_graphs=None,
                return_details: bool = False):
        mask = x[:, -1] > 0.5
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        n_graphs = num_graphs if num_graphs is not None else int(batch.max()) + 1
        per_graph = torch.bincount(batch[mask], minlength=n_graphs)
        if per_graph.numel() != n_graphs or not bool((per_graph == 1).all()):
            raise ValueError("anchor_gat needs exactly one anchor node per graph "
                             "(set features.anchor_node=true)")
        self.pool.anchor_mask = mask
        try:
            out = super().forward(x, edge_index, edge_attr, batch, num_graphs, return_details)
        finally:
            self.pool.anchor_mask = None
        if return_details:
            out[1]["anchor_mask"] = mask
        return out
