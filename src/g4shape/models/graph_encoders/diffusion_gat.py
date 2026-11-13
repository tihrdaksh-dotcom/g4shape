from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import BatchNorm as GraphBatchNorm
from torch_geometric.utils import add_remaining_self_loops

from g4shape.models.graph_encoders.diffusion_conv import DiffusionAttentionConv
from g4shape.models.graph_encoders.pooling import WeightAndSumPool, build_pool
from g4shape.models.stages import GraphEncoder

_ACTIVATIONS = {"elu": F.elu, "relu": F.relu, "gelu": F.gelu, "none": None}


def _resolve_activation(act: Union[None, str, Callable]) -> Callable:
    if isinstance(act, str):
        if act.lower() not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation '{act}'. Available: {sorted(_ACTIVATIONS)}")
        act = _ACTIVATIONS[act.lower()]
    return nn.Identity() if act is None else act


class DiffusionGATLayer(nn.Module):
    """
    Multi-head DiffusionAttentionConv (DADGNN ``GATLayer``): heads share the input,
    differ in their attention/temp parameters, and are merged by 'mean' or 'cat'+Wo.
    Optional post-merge residual / BatchNorm / activation (all off by default, as in DADGNN).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: Optional[int] = None,
        k: int = 3,
        alpha: float = 0.1,
        num_heads: int = 4,
        merge: str = "mean",
        dropout: float = 0.2,
        residual: bool = False,
        batch_norm: bool = False,
        activation: Union[None, str, Callable] = None,
        project: Optional[bool] = None,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        assert merge in {"cat", "mean"}
        self.merge = merge
        if project is None:
            project = in_dim != out_dim

        self.heads = nn.ModuleList(
            [
                DiffusionAttentionConv(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    edge_dim=edge_dim,
                    k=k,
                    alpha=alpha,
                    dropout=dropout,
                    residual=False,  # residual applied after the head merge
                    project=project,
                    negative_slope=negative_slope,
                )
                for _ in range(num_heads)
            ]
        )

        self.merge_proj = nn.Linear(num_heads * out_dim, out_dim, bias=True) if merge == "cat" else None
        self.bn = GraphBatchNorm(out_dim) if batch_norm else None
        self.activation = _resolve_activation(activation)

        if residual:
            self.residual = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        else:
            self.residual = None

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if self.merge_proj is not None:
            nn.init.xavier_normal_(self.merge_proj.weight, gain=gain)
        if isinstance(self.residual, nn.Linear):
            nn.init.xavier_normal_(self.residual.weight, gain=gain)

    def diffusion_coefficients(self) -> torch.Tensor:
        """Learned diffusion coefficients, [num_heads, k+1]."""
        return torch.stack([h.temp for h in self.heads])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        """With return_attention, also returns attention [num_heads, k, E]."""
        res = [head(x, edge_index, edge_attr, return_attention=return_attention) for head in self.heads]
        outs = [r[0] for r in res] if return_attention else res

        if self.merge == "cat":
            out = self.merge_proj(torch.cat(outs, dim=-1))
        else:
            out = torch.mean(torch.stack(outs, dim=0), dim=0)

        if self.residual is not None:
            out = out + self.residual(x)

        if self.bn is not None and not (self.training and out.size(0) <= 1):
            out = self.bn(out)  # BatchNorm needs >1 node in training mode

        out = self.activation(out)

        if not return_attention:
            return out
        return out, torch.stack([r[1] for r in res])


class DiffusionGATEncoder(GraphEncoder):
    """
    DADGNN graph branch (``Model`` minus the text-specific parts), PyG version:

        dropout -> Linear -> ReLU -> dropout -> Linear      (input_mlp)
        num_layers x DiffusionGATLayer (attention diffusion, no inter-layer dropout/act/norm)
        WeightAndSum readout                                 (pool="weight_and_sum")

    Defaults reproduce that. Extras (edge features, self-loops, residual, BatchNorm,
    other pools) are opt-in flags; ``use_edge_attr=True`` is the one
    deliberate default extension (G4 graphs carry edge features; set False for exact DADGNN).

    ``forward(..., return_details=True)`` additionally returns a dict for later analysis:
        edge_index        [2, E']   edges used (after self-loops)
        attention         list[layer] of [heads, k, E']   per-hop softmax attention
        diffusion_coeffs  list[layer] of [heads, k+1]     learned temp
        node_embeddings   list of [N, hidden]   [after input MLP, after layer 1, ...]
        pool_weights      [N, 1] | None         readout gate (weight_and_sum only)
        batch             [N]
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
        pool: Union[str, Callable, nn.Module] = "weight_and_sum",
        input_mlp: bool = True,
        use_edge_attr: bool = True,
        add_self_loops: bool = True,
        negative_slope: float = 0.01,
        project: Optional[bool] = None,
        layer_dropout: float = 0.0,
        residual: bool = False,
        batch_norm: bool = False,
        activation: Union[None, str, Callable] = None,
    ):
        super().__init__(out_dim=hidden_dim)

        self.dropout = dropout
        self.layer_dropout = layer_dropout
        self.hidden_dim = hidden_dim
        self.add_self_loops = add_self_loops
        self.use_edge_attr = use_edge_attr and edge_dim is not None and edge_dim > 0

        if input_mlp:
            self.input_lin1 = nn.Linear(node_dim, hidden_dim, bias=True)
            self.input_lin2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
            layer_in = hidden_dim
        else:
            self.input_lin1 = self.input_lin2 = None
            layer_in = node_dim

        self.convs = nn.ModuleList(
            [
                DiffusionGATLayer(
                    in_dim=layer_in if i == 0 else hidden_dim,
                    out_dim=hidden_dim,
                    edge_dim=edge_dim if self.use_edge_attr else None,
                    k=k,
                    alpha=alpha,
                    num_heads=num_heads,
                    merge=merge,
                    dropout=dropout,
                    residual=residual,
                    batch_norm=batch_norm,
                    activation=activation,
                    project=project,
                    negative_slope=negative_slope,
                )
                for i in range(num_layers)
            ]
        )

        self.pool = build_pool(pool, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        num_graphs: Optional[int] = None,
        return_details: bool = False,
    ):
        """
        Returns the graph embedding [num_graphs, hidden_dim] (and details if requested).

        ``num_graphs`` keeps output rows aligned 1:1 with the batch when some records
        have no nodes (has_graph=False); omit it for the batch.max()+1 behaviour.
        """
        if x.size(0) == 0:
            # No nodes at all (lone graph-less record): zero embedding, nothing to pool.
            out = x.new_zeros((num_graphs if num_graphs is not None else 1, self.hidden_dim))
            return (out, {}) if return_details else out

        if not self.use_edge_attr:
            edge_attr = None

        if self.add_self_loops:
            edge_index, edge_attr = add_remaining_self_loops(
                edge_index, edge_attr, fill_value=0.0, num_nodes=x.size(0)
            )

        if self.input_lin1 is not None:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.input_lin1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.input_lin2(x)

        details = {"node_embeddings": [x], "attention": [], "diffusion_coeffs": []} if return_details else None

        for i, conv in enumerate(self.convs):
            if i > 0 and self.layer_dropout > 0:
                x = F.dropout(x, p=self.layer_dropout, training=self.training)
            if return_details:
                x, attn = conv(x, edge_index, edge_attr, return_attention=True)
                details["attention"].append(attn)
                details["diffusion_coeffs"].append(conv.diffusion_coefficients().detach())
                details["node_embeddings"].append(x)
            else:
                x = conv(x, edge_index, edge_attr)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        pool_weights = None
        if return_details and isinstance(self.pool, WeightAndSumPool):
            x_graph, pool_weights = self.pool(x, batch, num_graphs, return_weights=True)
        else:
            x_graph = self.pool(x, batch, num_graphs)

        if not return_details:
            return x_graph
        details.update(edge_index=edge_index, pool_weights=pool_weights, batch=batch)
        return x_graph, details
