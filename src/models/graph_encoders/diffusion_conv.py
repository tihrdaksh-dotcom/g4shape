from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import scatter, softmax

ATTENTION_TYPES = ("gat", "gatv2")


class DiffusionAttentionConv(nn.Module):
    """
    One head of DADGNN ``SingleHeadGATLayer`` in PyG form (attention diffusion).

        h_0 = x
        a_t = edge_softmax_dst( leaky_relu( attn([h_t[src], h_t[dst], (edge_emb)]) ) )
        h_{t+1}[dst] = sum_{(src->dst)} dropout(a_t) * (h_t[src] (+ edge_emb))
        out = sum_{t=0..k} temp_t * h_t,      temp init: alpha(1-alpha)^t, temp_k=(1-alpha)^k

    Attention is recomputed from the current state at every hop (as in DADGNN).
    Differences from DADGNN, all off / neutral by default:
        edge_dim   : adds a projected edge embedding to attention input and message.
        project    : input Linear before diffusion (DADGNN applies none; its fc is unused).
        residual   : out += res(x). DADGNN's residual path is dead code, so default False.
        attention  : "gat" (default, DADGNN/GATv1) scores an edge with one linear map,
                     e = leaky_relu(a^T [h_src | h_dst | edge_emb]). It is additive, so the
                     destination term is the same for all of a node's in-edges and (up to the
                     leaky-ReLU kink) cancels in the softmax, and the edge term only adds a
                     fixed bonus per edge: the ranking of neighbours cannot depend on the
                     receiving run.
                     "gatv2" (Brody et al., ICLR 2022) applies the nonlinearity before the
                     attention vector,
                     e = a^T leaky_relu(W_s h_src + W_d h_dst + W_e edge_emb),
                     so source, destination and edge features interact and every run gets its
                     own neighbour ranking ("dynamic" attention). Messages are unchanged.
        attention_dim : width of the GATv2 scoring layer (default out_dim); ignored for "gat".
    ``in_dim == out_dim`` is required unless ``project=True``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: Optional[int] = None,
        k: int = 3,
        alpha: float = 0.1,
        dropout: float = 0.2,
        residual: bool = False,
        project: bool = False,
        negative_slope: float = 0.01,
        attention: str = "gat",
        attention_dim: Optional[int] = None,
    ):
        super().__init__()
        if not project and in_dim != out_dim:
            raise ValueError(f"project=False needs in_dim == out_dim (got {in_dim} vs {out_dim})")
        if attention not in ATTENTION_TYPES:
            raise ValueError(f"attention must be one of {ATTENTION_TYPES}, got {attention!r}")

        self.in_dim, self.out_dim = in_dim, out_dim
        self.k, self.alpha = k, alpha
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.attention = attention

        self.lin = nn.Linear(in_dim, out_dim, bias=True) if project else None

        if edge_dim is not None and edge_dim > 0:
            self.edge_lin = nn.Linear(edge_dim, out_dim, bias=True)
            attn_in_dim = 3 * out_dim
        else:
            self.edge_lin = None
            attn_in_dim = 2 * out_dim
        if attention == "gat":
            self.attn_nn = nn.Linear(attn_in_dim, 1, bias=True)
        else:  # gatv2: W_s, W_d, W_e -> leaky_relu -> a  (one bias, on the destination map)
            attn_dim = attention_dim or out_dim
            self.attn_nn = None
            self.attn_src = nn.Linear(out_dim, attn_dim, bias=False)
            self.attn_dst = nn.Linear(out_dim, attn_dim, bias=True)
            self.attn_edge = nn.Linear(out_dim, attn_dim, bias=False) if self.edge_lin is not None else None
            self.attn_vec = nn.Linear(attn_dim, 1, bias=False)

        self.temp = nn.Parameter(torch.empty(k + 1, dtype=torch.float32))

        if residual:
            self.res_lin = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        else:
            self.res_lin = None

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if self.attention == "gat":
            nn.init.xavier_normal_(self.attn_nn.weight, gain=gain)
        else:
            for lin in (self.attn_src, self.attn_dst, self.attn_edge, self.attn_vec):
                if lin is not None:
                    nn.init.xavier_normal_(lin.weight, gain=gain)
            nn.init.zeros_(self.attn_dst.bias)
        for lin in (self.lin, self.edge_lin):
            if lin is not None:
                nn.init.xavier_normal_(lin.weight, gain=gain)
        if isinstance(self.res_lin, nn.Linear):
            nn.init.xavier_normal_(self.res_lin.weight, gain=gain)

        with torch.no_grad():
            ks = torch.arange(self.k + 1, dtype=torch.float32)
            self.temp.copy_(self.alpha * (1.0 - self.alpha) ** ks)
            self.temp[-1] = (1.0 - self.alpha) ** self.k

    def _scores(self, h_src, h_dst, edge_emb):
        """Unnormalised attention logits [E, 1] of every edge."""
        if self.attention == "gat":
            parts = [h_src, h_dst] if edge_emb is None else [h_src, h_dst, edge_emb]
            return F.leaky_relu(self.attn_nn(torch.cat(parts, dim=-1)), negative_slope=self.negative_slope)
        z = self.attn_src(h_src) + self.attn_dst(h_dst)
        if edge_emb is not None:
            z = z + self.attn_edge(edge_emb)
        return self.attn_vec(F.leaky_relu(z, negative_slope=self.negative_slope))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        """
        x [N, in_dim], edge_index [2, E] (row 0 = src, row 1 = dst), edge_attr [E, edge_dim] | None.
        Returns out [N, out_dim]; with return_attention also the per-hop attention
        weights (pre-dropout, softmax over each destination's in-edges), shape [k, E].
        """
        h = self.lin(x) if self.lin is not None else x
        out = self.temp[0] * h

        edge_emb = None
        if self.edge_lin is not None:
            edge_emb = (
                self.edge_lin(edge_attr)
                if edge_attr is not None
                else h.new_zeros((edge_index.size(1), self.out_dim))
            )

        src, dst = edge_index[0], edge_index[1]
        num_nodes = h.size(0)
        attn_hops = []

        for i in range(self.k):
            h_src, h_dst = h[src], h[dst]
            msg = h_src if edge_emb is None else h_src + edge_emb

            e = self._scores(h_src, h_dst, edge_emb)
            a = softmax(e, dst, num_nodes=num_nodes)
            if return_attention:
                attn_hops.append(a.squeeze(-1))
            a = F.dropout(a, p=self.dropout, training=self.training)

            h = scatter(msg * a, dst, dim=0, dim_size=num_nodes, reduce="sum")
            out = out + self.temp[i + 1] * h

        if self.res_lin is not None:
            out = out + self.res_lin(x)

        if not return_attention:
            return out
        attn = torch.stack(attn_hops) if attn_hops else x.new_zeros((0, edge_index.size(1)))
        return out, attn
