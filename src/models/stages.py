"""The contract of each stage of ``MultiModalModel``.

The network is ``graph encoder -> \\
                  global encoder -> fusion -> head``; a class can fill a stage only by
subclassing that stage's base below, which fixes two things:

1. **Call signature** (``forward``), identical for every class of a stage, including
   ``return_details``: with ``return_details=True`` every stage returns
   ``(output, details_dict)`` (an empty dict if it has nothing to report), otherwise just
   ``output``. ``MultiModalModel`` can therefore call every stage the same way, with no
   per-class special cases and no signature inspection at run time.
2. **Declared widths** (``out_dim``, and ``graph_dim`` / ``global_dim`` / ``in_dim`` on
   the consumers), so ``MultiModalModel`` checks at construction that the stages fit
   together and reports a mismatch by name instead of failing inside a matmul.

``forward`` signatures (B = graphs in the batch, N = nodes, E = edges):

* ``GraphEncoder.forward(x [N, node_dim], edge_index [2, E], edge_attr [E, edge_dim] | None,
  batch [N], num_graphs: int | None = None, return_details=False) -> [B, out_dim]``;
  ``num_graphs`` keeps one output row per graph even when trailing graphs have no nodes.
* ``GlobalEncoder.forward(x [B, global_dim], return_details=False) -> [B, out_dim]``
* ``Fusion.forward(x_graph [B, graph_dim], x_global [B, global_dim], return_details=False)
  -> [B, out_dim]``; a ``"gate"`` entry in its details is surfaced as ``details["gate"]``.
* ``Head.forward(x [B, in_dim], return_details=False) -> [B, out_dim]`` (``out_dim`` = 1:
  a logit or a standardised score); a ``"penultimate"`` entry is surfaced likewise.
"""

from __future__ import annotations

import torch.nn as nn

__all__ = ["GraphEncoder", "GlobalEncoder", "Fusion", "Head"]


class GraphEncoder(nn.Module):
    """Node/edge features of a batch of graphs -> one vector per graph."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = int(out_dim)


class GlobalEncoder(nn.Module):
    """Per-sample global feature vector (the topology vector) -> embedding."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = int(out_dim)


class Fusion(nn.Module):
    """Graph embedding + global embedding -> one fused vector."""

    def __init__(self, graph_dim: int, global_dim: int, out_dim: int):
        super().__init__()
        self.graph_dim, self.global_dim, self.out_dim = int(graph_dim), int(global_dim), int(out_dim)


class Head(nn.Module):
    """Fused vector -> model output (``out_dim`` = 1)."""

    def __init__(self, in_dim: int, out_dim: int = 1):
        super().__init__()
        self.in_dim, self.out_dim = int(in_dim), int(out_dim)
