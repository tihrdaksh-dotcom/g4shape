"""Structure-aware encoder for the K-mer topology vector (src/features/topology.py).

The vector is not an unstructured bag of 280 numbers: it is ``[Betti curves of C clouds]
[eigmin curves of C clouds]``, each curve sampled on the same R filtration radii. This
encoder uses that layout:

1. **Fixed, parameter-free preprocessing** (scale-free, needs no train statistics):
   per cloud, ``n = betti(r=0)`` (number of k-mer occurrences), the *survival curve*
   ``betti(r) / n`` in [0, 1] (fraction of clusters left at scale r; its drops are the
   merge events, i.e. the gap distribution), and ``log1p(eigmin(r))``; plus the scalar
   ``log1p(n)``. Raw counts (up to ~200) and eigenvalues no longer dominate the input.
2. **One curve encoder shared by all clouds**: Conv1d over the radius axis (the two curves
   of a cloud are the 2 input channels; local kernels see how a curve changes between
   neighbouring scales), then a Linear over the flattened radii, so each radius keeps its
   own meaning (scale is absolute, not translation-invariant). Weight sharing: a G cloud
   and a GG cloud are read by the same filters.
3. **Cloud identity**: a learned per-cloud embedding is added, and ``log1p(n)`` is
   concatenated, so the shared encoder can still tell A from GG.
4. **Cross-cloud mixing**: the C cloud embeddings are concatenated (fixed cloud order)
   and an MLP maps them to ``hidden_dim``.

``num_radii`` (= ``FeatureConfig.max_step``) is required and should come from
``src.features.topology.topology_layout(cfg)`` (``load_window_bundle`` returns it as
``global_layout``), together with ``num_clouds``; then ``global_dim`` must equal
``2 * num_clouds * num_radii`` exactly. A wrong ``num_radii`` would otherwise reshape the
vector into scrambled curves without any error whenever the sizes happen to divide.
Without ``num_clouds`` the number of clouds is inferred from ``global_dim``. ``forward(x, return_details=True)``
also returns the per-cloud embeddings ``[B, C, cloud_dim]`` for analysis.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from g4shape.models.stages import GlobalEncoder

__all__ = ["TopologyCurveEncoder", "preprocess_topology"]


def preprocess_topology(x: torch.Tensor, num_radii: int):
    """``[B, 2*C*R]`` raw vector -> (curves ``[B, C, 2, R]``, log1p(count) ``[B, C, 1]``)."""
    b = x.size(0)
    v = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).view(b, 2, -1, num_radii)
    betti, eigmin = v[:, 0].clamp(min=0), v[:, 1].clamp(min=0)  # [B, C, R]
    n = betti[..., :1]  # betti at r = 0 is the number of points
    survival = betti / n.clamp(min=1.0)
    curves = torch.stack([survival, torch.log1p(eigmin)], dim=2)  # [B, C, 2, R]
    return curves, torch.log1p(n)


class TopologyCurveEncoder(GlobalEncoder):
    def __init__(
        self,
        global_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        num_radii: Optional[int] = None,
        conv_channels: int = 16,
        kernel_size: int = 3,
        cloud_dim: int = 32,
        num_clouds: Optional[int] = None,
    ):
        super().__init__(out_dim=hidden_dim)
        if num_radii is None:
            raise ValueError("TopologyCurveEncoder needs num_radii (= FeatureConfig.max_step); pass "
                             "topology_layout(cfg), e.g. src.lightning.build.build_lit_model(global_layout=bundle['global_layout'])")
        if global_dim % (2 * num_radii):
            raise ValueError(f"global_dim={global_dim} is not 2 * num_clouds * num_radii (num_radii={num_radii})")
        if num_clouds is not None and global_dim != 2 * num_clouds * num_radii:
            raise ValueError(f"global_dim={global_dim} != 2 * num_clouds({num_clouds}) * num_radii({num_radii})")
        self.num_radii = num_radii
        self.num_clouds = global_dim // (2 * num_radii)
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(2, conv_channels, kernel_size, padding=pad),
            nn.GELU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size, padding=pad),
            nn.GELU(),
        )
        self.curve_proj = nn.Linear(conv_channels * num_radii + 1, cloud_dim)  # + log1p(count)
        self.cloud_embed = nn.Parameter(torch.zeros(self.num_clouds, cloud_dim))
        nn.init.normal_(self.cloud_embed, std=0.02)
        self.cloud_norm = nn.LayerNorm(cloud_dim)
        self.mix = nn.Sequential(
            nn.Linear(self.num_clouds * cloud_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor, return_details: bool = False):
        b = x.size(0)
        curves, log_count = preprocess_topology(x, self.num_radii)  # [B,C,2,R], [B,C,1]
        h = self.conv(curves.reshape(b * self.num_clouds, 2, self.num_radii))
        h = h.reshape(b, self.num_clouds, -1)
        clouds = self.curve_proj(torch.cat([h, log_count], dim=-1)) + self.cloud_embed
        clouds = self.cloud_norm(F.gelu(clouds))  # [B, C, cloud_dim]
        out = self.mix(clouds.reshape(b, -1))
        return (out, {"clouds": clouds}) if return_details else out
