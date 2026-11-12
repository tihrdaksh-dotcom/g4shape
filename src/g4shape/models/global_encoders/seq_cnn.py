"""A 1D-CNN branch over the raw one-hot window, as a drop-in global encoder.

Registry name ``seq_cnn``. It replaces the topology branch: instead of the topology vector,
the data path (``src.data.featurize`` with ``features.sequence_branch=true``) puts the raw
window into ``data.global_features`` as a flat ``[B, seq_channels * seq_len]`` one-hot tensor.
This encoder reshapes it to ``[B, seq_channels, seq_len]`` and applies a stack of
Conv1d -> BatchNorm -> ReLU -> MaxPool blocks, then global average pooling and a linear map
to ``hidden_dim`` so it satisfies the ``GlobalEncoder`` contract (``out_dim == hidden_dim``,
matched against ``fusion.global_dim`` at construction).

``seq_len`` and ``seq_channels`` arrive automatically from ``feature_dims(...)['global_layout']``
via ``build_components`` (it forwards layout keys an encoder accepts). If ``seq_len`` is not
supplied it is inferred from ``global_dim // seq_channels``. Adaptive pooling means the forward
pass is robust to a window length that differs from the one declared at build time; a clear
error is raised only if the flat width is not divisible by ``seq_channels``.
"""

import torch
import torch.nn as nn

from g4shape.models.stages import GlobalEncoder


class SeqCNNEncoder(GlobalEncoder):
    def __init__(self, global_dim: int, hidden_dim: int = 64, dropout: float = 0.2,
                 seq_len: int | None = None, seq_channels: int = 5,
                 conv_channels: int = 64, kernel_size: int = 9, n_layers: int = 3):
        super().__init__(out_dim=hidden_dim)
        self.seq_channels = int(seq_channels)
        if seq_len is None:
            if global_dim % self.seq_channels:
                raise ValueError(
                    f"SeqCNNEncoder: global_dim={global_dim} is not divisible by "
                    f"seq_channels={self.seq_channels}; pass seq_len explicitly or fix "
                    f"features.seq_window / features.sequence_branch.")
            seq_len = global_dim // self.seq_channels
        self.seq_len = int(seq_len)

        pad = kernel_size // 2
        blocks, c_in = [], self.seq_channels
        for _ in range(n_layers):
            blocks += [nn.Conv1d(c_in, conv_channels, kernel_size, padding=pad),
                       nn.BatchNorm1d(conv_channels), nn.ReLU(), nn.MaxPool1d(2)]
            c_in = conv_channels
        self.conv = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Dropout(p=dropout),
                                  nn.Linear(conv_channels, hidden_dim), nn.ReLU())

    def forward(self, x: torch.Tensor, return_details: bool = False):
        # x: [B, seq_channels * seq_len] flat one-hot -> [B, seq_channels, seq_len]
        b = x.shape[0]
        x = x.view(b, self.seq_channels, self.seq_len)
        h = self.pool(self.conv(x)).squeeze(-1)   # [B, conv_channels]
        out = self.head(h)
        return (out, {}) if return_details else out
