"""PyG dataset over arbitrary sequences (a list, or a FASTA file), for scoring with a
trained model outside the G4-seq genome index, e.g. TransG4's independent validation sets.

    ds = SequenceDataset.from_fasta("seqs.fa", feature_config=cfg)
    preds = trainer.predict(model, DataLoader(ds, batch_size=256))

Features come from ``featurize`` (identical to ``G4WindowDataset``); ``y`` is attached when
targets are given; ``name`` and ``seq_id`` (position in the input) are kept as metadata
(not ``index``: PyG offsets any attribute whose name contains "index" when batching). Like ``G4WindowDataset``,
features are computed on demand (inside DataLoader workers).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset as PyGDataset

from src.data.fasta import read_fasta
from src.data.featurize import feature_dims, featurize
from src.features.config import FeatureConfig
from src.features.sequence import encode, reverse_complement

__all__ = ["SequenceDataset"]


class SequenceDataset(PyGDataset):
    def __init__(
        self,
        sequences: Sequence[str],
        names: Optional[Sequence[str]] = None,
        y: Optional[Sequence[float]] = None,
        feature_config: Optional[FeatureConfig] = None,
        reverse_complement: bool = False,
        use_topology: bool = True,
        transform=None,
    ):
        super().__init__(root=None, transform=transform)
        self.sequences = list(sequences)
        n = len(self.sequences)
        self.names = list(names) if names is not None else [str(i) for i in range(n)]
        self.y = None if y is None else [float(v) for v in y]
        if len(self.names) != n or (self.y is not None and len(self.y) != n):
            raise ValueError("sequences, names and y must have the same length")
        self.cfg = feature_config or FeatureConfig()
        self.reverse_complement = reverse_complement
        self.use_topology = use_topology
        self.dims = feature_dims(self.cfg)

    @classmethod
    def from_fasta(cls, path: str | Path, **kwargs) -> "SequenceDataset":
        records = read_fasta(path)
        return cls([s for _, s in records], names=[n for n, _ in records], **kwargs)

    def len(self) -> int:
        return len(self.sequences)

    def get(self, idx: int) -> Data:
        codes = encode(self.sequences[idx])
        if self.reverse_complement:
            codes = reverse_complement(codes)
        data = featurize(codes, self.cfg, self.use_topology)
        if self.y is not None:
            data.y = torch.tensor([self.y[idx]], dtype=torch.float32)
        data.name = self.names[idx]
        data.seq_id = idx
        return data
