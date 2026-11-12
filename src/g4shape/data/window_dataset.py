"""PyG dataset over the G4-seq row index: each sample is a genome window, featurized on demand.

``get(i)``: index row -> genome window (``flank + 15 + flank`` nt) -> ``featurize``
(``src.data.featurize``: G-run graph + topology vector) -> ``Data`` + target and metadata.
Nothing but the genome codes and the 10-byte index rows lives on disk, the work happens
inside DataLoader workers, and the per-sample memory is the sample itself.

The returned ``Data`` carries the ``featurize`` fields (``x``, ``edge_index``,
``edge_attr``, ``global_features`` = topology, ``had_graph``) plus ``y`` -- the float
mismatch score (regression), not a class id -- and ``chrom``, ``start``, ``strand``,
``split``; on a weighted index split also ``sample_weight`` (the loss weight of the row).

Strand: rows from the ``+`` bedGraph are reverse-complemented, as in TransG4's
``MissGen.read_seq`` (``seq[::-1, ::-1]``); set ``reverse_strand=None`` to disable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset as PyGDataset

from g4shape.data.featurize import feature_dims, featurize
from g4shape.data.genome_store import GenomeStore
from g4shape.data.g4seq_index import STRAND_NAME, STRAND_SIGN, G4SeqIndex
from g4shape.features.config import FeatureConfig

__all__ = ["G4WindowDataset", "subsample_rows", "load_window_bundle"]


def subsample_rows(n_rows: int, n_keep: int, seed: int = 0) -> np.ndarray:
    """Sorted random subset of ``n_keep`` row ids without replacement (O(n_keep) memory)."""
    n_keep = min(n_keep, n_rows)
    return np.sort(np.random.default_rng(seed).choice(n_rows, size=n_keep, replace=False))


class G4WindowDataset(PyGDataset):
    def __init__(
        self,
        genome_dir: str | Path,
        index_dir: str | Path,
        split: str,
        flank: int = 100,
        feature_config: Optional[FeatureConfig] = None,
        rows: Optional[np.ndarray] = None,
        reverse_strand: Optional[str] = "+",
        use_topology: bool = True,
        transform=None,
    ):
        super().__init__(root=None, transform=transform)
        self.genome = GenomeStore(genome_dir)
        self.index = G4SeqIndex(index_dir, split)
        self.split = split
        self.flank = flank
        self.cfg = feature_config or FeatureConfig()
        self.rows = None if rows is None else np.asarray(rows, dtype=np.int64)
        self._reverse_sign = None if reverse_strand is None else STRAND_SIGN[reverse_strand]
        self.use_topology = use_topology
        self.dims = feature_dims(self.cfg)
        self.node_feature_dim = self.dims["node_dim"]
        self.edge_feature_dim = self.dims["edge_dim"]
        self.topology_dim = self.dims["global_dim"]

    def len(self) -> int:
        return len(self.index) if self.rows is None else int(self.rows.size)

    def get(self, idx: int) -> Data:
        row = idx if self.rows is None else int(self.rows[idx])
        chrom, start, strand, score = self.index.row(row)
        w = self.index.window
        codes = self.genome.window(chrom, start, start + w, self.flank,
                                   reverse=strand == self._reverse_sign)
        data = featurize(codes, self.cfg, self.use_topology)
        data.y = torch.tensor([score], dtype=torch.float32)
        data.chrom = self.genome.names[chrom]
        data.start = start
        data.strand = STRAND_NAME[strand]
        data.split = self.split
        if self.index.weighted:
            data.sample_weight = torch.tensor([self.index.weight(row)], dtype=torch.float32)
        return data


def load_window_bundle(
    genome_dir: str | Path,
    index_dir: str | Path,
    flank: int = 100,
    feature_config: Optional[FeatureConfig] = None,
    max_train: Optional[int] = None,
    max_val: Optional[int] = None,
    max_test: Optional[int] = None,
    seed: int = 0,
    **kwargs: Any,
) -> dict[str, Any]:
    """train / val / test datasets + ``feature_dims`` (``node_dim``, ``edge_dim``,
    ``global_dim``, ``global_layout``; pass them to ``src.lightning.build.build_lit_model``).

    ``max_train`` / ``max_val`` / ``max_test`` cap the number of rows of a split with a seeded
    random subset, the practical way to bound an epoch on genome-scale data (keep
    ``max_test=None`` to report on the whole test chromosome, as TransG4 does).
    """
    def make(split: str, cap: Optional[int]):
        n = len(G4SeqIndex(index_dir, split))
        rows = subsample_rows(n, cap, seed) if cap is not None and cap < n else None
        return G4WindowDataset(genome_dir, index_dir, split, flank, feature_config, rows, **kwargs)

    train = make("train", max_train)
    return {
        **train.dims,
        "train_dataset": train,
        "val_dataset": make("val", max_val),
        "test_dataset": make("test", max_test),
    }
