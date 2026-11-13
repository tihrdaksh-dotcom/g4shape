"""One sequence -> one model input: the G-run graph + topology vector as a PyG ``Data``.

This is the only place a sequence becomes model input. ``G4WindowDataset`` (genome
windows of the G4-seq index) and ``SequenceDataset`` (any sequences, e.g. a FASTA file)
both call ``featurize`` and only add their own target / metadata, so a window scored
through either path gets exactly the same features.

``Data`` fields: ``x`` [N, node_dim], ``edge_index`` [2, E], ``edge_attr`` [E, edge_dim],
``global_features`` [1, global_dim] (the topology vector), ``had_graph`` (bool).

Empty graph: a sequence without any qualifying G-run gets one all-zero node and no edges,
so it still occupies exactly one row of the batch (``had_graph=False`` marks it). With
``anchor_node`` that node is the target-bin anchor instead.

Optional graph extras (signed position, loop-length buckets, anchor node, one-hot context;
``FeatureConfig`` flags, all off by default) come from ``src.features.graph_extras``.

Feature scale depends on the sequence length (the topology counts, the edge gaps), so a
model trained on 215-nt windows should be scored on windows of that length.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from torch_geometric.data import Data

from g4shape.features.config import FeatureConfig
from g4shape.features.graph_extras import (extract_graph_full, graph_edge_feature_names,
                                       graph_node_feature_names)
from g4shape.features.sequence import encode
from g4shape.features.topology import extract_topology, topology_dim, topology_layout

__all__ = ["feature_dims", "featurize", "featurize_sequence"]


def feature_dims(cfg: Optional[FeatureConfig] = None) -> dict[str, Any]:
    """Model input sizes for ``cfg``: ``node_dim``, ``edge_dim``, ``global_dim`` and
    ``global_layout`` -- the arguments of ``build_lit_model`` / ``build_components`` that
    come from the data."""
    cfg = cfg or FeatureConfig()
    dims = {"node_dim": len(graph_node_feature_names(cfg)),
            "edge_dim": len(graph_edge_feature_names(cfg))}
    dims["global_dim"] = topology_dim(cfg)
    dims["global_layout"] = topology_layout(cfg)
    return dims


def featurize(codes: np.ndarray, cfg: Optional[FeatureConfig] = None,
              use_topology: bool = True) -> Data:
    """uint8 nucleotide codes (``src.features.sequence``) -> ``Data``.

    ``use_topology=False`` gives an all-zero topology vector (graph-only ablation).
    """
    cfg = cfg or FeatureConfig()
    g = extract_graph_full(codes, cfg)  # = extract_graph unless a graph extra is enabled
    had_graph = g.get("n_runs", g["n_nodes"]) > 0
    if g["n_nodes"] > 0:
        x, ei, ea = g["node_features"], g["edge_index"], g["edge_features"]
    else:
        x = np.zeros((1, len(graph_node_feature_names(cfg))), np.float32)
        ei = np.zeros((2, 0), np.int64)
        ea = np.zeros((0, len(graph_edge_feature_names(cfg))), np.float32)
    glob = (extract_topology(codes, cfg) if use_topology
            else np.zeros(topology_dim(cfg), np.float32))
    data = Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(ei),
                edge_attr=torch.from_numpy(ea))
    data.global_features = torch.from_numpy(glob).reshape(1, -1)
    data.had_graph = had_graph
    return data


def featurize_sequence(seq: str, cfg: Optional[FeatureConfig] = None,
                       use_topology: bool = True) -> Data:
    """Same as ``featurize`` for a sequence string (ACGT, case-insensitive; U -> T; other -> N)."""
    return featurize(encode(seq), cfg, use_topology)
