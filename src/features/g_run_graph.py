"""G-run graph of one sequence, computed from nucleotide codes.

Nodes = maximal G-runs of length >= ``min_run_len`` (default 2); edges = every run
pair whose gap (loop length) is <= ``edge_radius`` (default 50), in both directions.
Node features: length (scaled + one-hot), window-edge flags, 2 context bases each
side, A/C/G/T composition of the ``flank_window`` nt each side. Edge features:
direction, absolute gap in nt, is-next-run, loop A/C/G/T composition, one-hot length
class of the source and of the target run.

This deliberately departs from the project's earlier HDF5 feature pipeline (which used
min_run_len=1, radius 125, log/RBF gap, runs_between, loop first/last base);
tests/test_features_equivalence.py checks it against a brute-force reference.

A sequence without any qualifying run gives a graph with zero nodes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from src.features.config import FeatureConfig

__all__ = ["extract_graph", "find_g_runs", "node_feature_names", "edge_feature_names"]

_BASES = "ACGT"
_MAX_LEN_SCALE = 12
_G = 2


def _len_classes(cfg: FeatureConfig) -> List[int]:
    """Run lengths that get their own one-hot slot (min_run_len..4); the last slot is >= 5."""
    return list(range(cfg.min_run_len, 5))


def _len_onehot(length: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    cols = [length == k for k in _len_classes(cfg)] + [length >= 5]
    return np.stack(cols, axis=1)


def node_feature_names(cfg: FeatureConfig) -> List[str]:
    names = (["len_scaled"] + [f"len_is_{k}" for k in _len_classes(cfg)] + ["len_ge_5"]
             + ["touches_5p_end", "touches_3p_end"])
    c = cfg.context_bases
    for k in range(c, 0, -1):
        names += [f"up{k}_{b}" for b in _BASES]
    for k in range(1, c + 1):
        names += [f"down{k}_{b}" for b in _BASES]
    names += [f"up_flank_{b}" for b in _BASES] + ["up_flank_available"]
    names += [f"down_flank_{b}" for b in _BASES] + ["down_flank_available"]
    return names


def edge_feature_names(cfg: FeatureConfig) -> List[str]:
    lens = [f"len_is_{k}" for k in _len_classes(cfg)] + ["len_ge_5"]
    return (["direction", "gap", "is_next_run"]
            + [f"loop_{b}" for b in _BASES]
            + [f"src_{n}" for n in lens] + [f"dst_{n}" for n in lens])


def find_g_runs(codes: np.ndarray, min_len: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Maximal G-runs of length >= ``min_len`` as (starts, ends), ``end`` exclusive."""
    edge = np.zeros(codes.size + 2, dtype=np.int8)
    edge[1:-1] = codes == _G
    d = np.diff(edge)
    starts = np.flatnonzero(d == 1).astype(np.int64)
    ends = np.flatnonzero(d == -1).astype(np.int64)
    keep = (ends - starts) >= min_len
    return starts[keep], ends[keep]


def _run_pairs_within_radius(starts, ends, radius):
    """All run pairs i < j with gap = start_j - end_i <= radius (ordered by i, then j)."""
    n = starts.size
    empty = np.zeros(0, dtype=np.int64)
    if n < 2:
        return empty, empty, empty
    j_stop = np.searchsorted(starts, ends + radius, side="right")
    per_i = np.maximum(j_stop - (np.arange(n) + 1), 0)
    n_pairs = int(per_i.sum())
    if n_pairs == 0:
        return empty, empty, empty
    src = np.repeat(np.arange(n, dtype=np.int64), per_i)
    first = np.repeat(np.cumsum(per_i) - per_i, per_i)
    dst = src + 1 + (np.arange(n_pairs, dtype=np.int64) - first)
    return src, dst, starts[dst] - ends[src]


def _prefix_counts(codes: np.ndarray) -> np.ndarray:
    """cum[b, i] = number of base b in codes[:i]  (4 x (n + 1))."""
    onehot = (codes[None, :] == np.arange(4)[:, None]).astype(np.int32)
    cum = np.zeros((4, codes.size + 1), dtype=np.int32)
    np.cumsum(onehot, axis=1, out=cum[:, 1:])
    return cum


def _composition(cum, lo, hi):
    """A/C/G/T fractions of seq[lo:hi] per interval; denominator = interval length."""
    counts = (cum[:, hi] - cum[:, lo]).T.astype(np.float32)
    length = (hi - lo).astype(np.float32)[:, None]
    return np.divide(counts, length, out=np.zeros_like(counts), where=length > 0)


def extract_graph(codes: np.ndarray, cfg: FeatureConfig) -> Dict[str, Any]:
    """G-run graph of ``codes``.

    Returns node_features (N, d_node) f32, edge_index (2, E) i64 (both
    directions), edge_features (E, d_edge) f32, run_starts / run_ends (N,) i32,
    n_nodes, n_edges.
    """
    codes = np.asarray(codes)
    n = codes.size
    starts, ends = find_g_runs(codes, cfg.min_run_len)
    n_nodes = starts.size
    d_node, d_edge = len(node_feature_names(cfg)), len(edge_feature_names(cfg))

    if n_nodes == 0:
        return {
            "node_features": np.zeros((0, d_node), np.float32),
            "edge_index": np.zeros((2, 0), np.int64),
            "edge_features": np.zeros((0, d_edge), np.float32),
            "run_starts": np.zeros(0, np.int32),
            "run_ends": np.zeros(0, np.int32),
            "n_nodes": 0,
            "n_edges": 0,
        }

    cum = _prefix_counts(codes)

    # ---- nodes ----
    length = ends - starts
    blocks = [
        (np.minimum(length, _MAX_LEN_SCALE) / _MAX_LEN_SCALE)[:, None],
        _len_onehot(length, cfg),
        (starts == 0)[:, None],
        (ends == n)[:, None],
    ]
    c = cfg.context_bases
    for pos in [starts - k for k in range(c, 0, -1)] + [ends + k for k in range(c)]:
        valid = (pos >= 0) & (pos < n)
        code = np.where(valid, codes[np.clip(pos, 0, max(n - 1, 0))], 4)
        blocks.append(code[:, None] == np.arange(4)[None, :])

    w = cfg.flank_window
    up_lo, up_hi = np.maximum(starts - w, 0), starts
    dn_lo, dn_hi = ends, np.minimum(ends + w, n)
    blocks += [
        _composition(cum, up_lo, up_hi), ((up_hi - up_lo) / w)[:, None],
        _composition(cum, dn_lo, dn_hi), ((dn_hi - dn_lo) / w)[:, None],
    ]
    node_features = np.concatenate([np.asarray(b, np.float32) for b in blocks], axis=1)

    # ---- edges ----
    radius = cfg.edge_radius
    src, dst, gap = _run_pairs_within_radius(starts, ends, radius)
    n_pairs = int(src.size)

    if n_pairs == 0:
        edge_index = np.zeros((2, 0), np.int64)
        edge_features = np.zeros((0, d_edge), np.float32)
    else:
        loop_lo, loop_hi = ends[src], starts[dst]
        len_oh = _len_onehot(length, cfg).astype(np.float32)
        pair = np.concatenate([
            gap[:, None].astype(np.float32),
            (dst - src == 1)[:, None],
            _composition(cum, loop_lo, loop_hi),
        ], axis=1).astype(np.float32)

        # pair k -> rows 2k (i->j, +1) and 2k+1 (j->i, -1); src/dst length one-hots follow
        edge_index = np.stack([np.stack([src, dst], 1).ravel(),
                               np.stack([dst, src], 1).ravel()]).astype(np.int64)
        direction = np.tile(np.array([1.0, -1.0], np.float32), n_pairs)[:, None]
        edge_features = np.concatenate([direction, np.repeat(pair, 2, axis=0),
                                        len_oh[edge_index[0]], len_oh[edge_index[1]]], axis=1)

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "run_starts": starts.astype(np.int32),
        "run_ends": ends.astype(np.int32),
        "n_nodes": n_nodes,
        "n_edges": int(edge_index.shape[1]),
    }
