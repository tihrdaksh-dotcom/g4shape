"""Opt-in extensions of the G-run graph (``FeatureConfig`` flags; all off by default).

``extract_graph_full(codes, cfg)`` = ``extract_graph`` + the enabled blocks below, appended
after the original columns (so with every flag off it returns ``extract_graph``'s output
unchanged and ``tests/test_features_equivalence.py`` still covers the base graph).

The target bin is the central ``target_len`` nt of the sequence (215-nt window, flank 100:
positions 100..114, centre 107). Offsets are SIGNED, in the orientation the model sees
(5' -> 3' of the scored strand): negative = 5' of the bin. On G4-seq K+ (chr2) a canonical
motif lifts the bin's score when it sits 0-50 nt 5' of the bin and not when it sits > 30 nt
3' of it, so a distance-only encoding cannot express the response.

* ``position_encoding`` -- nodes: run-centre offset / (n/2), Gaussian RBF of the offset
  (``pos_rbf`` centres over -n/2..n/2, width ``pos_sigma`` nt), in-bin flag, overlap with the
  bin / target_len. Edges: the same offset + RBF for the loop midpoint.
* ``gap_encoding="bucket"`` -- the raw ``gap`` column becomes gap / edge_radius (it dominated
  the attention logits at its raw scale) and loop-length classes 1, 2, 3, 4-5, 6-7, 8-12,
  13-20, 21-30, >30 are appended (one-hot), so each head can learn a bias per loop class.
* ``onehot_context`` = K -- one-hot of the K nt before and after each run (N or outside the
  sequence = 0.25 each, as TransG4 encodes N); the first node layer is then a set of
  learned position-specific motifs around the run.
* ``loop_end_bases`` = K -- one-hot of the first and last K bases of each loop (0 past the
  loop end).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from g4shape.features.config import FeatureConfig
from g4shape.features.g_run_graph import edge_feature_names, extract_graph, node_feature_names

__all__ = ["extract_graph_full", "graph_node_feature_names", "graph_edge_feature_names",
           "uses_extras", "target_bin", "GAP_BUCKETS"]

_BASES = "ACGT"
# loop-length classes: upper bound (inclusive) of each class; the last class is open
GAP_BUCKETS: Tuple[Tuple[str, int], ...] = (("1", 1), ("2", 2), ("3", 3), ("4_5", 5), ("6_7", 7),
                                            ("8_12", 12), ("13_20", 20), ("21_30", 30),
                                            ("gt30", 10 ** 9))


def uses_extras(cfg: FeatureConfig) -> bool:
    return bool(cfg.position_encoding or cfg.gap_encoding != "raw"
                or cfg.onehot_context or cfg.loop_end_bases)


def target_bin(n: int, cfg: FeatureConfig) -> Tuple[int, int, float]:
    """(lo, hi, centre) of the central ``target_len`` nt of an n-nt sequence (hi exclusive)."""
    lo = max((n - cfg.target_len) // 2, 0)
    hi = min(lo + cfg.target_len, n)
    return lo, hi, (lo + hi - 1) / 2


def _rbf_centres(cfg: FeatureConfig, n: int) -> np.ndarray:
    return np.linspace(-n / 2, n / 2, cfg.pos_rbf)


def _position_block(offset: np.ndarray, cfg: FeatureConfig, n: int) -> np.ndarray:
    """[offset / (n/2), RBF(offset)] for signed offsets in nt."""
    rbf = np.exp(-0.5 * ((offset[:, None] - _rbf_centres(cfg, n)[None, :]) / cfg.pos_sigma) ** 2)
    return np.concatenate([(offset / max(n / 2, 1))[:, None], rbf], axis=1)


def _onehot_at(codes: np.ndarray, pos: np.ndarray, valid: np.ndarray, fill: float) -> np.ndarray:
    """One-hot (4 cols) of codes[pos]; N -> 0.25 each; invalid positions -> ``fill`` each."""
    n = codes.size
    code = codes[np.clip(pos, 0, max(n - 1, 0))] if n else np.zeros_like(pos)
    out = (code[:, None] == np.arange(4)[None, :]).astype(np.float32)
    out[valid & (code > 3)] = 0.25
    out[~valid] = fill
    return out


def _gap_buckets(gap: np.ndarray) -> np.ndarray:
    upper = np.array([u for _, u in GAP_BUCKETS[:-1]])
    idx = np.searchsorted(upper, gap, side="left")
    return (idx[:, None] == np.arange(len(GAP_BUCKETS))[None, :]).astype(np.float32)


def _pos_names(prefix: str, cfg: FeatureConfig) -> List[str]:
    return [f"{prefix}_offset_scaled"] + [f"{prefix}_rbf_{i}" for i in range(cfg.pos_rbf)]


def graph_node_feature_names(cfg: FeatureConfig) -> List[str]:
    """Node columns of ``extract_graph_full`` (base columns first)."""
    names = list(node_feature_names(cfg))
    if cfg.position_encoding:
        names += _pos_names("node", cfg) + ["in_bin", "bin_overlap"]
    k = cfg.onehot_context
    names += [f"ctx_up{i}_{b}" for i in range(k, 0, -1) for b in _BASES]
    names += [f"ctx_down{i}_{b}" for i in range(1, k + 1) for b in _BASES]
    return names


def graph_edge_feature_names(cfg: FeatureConfig) -> List[str]:
    """Edge columns of ``extract_graph_full`` (base columns first; ``gap`` -> ``gap_scaled``
    with gap_encoding='bucket')."""
    names = list(edge_feature_names(cfg))
    if cfg.gap_encoding == "bucket":
        names[names.index("gap")] = "gap_scaled"
        names += [f"gap_bucket_{n}" for n, _ in GAP_BUCKETS]
    if cfg.position_encoding:
        names += _pos_names("edge", cfg)
    k = cfg.loop_end_bases
    names += [f"loop_first{i}_{b}" for i in range(1, k + 1) for b in _BASES]
    names += [f"loop_last{i}_{b}" for i in range(1, k + 1) for b in _BASES]
    return names


def extract_graph_full(codes: np.ndarray, cfg: FeatureConfig) -> Dict[str, Any]:
    """``extract_graph`` + the enabled extra blocks (same keys as ``extract_graph``)."""
    codes = np.asarray(codes)
    g = extract_graph(codes, cfg)
    if not uses_extras(cfg):
        return g
    n = codes.size
    starts, ends = g["run_starts"].astype(np.int64), g["run_ends"].astype(np.int64)
    lo, hi, centre = target_bin(n, cfg)

    # ---- nodes ----
    blocks = [g["node_features"]]
    if cfg.position_encoding:
        off = (starts + ends - 1) / 2 - centre
        overlap = np.clip(np.minimum(ends, hi) - np.maximum(starts, lo), 0, None)
        blocks += [_position_block(off, cfg, n), (overlap > 0)[:, None],
                   (overlap / cfg.target_len)[:, None]]
    k = cfg.onehot_context
    if k:
        for pos in [starts - i for i in range(k, 0, -1)] + [ends + i for i in range(k)]:
            blocks.append(_onehot_at(codes, pos, (pos >= 0) & (pos < n), 0.25))
    x = np.concatenate([np.asarray(b, np.float32) for b in blocks], axis=1)

    # ---- edges (pair k = rows 2k (i->j, i < j) and 2k+1 (j->i)) ----
    ei, ea = g["edge_index"], g["edge_features"].copy()
    src, dst = ei[0, ::2], ei[1, ::2]
    loop_lo, loop_hi = ends[src], starts[dst]
    eblocks = [ea]
    if cfg.gap_encoding == "bucket":
        gcol = edge_feature_names(cfg).index("gap")
        ea[:, gcol] = ea[:, gcol] / cfg.edge_radius
        eblocks.append(np.repeat(_gap_buckets(loop_hi - loop_lo), 2, axis=0))
    if cfg.position_encoding:
        eblocks.append(np.repeat(_position_block((loop_lo + loop_hi - 1) / 2 - centre, cfg, n),
                                 2, axis=0))
    k = cfg.loop_end_bases
    if k:
        firsts = [_onehot_at(codes, loop_lo + i, loop_lo + i < loop_hi, 0.0) for i in range(k)]
        lasts = [_onehot_at(codes, loop_hi - 1 - i, loop_hi - 1 - i >= loop_lo, 0.0)
                 for i in range(k)]
        eblocks.append(np.repeat(np.concatenate(firsts + lasts, axis=1), 2, axis=0))
    ea = np.concatenate([np.asarray(b, np.float32) for b in eblocks], axis=1)

    return {**g, "node_features": x.astype(np.float32), "edge_index": ei,
            "edge_features": ea.astype(np.float32), "n_nodes": int(x.shape[0]),
            "n_edges": int(ei.shape[1])}
