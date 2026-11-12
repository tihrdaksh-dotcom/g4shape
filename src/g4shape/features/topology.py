"""K-mer topology vector of one sequence: thin wrapper around the authors' package.

Engine: the ``KmerTopology`` package of Hozumi & Wei 2024, "Revealing the Shape of
Genome Space via K-mer Topology" (arXiv:2412.20202), installed from
``KmerTopology/`` at the repo root (``pip install -e ./KmerTopology --config-settings
editable_mode=compat`` in env g4gnn; ``compat`` keeps the repo-root folder from shadowing
the package as a namespace package).
The maths is not recomputed here; the wrapper only

* turns the uint8 code array into the upper-case string the package expects
  (``src.features.sequence.decode``);
* gets k-mer positions with the package's ``find_kmers_position`` (once per k-mer
  length needed), silencing its ``"<kmer> not valid dna"`` print for k-mers that
  contain N (N-padded windows would otherwise flood every DataLoader worker's stdout);
  such k-mers are skipped by the package, which is kept;
* builds one point cloud per entry of ``cfg.topo_clouds``: a single k-mer ("G", "GG"),
  or a "|"-union of k-mers whose start positions are pooled ("GAGG|GGAG|GGGA");
* runs the package's ``compute_filtration_topology`` on each cloud.

Per cloud the package returns, for each radius of ``linspace(0, step_size * max_step,
max_step)`` on the unweighted threshold graph of the positions (edge when distance <=
radius): Betti-0 (= number of zero Laplacian eigenvalues = number of components) and
the smallest non-zero Laplacian eigenvalue (0 if no edge).

Layout (as the package's ``KmerTopology``): ``[Betti curves of all clouds][eigmin curves
of all clouds]``, clouds in ``cfg.topo_clouds`` order, each curve ``max_step`` long.
Raw values, as in the paper. Empty cloud -> zeros; a single point -> Betti 1, eigmin 0.
"""

from __future__ import annotations

import contextlib
import io
from typing import Dict, List

import numpy as np
from KmerTopology._compute_topology import compute_filtration_topology
from KmerTopology._extract_kmers import find_kmers_position

from g4shape.features.config import FeatureConfig
from g4shape.features.sequence import N, decode

__all__ = ["TOPOLOGY_CURVES", "topology_dim", "topology_layout", "topology_feature_names",
           "filtration_radii", "cloud_positions", "extract_topology"]

TOPOLOGY_CURVES = ("betti", "eigmin")  # outermost axis of the vector, in this order


def topology_layout(cfg: FeatureConfig) -> Dict[str, int]:
    """Shape of the vector: it reshapes to ``[len(TOPOLOGY_CURVES), num_clouds, num_radii]``.
    Consumers that read the curves (``TopologyCurveEncoder``) must be built from this."""
    return {"num_clouds": len(cfg.topo_clouds), "num_radii": cfg.max_step}


def topology_dim(cfg: FeatureConfig) -> int:
    layout = topology_layout(cfg)
    return len(TOPOLOGY_CURVES) * layout["num_clouds"] * layout["num_radii"]


def filtration_radii(cfg: FeatureConfig) -> np.ndarray:
    """The package's filtration grid (``compute_filtration_topology``)."""
    return np.linspace(0, cfg.step_size * cfg.max_step, cfg.max_step)


def topology_feature_names(cfg: FeatureConfig) -> List[str]:
    radii = filtration_radii(cfg)
    return [f"{curve}_{cloud}_r{r:.2f}" for curve in TOPOLOGY_CURVES
            for cloud in cfg.topo_clouds for r in radii]


def cloud_positions(seq: str, clouds, quiet: bool = False) -> Dict[str, np.ndarray]:
    """{cloud: sorted 1-based start positions (float)} from the package's k-mer finder."""
    by_k: Dict[int, Dict[str, np.ndarray]] = {}
    for k in sorted({len(km) for c in clouds for km in c.split("|")}):
        if quiet:
            with contextlib.redirect_stdout(io.StringIO()):
                by_k[k] = find_kmers_position(seq, k)[1]
        else:
            by_k[k] = find_kmers_position(seq, k)[1]
    out = {}
    for c in clouds:
        parts = [by_k[len(km)][km] for km in c.split("|")]
        # unique: k-mers of different lengths can share a start, and the package treats a
        # zero distance as "no edge", so duplicate points would never connect
        out[c] = np.unique(np.concatenate(parts)) if len(parts) > 1 else parts[0]
    return out


def extract_topology(codes: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """Flat float32 vector of length ``topology_dim(cfg)``."""
    codes = np.asarray(codes)
    pos = cloud_positions(decode(codes), cfg.topo_clouds, quiet=bool((codes == N).any()))
    betti, eigmin = [], []
    for c in cfg.topo_clouds:
        b, e = compute_filtration_topology(pos[c], cfg.step_size, cfg.max_step)
        betti.append(np.asarray(b, np.float32))
        eigmin.append(np.asarray(e, np.float32))
    return np.concatenate(betti + eigmin)
