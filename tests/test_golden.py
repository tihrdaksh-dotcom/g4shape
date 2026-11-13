"""Golden tests: what the cleaning must not change (built once by ``scripts/make_golden.py``).

FROZEN. Never regenerate ``tests/golden/``: if a test here fails, the code is wrong. The only
allowed edit is adding ``old -> new`` parameter names to ``RENAME`` below when a refactor renames
a weight (phase 4 of the cleaning plan), or a change decided beforehand in docs/DECISIONS.md.

* A. Features: SHA-256 of ``x``, ``edge_index``, ``edge_attr``, ``y`` (exact).
* B. Encoder: ``diffusion_gat`` output for a saved input under saved weights (atol 1e-6).
* C. Head: ``binary_mlp`` output for a saved input under saved weights (atol 1e-6).

Weights are loaded by parameter name (strict), never re-created from a random seed.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from g4shape.data.featurize import feature_dims
from g4shape.data.window_dataset import G4WindowDataset
from g4shape.features.config import FeatureConfig
from g4shape.models.factory import CLASSIFIER_REGISTRY, GRAPH_ENCODER_REGISTRY

# Old parameter name -> new parameter name, for weights renamed by a refactor.
RENAME: dict[str, str] = {}

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path(__file__).resolve().parent / "golden"
REAL_GENOME = ROOT / "data" / "hg19"

FEATURE_SETTINGS = dict(min_run_len=2, edge_radius=50, context_bases=2, flank_window=20,
                        position_encoding=True)
FEATURE_CONFIG = FeatureConfig(**FEATURE_SETTINGS)
FLANK = 100
REVERSE_STRAND = "+"
FIELDS = ("x", "edge_index", "edge_attr", "y")
ATOL = 1e-6


def _digest(tensors) -> str:
    h = hashlib.sha256()
    for t in tensors:
        a = np.ascontiguousarray(t.detach().cpu().numpy())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _hash_windows(datasets) -> dict:
    items = [ds[i] for ds in datasets for i in range(len(ds))]
    return {"n_windows": len(items), **{f: _digest(getattr(d, f) for d in items) for f in FIELDS}}


def _load(name: str) -> dict:
    return torch.load(GOLDEN / name, weights_only=True)


def _renamed(state_dict: dict) -> dict:
    return {RENAME.get(k, k): v for k, v in state_dict.items()}


@pytest.fixture(scope="module")
def features_golden():
    g = json.loads((GOLDEN / "features.json").read_text())
    assert g["feature_config"] == FEATURE_SETTINGS
    assert (g["flank"], g["reverse_strand"]) == (FLANK, REVERSE_STRAND)
    return g


def _check(expected: dict, got: dict) -> None:
    assert got["n_windows"] == expected["n_windows"]
    for f in FIELDS:
        assert got[f] == expected[f], f"golden A: {f} hash changed"


def test_features_synthetic(data, features_golden):
    d, _, _ = data
    got = _hash_windows(
        [G4WindowDataset(d / "genome", d / "index", split, flank=FLANK,
                         feature_config=FEATURE_CONFIG, reverse_strand=REVERSE_STRAND)
         for split in ("train", "val", "test")])
    _check(features_golden["synthetic"], got)


def test_features_real(features_golden):
    g = features_golden["pds_chr1"]
    index_dir = ROOT / g["index_dir"]
    if not (REAL_GENOME.is_dir() and index_dir.is_dir()):
        pytest.skip(f"real-data golden A needs {REAL_GENOME} and {index_dir} (not present)")
    ds = G4WindowDataset(REAL_GENOME, index_dir, g["split"], flank=FLANK,
                         feature_config=FEATURE_CONFIG, rows=np.asarray(g["rows"]),
                         reverse_strand=REVERSE_STRAND)
    _check(g, _hash_windows([ds]))


def test_encoder():
    g = _load("encoder.pt")
    dims = feature_dims(FEATURE_CONFIG)
    assert (dims["node_dim"], dims["edge_dim"]) == (g["node_dim"], g["edge_dim"])
    enc = GRAPH_ENCODER_REGISTRY["diffusion_gat"](
        node_dim=g["node_dim"], edge_dim=g["edge_dim"], hidden_dim=g["hidden_dim"],
        dropout=g["dropout"])
    enc.load_state_dict(_renamed(g["state_dict"]), strict=True)
    enc.eval()
    i = g["input"]
    with torch.no_grad():
        out = enc(i["x"], i["edge_index"], i["edge_attr"], i["batch"], num_graphs=i["num_graphs"])
    torch.testing.assert_close(out, g["output"], atol=ATOL, rtol=0)


def test_head():
    g = _load("head.pt")
    head = CLASSIFIER_REGISTRY["binary_mlp"](in_dim=g["in_dim"], hidden_dim=g["hidden_dim"],
                                             dropout=g["dropout"])
    head.load_state_dict(_renamed(g["state_dict"]), strict=True)
    head.eval()
    with torch.no_grad():
        out = head(g["input"])
    torch.testing.assert_close(out, g["output"], atol=ATOL, rtol=0)
