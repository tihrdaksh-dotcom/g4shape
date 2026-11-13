"""Build the golden files that freeze what the cleaning must not change (phase 1 of the cleaning
plan). Run once from the repo root, in env g4gnn:

    python -m scripts.make_golden

Writes, under ``tests/golden/``:

* ``features.json`` (golden A): SHA-256 of ``x``, ``edge_index``, ``edge_attr`` and ``y`` of
  (i) every window of the synthetic genome of ``tests/conftest.py`` (rebuilt here with the same
  recipe) and (ii) 64 evenly spaced chr1 test windows of ``data/g4seq_PDS_fast`` (skipped with a
  message when ``data/`` is missing). ``global_features`` is not hashed: it is removed later.
* ``encoder.pt`` (golden B): a ``diffusion_gat`` encoder's ``state_dict``, a fixed input batch
  of 8 graphs and the encoder's output embedding for it.
* ``head.pt`` (golden C): a ``binary_mlp`` head's ``state_dict``, a fixed [8, 64] input and its
  output.

The feature settings are written out here on purpose (not read from ``configs/``, which change
during the cleaning). Random seeds are used only to make the initial weights and inputs once;
``tests/test_golden.py`` loads the saved tensors and never re-creates them from a seed.

After the goldens are committed, nobody reruns this script: a failing golden test means the
code is wrong, not the golden.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from g4shape.data.featurize import feature_dims, featurize_sequence
from g4shape.data.g4seq_index import G4SeqIndex, build_g4seq_index
from g4shape.data.genome_store import GenomeStore, build_genome_store
from g4shape.data.window_dataset import G4WindowDataset
from g4shape.features.config import FeatureConfig
from g4shape.models.factory import CLASSIFIER_REGISTRY, GRAPH_ENCODER_REGISTRY

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "golden"

FEATURE_CONFIG = FeatureConfig(min_run_len=2, edge_radius=50, context_bases=2, flank_window=20,
                               position_encoding=True)
FLANK = 100
REVERSE_STRAND = "+"
HIDDEN_DIM = 64
DROPOUT = 0.2
FIELDS = ("x", "edge_index", "edge_attr", "y")

REAL_GENOME = ROOT / "data" / "hg19"
REAL_INDEX = ROOT / "data" / "g4seq_PDS_fast"
REAL_SPLIT = "test"  # = chr1
N_REAL = 64

# Golden B input: 8 fixed sequences (canonical G4, bulged, long loop, two-tetrad, G-poor,
# no G-run at all -> the one-node empty graph, ...).
ENCODER_SEQUENCES = (
    "TTAGGGTTAGGGTTAGGGTTAGGGTTA",
    "AAGGGAGGGAGGGAGGGAA",
    "CCGGGTTGGAGGTTGGGTTGGGCC",
    "GGGTTTTTTTTTTTTGGGAGGGTGGG",
    "GGTTGGTTGGTTGG",
    "ACGTACGTACGTACGTACGT",
    "ATATATATATATATAT",
    "GGGGCGGGGCGGGGCGGGGAATTGGGCGGG",
)


def digest(tensors) -> str:
    """SHA-256 over dtype, shape and raw bytes of each tensor, in order."""
    h = hashlib.sha256()
    for t in tensors:
        a = np.ascontiguousarray(t.detach().cpu().numpy())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def hash_windows(datasets) -> dict:
    """{field: sha256} over every window of ``datasets`` (in order), plus the window count."""
    items = [ds[i] for ds in datasets for i in range(len(ds))]
    out = {"n_windows": len(items)}
    for f in FIELDS:
        out[f] = digest(getattr(d, f) for d in items)
    return out


def build_synthetic(d: Path) -> None:
    """The synthetic genome + index of ``tests/conftest.py::data``, same recipe, into ``d``."""
    rng = np.random.default_rng(0)
    seqs = {c: "".join(rng.choice(list("ACGT"), 1500, p=[0.25, 0.2, 0.3, 0.25]))
            for c in ("chr1", "chr2", "chr3")}
    seqs["chr3"] = "acgn" + seqs["chr3"][4:]
    fa = d / "g.fa"
    fa.write_text("".join(f">{c} desc\n" + "\n".join(s[i:i + 60] for i in range(0, len(s), 60))
                          + "\n" for c, s in seqs.items()))
    build_genome_store(fa, d / "genome")
    rows = []
    for c in seqs:
        for st in (0, 5, 300, 700, 1480):
            rows.append((c, st, st + 15, float(st) / 10))
    rows.append(("chr3", 10, 40, 1.0))
    rows.append(("chrUn", 10, 25, 1.0))
    for name in ("plus", "minus"):
        (d / f"{name}.bedGraph").write_text("".join("\t".join(map(str, r)) + "\n" for r in rows))
    build_g4seq_index({"+": d / "plus.bedGraph", "-": d / "minus.bedGraph"},
                      GenomeStore(d / "genome"), d / "index")


def real_rows(n_rows: int) -> np.ndarray:
    """``N_REAL`` evenly spaced row ids of a split with ``n_rows`` rows (no randomness)."""
    return np.linspace(0, n_rows - 1, N_REAL).round().astype(np.int64)


def golden_features() -> dict:
    out = {"feature_config": {"min_run_len": 2, "edge_radius": 50, "context_bases": 2,
                              "flank_window": 20, "position_encoding": True},
           "flank": FLANK, "reverse_strand": REVERSE_STRAND}
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        build_synthetic(d)
        out["synthetic"] = hash_windows(
            [G4WindowDataset(d / "genome", d / "index", split, flank=FLANK,
                             feature_config=FEATURE_CONFIG, reverse_strand=REVERSE_STRAND)
             for split in ("train", "val", "test")])
    if REAL_GENOME.is_dir() and REAL_INDEX.is_dir():
        rows = real_rows(len(G4SeqIndex(REAL_INDEX, REAL_SPLIT)))
        ds = G4WindowDataset(REAL_GENOME, REAL_INDEX, REAL_SPLIT, flank=FLANK,
                             feature_config=FEATURE_CONFIG, rows=rows,
                             reverse_strand=REVERSE_STRAND)
        out["pds_chr1"] = {"index_dir": "data/g4seq_PDS_fast", "split": REAL_SPLIT,
                           "rows": rows.tolist(), **hash_windows([ds])}
    else:
        print(f"WARNING: {REAL_GENOME} or {REAL_INDEX} missing; real-data part of golden A "
              "not written", file=sys.stderr)
    return out


def golden_encoder() -> dict:
    dims = feature_dims(FEATURE_CONFIG)
    torch.manual_seed(0)  # initial weights only; the test loads the saved state_dict
    enc = GRAPH_ENCODER_REGISTRY["diffusion_gat"](
        node_dim=dims["node_dim"], edge_dim=dims["edge_dim"], hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT).eval()
    b = Batch.from_data_list([featurize_sequence(s, FEATURE_CONFIG) for s in ENCODER_SEQUENCES])
    with torch.no_grad():
        y = enc(b.x, b.edge_index, b.edge_attr, b.batch, num_graphs=b.num_graphs)
    return {"node_dim": dims["node_dim"], "edge_dim": dims["edge_dim"], "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT, "state_dict": enc.state_dict(),
            "input": {"x": b.x, "edge_index": b.edge_index, "edge_attr": b.edge_attr,
                      "batch": b.batch, "num_graphs": b.num_graphs},
            "output": y}


def golden_head() -> dict:
    torch.manual_seed(1)  # initial weights only
    head = CLASSIFIER_REGISTRY["binary_mlp"](in_dim=HIDDEN_DIM, hidden_dim=HIDDEN_DIM,
                                             dropout=DROPOUT).eval()
    x = torch.randn(8, HIDDEN_DIM, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        y = head(x)
    return {"in_dim": HIDDEN_DIM, "hidden_dim": HIDDEN_DIM, "dropout": DROPOUT,
            "state_dict": head.state_dict(), "input": x, "output": y}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    feats = golden_features()
    (OUT / "features.json").write_text(json.dumps(feats, indent=1) + "\n")
    torch.save(golden_encoder(), OUT / "encoder.pt")
    torch.save(golden_head(), OUT / "head.pt")
    for part in ("synthetic", "pds_chr1"):
        if part in feats:
            print(part, f"({feats[part]['n_windows']} windows)")
            for f in FIELDS:
                print(f"  {f:<10} {feats[part][f]}")
    for p in sorted(OUT.iterdir()):
        print(f"{p.relative_to(ROOT)}: {p.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
