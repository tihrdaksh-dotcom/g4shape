"""Weighted subsample of a full G4-seq row index, for fast training runs. Run from the repo root,
in env g4gnn:

    python -m scripts.make_subset --source data/g4seq_K --out data/g4seq_K_fast \\
        --cutoff 10 --keep-low 0.02 --val 1000000 --test 2000000 --seed 0

* train: every row with score >= ``cutoff`` (weight 1) plus a Bernoulli(``keep_low``) sample of
  the rows below it (weight 1 / keep_low), in source row order. The weighted MSE is then a
  Horvitz-Thompson estimate of the full-split MSE (see project/docs/Low mismatch score filtering.md), so the
  low-score background is still learned, at ~2 % of the cost.
* val / test: uniform random rows without replacement, unweighted (natural distribution; they
  select checkpoints, tune the G4 threshold and give the metrics). Same rows as
  ``load_window_bundle(max_val=..., seed=...)`` would pick from the full index.

Writes the source column files restricted to the kept rows, ``train.weight.bin`` (float32) and
``index.json`` (source metadata + new ``counts`` + a ``subset`` block: parameters, source
counts, exact full-split score mean/std and kept rows >= / < cutoff). Refuses to write into an
existing output directory unless ``--force``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from g4shape.data.g4seq_index import SPLITS, G4SeqIndex
from g4shape.data.window_dataset import subsample_rows

COLUMNS = ("chrom", "start", "strand", "score")


def _train_rows(score: np.ndarray, cutoff: float, keep_low: float, seed: int,
                chunk: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """(sorted kept rows, their weights, stats) of the train split; one uniform draw per row,
    in fixed-size chunks, so the result depends only on (seed, chunk), not on memory."""
    rng = np.random.default_rng(seed)
    rows, weights = [], []
    n_high = n_low = 0
    total = sq = 0.0
    for a in range(0, len(score), chunk):
        s = np.asarray(score[a:a + chunk])
        v = s.astype(np.float64)
        total += v.sum()
        sq += (v * v).sum()
        high = s >= cutoff
        keep = high | (rng.random(len(s)) < keep_low)
        idx = np.flatnonzero(keep)
        rows.append(idx + a)
        weights.append(np.where(high[idx], 1.0, 1.0 / keep_low).astype(np.float32))
        n_high += int(high.sum())
        n_low += len(idx) - int(high[idx].sum())
    n = len(score)
    mean = total / n if n else 0.0
    std = float(np.sqrt(max(sq / n - mean * mean, 0.0))) if n else 1.0
    stats = {"kept_ge_cutoff": n_high, "kept_lt_cutoff": n_low,
             "full_score_mean": float(mean), "full_score_std": std}
    return np.concatenate(rows), np.concatenate(weights), stats


def _write_columns(src: Path, split: str, n: int, rows: np.ndarray, out: Path, chunk: int) -> None:
    dtypes = {"chrom": np.uint8, "start": np.int32, "strand": np.int8, "score": np.float32}
    for c in COLUMNS:
        col = np.memmap(src / f"{split}.{c}.bin", dtype=dtypes[c], mode="r", shape=(n,))
        with open(out / f"{split}.{c}.bin", "wb") as fh:
            for a in range(0, len(rows), chunk):
                fh.write(np.ascontiguousarray(col[rows[a:a + chunk]]).tobytes())


def build_subset(source: str | Path, out: str | Path, cutoff: float = 10.0, keep_low: float = 0.02,
                 n_val: Optional[int] = 1_000_000, n_test: Optional[int] = 2_000_000, seed: int = 0,
                 chunk: int = 20_000_000, force: bool = False) -> dict:
    """Build the subset index in ``out``; returns its ``index.json`` contents.
    ``n_val`` / ``n_test`` = None keeps the whole split."""
    if not 0 < keep_low <= 1:
        raise ValueError("keep_low must be in (0, 1]")
    source, out = Path(source), Path(out)
    meta = json.loads((source / "index.json").read_text())
    if out.exists():
        if not force:
            raise FileExistsError(f"{out} exists (pass force=True / --force to rebuild it)")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    src_counts = meta["counts"]
    counts, full_stats = {}, {}
    for split, cap in (("val", n_val), ("test", n_test)):
        n = src_counts[split]
        rows = np.arange(n) if cap is None or cap >= n else subsample_rows(n, cap, seed)
        _write_columns(source, split, n, rows, out, chunk)
        counts[split] = int(len(rows))
        full_stats[split] = dict(zip(("mean", "std"), G4SeqIndex(source, split).score_stats()))

    n = src_counts["train"]
    score = np.memmap(source / "train.score.bin", dtype=np.float32, mode="r", shape=(n,))
    rows, weights, st = _train_rows(score, cutoff, keep_low, seed, chunk)
    _write_columns(source, "train", n, rows, out, chunk)
    weights.tofile(out / "train.weight.bin")
    counts["train"] = int(len(rows))
    full_stats["train"] = {"mean": st["full_score_mean"], "std": st["full_score_std"]}

    new = {k: meta[k] for k in ("window", "genome", "test_chroms", "val_chroms")}
    new["counts"] = {s: counts[s] for s in SPLITS}
    new["subset"] = {
        "source": str(source.resolve()), "cutoff": cutoff, "keep_low": keep_low, "seed": seed,
        "chunk": chunk, "val": n_val, "test": n_test, "source_counts": src_counts,
        "full_score_stats": full_stats,
        "train_kept_ge_cutoff": st["kept_ge_cutoff"], "train_kept_lt_cutoff": st["kept_lt_cutoff"],
    }
    (out / "index.json").write_text(json.dumps(new, indent=1))
    return new


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--source", required=True, help="full index dir (e.g. .../data/g4seq_K)")
    p.add_argument("--out", required=True)
    p.add_argument("--cutoff", type=float, default=10.0, help="train rows >= cutoff are all kept")
    p.add_argument("--keep-low", type=float, default=0.02, help="sampling rate below the cutoff")
    p.add_argument("--val", type=int, default=1_000_000, help="val rows (-1 = all)")
    p.add_argument("--test", type=int, default=2_000_000, help="test rows (-1 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="replace an existing --out directory")
    a = p.parse_args(argv)
    meta = build_subset(a.source, a.out, a.cutoff, a.keep_low, None if a.val < 0 else a.val,
                        None if a.test < 0 else a.test, a.seed, force=a.force)
    print(json.dumps({"counts": meta["counts"], **{k: meta["subset"][k] for k in
                      ("train_kept_ge_cutoff", "train_kept_lt_cutoff", "full_score_stats")}}, indent=1))
    for split in SPLITS:
        print(split, "score_stats (weighted if train):", G4SeqIndex(a.out, split).score_stats())


if __name__ == "__main__":
    main()
