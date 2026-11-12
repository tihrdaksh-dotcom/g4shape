"""G4-seq bedGraph -> a compact on-disk row index, split by chromosome.

Replaces ``prep_data.sh`` (gunzip / append strand / grep chr1, chr2). Only the
coordinates, strand and mismatch score of every 15-nt bin are stored (10 bytes
per row, plain binary columns read back with ``np.memmap``); the sequence of a
row is looked up in the ``GenomeStore`` when the row is used.

Split (paper): test = chr1, validation = chr2, train = every other chromosome.
Rows whose length is not ``window`` and rows on chromosomes missing from the
genome store are dropped (the reference notebook drops length != 15 too).

A split may carry an optional ``weight`` column (float32, ``<split>.weight.bin``): the inverse
sampling probability of each row of a subsampled index (``scripts/make_subset.py``), used as the
sample weight of the loss and of ``score_stats``; without it every row weighs 1.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.genome_store import GenomeStore

__all__ = ["build_g4seq_index", "G4SeqIndex", "SPLITS", "STRAND_SIGN", "STRAND_NAME"]

SPLITS = ("train", "val", "test")
STRAND_SIGN = {"+": 1, "-": 0}  # how a strand is stored in the index's "strand" column
STRAND_NAME = {v: k for k, v in STRAND_SIGN.items()}
_COLUMNS = {"chrom": np.uint8, "start": np.int32, "strand": np.int8, "score": np.float32}
_WEIGHT = {"weight": np.float32}  # optional column (subsampled indexes)
_META = "index.json"


def _open(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "r")


def _header_lines(path) -> int:
    """Leading ``track`` / ``browser`` / ``#`` lines of a bedGraph (UCSC headers)."""
    n = 0
    with _open(path) as fh:
        for line in fh:
            if not line.startswith(("track", "browser", "#")):
                break
            n += 1
    return n


def build_g4seq_index(
    bedgraphs: Mapping[str, str | Path] | Sequence[tuple[str, str | Path]],
    genome: GenomeStore,
    out_dir: str | Path,
    window: int = 15,
    test_chroms: Sequence[str] = ("chr1",),
    val_chroms: Sequence[str] = ("chr2",),
    chunksize: int = 2_000_000,
) -> dict[str, int]:
    """``bedgraphs``: ``{"+": plus.bedGraph[.gz], "-": minus.bedGraph[.gz]}`` (4 columns:
    chr, start, end, mismatch score). Writes ``out_dir/<split>.<column>.bin`` +
    ``index.json``; streams in chunks, so memory does not depend on the file size.
    Returns the number of rows per split."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = list(bedgraphs.items() if isinstance(bedgraphs, Mapping) else bedgraphs)
    if len(genome.names) > 255:
        raise ValueError(f"the index stores chromosome ids as uint8; the genome store has "
                         f"{len(genome.names)} records (build it with chroms=... to keep fewer)")
    chrom_ids = {n: genome.chrom_id(n) for n in genome.names}
    split_of = np.zeros(len(genome.names), dtype=np.int8)  # 0 train, 1 val, 2 test
    for n in val_chroms:
        split_of[chrom_ids[n]] = 1
    for n in test_chroms:
        split_of[chrom_ids[n]] = 2

    files = {(s, c): open(out_dir / f"{s}.{c}.bin", "wb") for s in SPLITS for c in _COLUMNS}
    counts = dict.fromkeys(SPLITS, 0)
    dropped = 0
    try:
        for strand, path in items:
            sign = STRAND_SIGN[strand]
            skip = _header_lines(path)
            with _open(path) as fh:
                for df in pd.read_csv(fh, sep="\t", header=None, usecols=[0, 1, 2, 3], skiprows=skip,
                                      names=["chr", "start", "end", "mm"], chunksize=chunksize,
                                      dtype={"chr": str, "start": np.int64, "end": np.int64,
                                             "mm": np.float32}):
                    cid = df["chr"].map(chrom_ids)
                    ok = cid.notna() & ((df["end"] - df["start"]) == window)
                    dropped += int((~ok).sum())
                    df, cid = df[ok], cid[ok].to_numpy(np.uint8)
                    sp = split_of[cid]
                    cols = {"chrom": cid, "start": df["start"].to_numpy(np.int32),
                            "strand": np.full(len(df), sign, np.int8),
                            "score": df["mm"].to_numpy(np.float32)}
                    for k, s in enumerate(SPLITS):
                        m = sp == k
                        if m.any():
                            for c, a in cols.items():
                                files[(s, c)].write(np.ascontiguousarray(a[m]).tobytes())
                            counts[s] += int(m.sum())
    finally:
        for f in files.values():
            f.close()
    (out_dir / _META).write_text(json.dumps({
        "window": window, "counts": counts, "dropped": dropped, "genome": genome.names,
        "test_chroms": list(test_chroms), "val_chroms": list(val_chroms),
    }, indent=1))
    return counts


class G4SeqIndex:
    """One split of the row index; columns are read-only memmaps (shared across workers)."""

    def __init__(self, index_dir: str | Path, split: str):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        index_dir = Path(index_dir)
        meta = json.loads((index_dir / _META).read_text())
        self.window: int = meta["window"]
        self.split = split
        n = meta["counts"][split]
        self._paths = {c: index_dir / f"{split}.{c}.bin" for c in {**_COLUMNS, **_WEIGHT}}
        self.weighted = self._paths["weight"].exists()
        self._n = n
        self._cols: dict[str, np.ndarray] = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cols"] = {}
        return state

    def __len__(self) -> int:
        return self._n

    def col(self, name: str) -> np.ndarray:
        arr = self._cols.get(name)
        if arr is None:
            if name in _WEIGHT and not self.weighted:
                raise KeyError(f"split {self.split!r} of this index has no weight column")
            dtype = {**_COLUMNS, **_WEIGHT}[name]
            arr = (np.memmap(self._paths[name], dtype=dtype, mode="r", shape=(self._n,))
                   if self._n else np.zeros(0, dtype))
            self._cols[name] = arr
        return arr

    def row(self, i: int) -> tuple[int, int, int, float]:
        """(chrom_id, start, strand_sign, score); strand_sign as in ``STRAND_SIGN``."""
        return (int(self.col("chrom")[i]), int(self.col("start")[i]),
                int(self.col("strand")[i]), float(self.col("score")[i]))

    def weight(self, i: int) -> float:
        """Sample weight of row ``i`` (1.0 for an unweighted split)."""
        return float(self.col("weight")[i]) if self.weighted else 1.0

    def score_stats(self, rows: Optional[np.ndarray] = None,
                    chunk: int = 10_000_000) -> tuple[float, float]:
        """(mean, std) of the mismatch score over this split (or over ``rows``), streamed in
        float64 chunks so a genome-scale split never sits in memory; std is the population
        std (ddof=0), 1.0 if the scores are constant or the split is empty. A weighted split
        gives the weighted statistics (= estimates of the full split's, for a subsample)."""
        col = self.col("score")
        n = len(col) if rows is None else len(rows)
        if n == 0:
            return 0.0, 1.0
        total = sq = wsum = 0.0
        for a in range(0, n, chunk):
            idx = slice(a, a + chunk) if rows is None else np.sort(np.asarray(rows[a:a + chunk]))
            v = col[idx].astype(np.float64)
            w = self.col("weight")[idx].astype(np.float64) if self.weighted else np.ones_like(v)
            total += (w * v).sum()
            sq += (w * v * v).sum()
            wsum += w.sum()
        mean = total / wsum
        var = max(sq / wsum - mean * mean, 0.0)
        return float(mean), float(np.sqrt(var)) if var > 0 else 1.0
