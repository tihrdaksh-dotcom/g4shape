"""Reference genome as one ``uint8`` code array per chromosome, read via memmap.

TransG4's generator one-hot encodes the whole genome in RAM (5 int8 columns per
base, per process). Here a chromosome is 1 byte per base on disk
(``<out_dir>/<chrom>.npy``, codes of ``src.features.sequence``) and is opened
with ``np.load(mmap_mode="r")``: the OS page cache is shared by all DataLoader
workers, nothing is copied on fork, and only the 215-nt windows that are asked
for are ever touched. hg19 is ~3.1 GB on disk.

``GenomeStore.window`` pads with N outside the chromosome and optionally
reverse-complements; windows are returned as code arrays (no one-hot).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from g4shape.data.fasta import iter_fasta
from g4shape.features.sequence import N, encode, reverse_complement

__all__ = ["build_genome_store", "GenomeStore"]

_META = "genome.json"


def build_genome_store(
    fasta: str | Path,
    out_dir: str | Path,
    chroms: Optional[Iterable[str] | Callable[[str], bool]] = None,
) -> dict[str, int]:
    """FASTA -> ``out_dir/<chrom>.npy`` (+ ``genome.json`` with the lengths).

    ``chroms``: names to keep, or a predicate on the name (default: all records).
    One chromosome is held in memory at a time (~4x its length in bytes at peak); ``.gz``
    FASTA is read transparently. Returns ``{name: length}``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if chroms is None:
        keep = lambda name: True  # noqa: E731
    elif callable(chroms):
        keep = chroms
    else:
        wanted = set(chroms)
        keep = wanted.__contains__

    lengths: dict[str, int] = {}
    for name, lines in iter_fasta(fasta):
        if not keep(name):
            continue
        codes = encode("".join(lines))  # one vectorised pass per chromosome (hg19 chr1: ~1 s)
        del lines
        np.save(out_dir / f"{name}.npy", codes)
        lengths[name] = int(codes.size)
    (out_dir / _META).write_text(json.dumps({"chroms": lengths}, indent=1))
    return lengths


class GenomeStore:
    """Read-only, memmap-backed genome. Cheap to pickle (arrays open lazily per process)."""

    def __init__(self, store_dir: str | Path):
        self.store_dir = Path(store_dir)
        meta = json.loads((self.store_dir / _META).read_text())
        self.lengths: dict[str, int] = meta["chroms"]
        self.names: list[str] = list(self.lengths)
        self._ids = {n: i for i, n in enumerate(self.names)}
        self._maps: dict[int, np.ndarray] = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_maps"] = {}
        return state

    def chrom_id(self, name: str) -> int:
        return self._ids[name]

    def _chrom(self, chrom_id: int) -> np.ndarray:
        arr = self._maps.get(chrom_id)
        if arr is None:
            arr = np.load(self.store_dir / f"{self.names[chrom_id]}.npy", mmap_mode="r")
            self._maps[chrom_id] = arr
        return arr

    def window(self, chrom_id: int, start: int, end: int, flank: int = 0,
               reverse: bool = False) -> np.ndarray:
        """Codes of ``[start - flank, end + flank)`` (0-based, end exclusive), N-padded
        past either chromosome end; reverse-complemented when ``reverse``."""
        chrom = self._chrom(chrom_id)
        lo, hi = start - flank, end + flank
        out = np.full(hi - lo, N, dtype=np.uint8)
        a, b = max(lo, 0), min(hi, chrom.size)
        if b > a:
            out[a - lo:b - lo] = chrom[a:b]
        return reverse_complement(out) if reverse else out
