"""Streaming FASTA reader (used by the genome store and by ``SequenceDataset.from_fasta``)."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterator

__all__ = ["iter_fasta", "read_fasta"]


def iter_fasta(path: str | Path) -> Iterator[tuple[str, list[str]]]:
    """Yield ``(name, [sequence lines])`` per record, streaming; name = header up to whitespace.
    ``.gz`` files are read transparently."""
    name, lines = None, []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    yield name, lines
                name, lines = line[1:].split()[0], []
            elif name is not None:
                lines.append(line.strip())
    if name is not None:
        yield name, lines


def read_fasta(path: str | Path) -> list[tuple[str, str]]:
    """All records as ``(name, sequence)``; for files of short sequences (not genomes)."""
    return [(name, "".join(lines)) for name, lines in iter_fasta(path)]
