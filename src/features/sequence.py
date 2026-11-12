"""Nucleotide codes: the one sequence representation used by the whole data path.

A sequence is a ``uint8`` array with A=0, C=1, G=2, T=3, N=4 (anything else,
including padding outside a chromosome). The genome store, the windows and both
feature extractors all work on these arrays, never on Python strings.
"""

from __future__ import annotations

import numpy as np

__all__ = ["A", "C", "G", "T", "N", "encode", "decode", "reverse_complement"]

A, C, G, T, N = 0, 1, 2, 3, 4

_ENC = np.full(256, N, dtype=np.uint8)
for _code, _b in enumerate("ACGT"):
    _ENC[ord(_b)] = _ENC[ord(_b.lower())] = _code
_ENC[ord("U")] = _ENC[ord("u")] = T  # RNA U -> T

_COMP = np.array([T, G, C, A, N], dtype=np.uint8)
_LETTERS = np.frombuffer(b"ACGTN", dtype=np.uint8)


def encode(seq: str) -> np.ndarray:
    """str -> codes. Whitespace is dropped, ``U`` -> ``T``, non-ACGT -> N."""
    s = "".join(seq.split())
    return _ENC[np.frombuffer(s.encode("ascii", "replace"), dtype=np.uint8)]


def decode(codes: np.ndarray) -> str:
    """codes -> upper-case ``ACGTN`` string (vectorised; inverse of ``encode`` on that alphabet)."""
    return _LETTERS[np.asarray(codes, dtype=np.uint8)].tobytes().decode("ascii")


def reverse_complement(codes: np.ndarray) -> np.ndarray:
    return _COMP[codes[::-1]]
