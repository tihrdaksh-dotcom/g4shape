"""Shared fixtures: a small synthetic genome + G4-seq index (no external data)."""

import numpy as np
import pytest

from src.data.genome_store import GenomeStore, build_genome_store
from src.data.g4seq_index import build_g4seq_index


@pytest.fixture(scope="session")
def data(tmp_path_factory):
    """(dir with genome/ and index/, {chrom: sequence}, rows per split); 3 chromosomes x
    5 positions x 2 strands, chr1 = test, chr2 = val, chr3 = train."""
    d = tmp_path_factory.mktemp("g4")
    rng = np.random.default_rng(0)
    seqs = {c: "".join(rng.choice(list("ACGT"), 1500, p=[.25, .2, .3, .25]))
            for c in ("chr1", "chr2", "chr3")}
    seqs["chr3"] = "acgn" + seqs["chr3"][4:]  # lower case + N must be handled
    fa = d / "g.fa"
    fa.write_text("".join(f">{c} desc\n" + "\n".join(s[i:i + 60] for i in range(0, len(s), 60)) + "\n"
                          for c, s in seqs.items()))
    build_genome_store(fa, d / "genome")
    rows = []  # chr, start, end, mm
    for c in seqs:
        for st in (0, 5, 300, 700, 1480):  # 0/5 and 1480 hit both chromosome ends
            rows.append((c, st, st + 15, float(st) / 10))
    rows.append(("chr3", 10, 40, 1.0))        # wrong length -> dropped
    rows.append(("chrUn", 10, 25, 1.0))       # not in genome -> dropped
    for strand, name in (("+", "plus"), ("-", "minus")):
        (d / f"{name}.bedGraph").write_text("".join("\t".join(map(str, r)) + "\n" for r in rows))
    counts = build_g4seq_index({"+": d / "plus.bedGraph", "-": d / "minus.bedGraph"},
                               GenomeStore(d / "genome"), d / "index")
    return d, seqs, counts
