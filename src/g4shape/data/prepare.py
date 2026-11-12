"""Raw GEO GSE110582 + hg19 FASTA -> everything ``python -m src.train`` reads.

    python -m src.data.prepare --gse /path/GSE110582_RAW.tar --genome /path/hg19.fa --out data
    python -m src.data.prepare --gse /path/extracted_dir --genome /path/hg19.fa.gz --conditions K PDS

Replaces TransG4's ``G4/GSE110582_RAW/prep_data.sh`` (gunzip, append the strand, concatenate
plus + minus, ``grep chr1`` -> test, ``grep chr2`` -> val, rest -> train) and its
notebook's row filter (drop bins whose length != 15), without ever decompressing a
bedGraph to disk:

1. **Genome store** ``<out>/hg19/``: every FASTA record -> ``<chrom>.npy`` (uint8 codes,
   1 byte/base, memory-mapped in training). TransG4 loads the chromosomes named in the
   bedGraph; keeping every record gives the same rows (rows on a chromosome missing from
   the FASTA are dropped and counted).
2. **Row index** ``<out>/g4seq_<condition>/`` per condition, from the two Homo sapiens
   bedGraphs of that condition (``CONDITIONS``): chrom / start / strand / mismatch score
   per 15-nt bin, split test = chr1, val = chr2, train = the rest, 10 bytes per row.
   The ``+`` bedGraph rows are reverse-complemented at load time (``G4WindowDataset``),
   exactly like TransG4's ``MissGen.read_seq``.
3. **Report** ``<out>/prepare_report.json``: rows per split, dropped rows, chromosomes,
   score statistics per split (mean, std, share of G4 bins = score > 35).

``--gse`` may be the 21 GB ``GSE110582_RAW.tar`` (only the 2 needed members per condition are
extracted, to ``<out>/raw/``), a directory containing the ``*.bedGraph.gz`` files (searched
recursively), or the files themselves. Finished steps are skipped on a re-run (``--force``
rebuilds).
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from g4shape.data.g4seq_index import SPLITS, G4SeqIndex, build_g4seq_index
from g4shape.data.genome_store import GenomeStore, build_genome_store

__all__ = ["CONDITIONS", "find_bedgraphs", "prepare", "main"]

# GSE110582, Homo sapiens, 15-nt bins, no threshold (th-1): the files TransG4 used
CONDITIONS = {
    "K": {"gsm": "GSM3003539", "label": "K+",
          "+": "GSM3003539_Homo_all_w15_th-1_plus.K.bedGraph.gz",
          "-": "GSM3003539_Homo_all_w15_th-1_minus.K.bedGraph.gz"},
    "PDS": {"gsm": "GSM3003540", "label": "K+ + PDS",
            "+": "GSM3003540_Homo_all_w15_th-1_plus.PDS.bedGraph.gz",
            "-": "GSM3003540_Homo_all_w15_th-1_minus.PDS.bedGraph.gz"},
}
G4_SCORE = 35.0


def _log(msg: str) -> None:
    print(f"[prepare {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _from_tar(tar_path: Path, names: Sequence[str], raw_dir: Path) -> dict[str, Path]:
    """Extract only ``names`` from the GEO tar (streamed; already extracted files of the
    right size are reused)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    found: dict[str, Path] = {}
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar:
            base = Path(member.name).name
            if base not in names or not member.isfile():
                continue
            dest = raw_dir / base
            if dest.exists() and dest.stat().st_size == member.size:
                _log(f"reusing {dest}")
            else:
                _log(f"extracting {base} ({member.size / 2**30:.2f} GB)")
                src = tar.extractfile(member)
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as out:
                    while chunk := src.read(1 << 24):
                        out.write(chunk)
                tmp.replace(dest)
            found[base] = dest
            if len(found) == len(names):
                break
    return found


def find_bedgraphs(gse: str | Path | Iterable[str | Path], condition: str,
                   raw_dir: Path) -> dict[str, Path]:
    """``{"+": path, "-": path}`` of ``condition``'s bedGraphs from a GEO tar, a directory
    or a list of files."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; known: {sorted(CONDITIONS)}")
    spec = CONDITIONS[condition]
    names = [spec["+"], spec["-"]]
    paths = [Path(p) for p in ([gse] if isinstance(gse, (str, Path)) else gse)]
    found: dict[str, Path] = {}
    for p in paths:
        if p.is_dir():
            for n in names:
                hits = sorted(p.rglob(n))
                if hits:
                    found[n] = hits[0]
        elif p.is_file() and p.name in names:
            found[p.name] = p
        elif p.is_file() and tarfile.is_tarfile(p):
            found.update(_from_tar(p, [n for n in names if n not in found], raw_dir))
    missing = [n for n in names if n not in found]
    if missing:
        raise FileNotFoundError(f"condition {condition} ({spec['label']}): not found in "
                                f"{[str(p) for p in paths]}: {missing}")
    return {"+": found[spec["+"]], "-": found[spec["-"]]}


def _split_stats(index_dir: Path) -> dict:
    out = {}
    for split in SPLITS:
        idx = G4SeqIndex(index_dir, split)
        n = len(idx)
        if n == 0:
            out[split] = {"rows": 0}
            continue
        mean, std = idx.score_stats()
        score = idx.col("score")
        g4 = sum(int((np.asarray(score[a:a + 10_000_000]) > G4_SCORE).sum())
                 for a in range(0, n, 10_000_000))
        out[split] = {"rows": n, "score_mean": mean, "score_std": std,
                      "g4_bins": g4, "g4_fraction": g4 / n}
    return out


def prepare(
    gse: str | Path | Iterable[str | Path],
    genome_fasta: str | Path,
    out_dir: str | Path = "data",
    conditions: Sequence[str] = ("K",),
    genome_name: str = "hg19",
    chroms: Optional[Iterable[str]] = None,
    test_chroms: Sequence[str] = ("chr1",),
    val_chroms: Sequence[str] = ("chr2",),
    force: bool = False,
) -> dict:
    """Build ``<out>/<genome_name>/`` and ``<out>/g4seq_<condition>/`` (see module doc);
    returns (and writes) the report."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "prepare_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() and not force else {}
    t0 = time.time()

    genome_dir = out_dir / genome_name
    if force or not (genome_dir / "genome.json").exists():
        _log(f"building the genome store {genome_dir} from {genome_fasta}")
        lengths = build_genome_store(genome_fasta, genome_dir, chroms=chroms)
        _log(f"genome store: {len(lengths)} records, {sum(lengths.values()) / 1e9:.2f} Gb")
    else:
        _log(f"genome store {genome_dir} exists, skipping (--force to rebuild)")
    genome = GenomeStore(genome_dir)
    missing = [c for c in (*test_chroms, *val_chroms) if c not in genome.names]
    if missing:
        raise ValueError(f"split chromosomes {missing} are not in the FASTA ({genome_fasta})")
    report["genome"] = {"dir": str(genome_dir), "fasta": str(genome_fasta),
                        "records": len(genome.names), "bases": int(sum(genome.lengths.values())),
                        "names": genome.names}

    for cond in conditions:
        index_dir = out_dir / f"g4seq_{cond}"
        if force or not (index_dir / "index.json").exists():
            bedgraphs = find_bedgraphs(gse, cond, out_dir / "raw")
            _log(f"indexing {cond} ({CONDITIONS[cond]['label']}): {bedgraphs['+'].name}, "
                 f"{bedgraphs['-'].name}")
            build_g4seq_index(bedgraphs, genome, index_dir, window=15,
                              test_chroms=test_chroms, val_chroms=val_chroms)
        else:
            _log(f"index {index_dir} exists, skipping (--force to rebuild)")
        meta = json.loads((index_dir / "index.json").read_text())
        report[cond] = {"index_dir": str(index_dir), "label": CONDITIONS[cond]["label"],
                        "dropped_rows": meta["dropped"], "test_chroms": meta["test_chroms"],
                        "val_chroms": meta["val_chroms"], "splits": _split_stats(index_dir)}
        s = report[cond]["splits"]
        _log(f"{cond}: " + ", ".join(f"{k} {v['rows']:,} rows" for k, v in s.items())
             + f"; dropped {meta['dropped']:,}")
    report["seconds"] = round(time.time() - t0, 1)
    report_path.write_text(json.dumps(report, indent=1))
    _log(f"done; report: {report_path}")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gse", nargs="+", required=True,
                    help="GSE110582_RAW.tar, a directory with the bedGraph.gz files, or the files")
    ap.add_argument("--genome", required=True, help="hg19 FASTA (.fa or .fa.gz)")
    ap.add_argument("--out", default="data", help="output directory (default: data)")
    ap.add_argument("--conditions", nargs="+", default=["K"], choices=sorted(CONDITIONS),
                    help="K (K+, default) and/or PDS (K+ + PDS)")
    ap.add_argument("--genome-name", default="hg19")
    ap.add_argument("--force", action="store_true", help="rebuild even if outputs exist")
    args = ap.parse_args(argv)
    gse = args.gse[0] if len(args.gse) == 1 else args.gse
    prepare(gse, args.genome, args.out, args.conditions, args.genome_name, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
