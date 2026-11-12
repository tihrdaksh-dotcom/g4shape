"""Raw GEO archive + FASTA -> genome store + indexes (src/data/prepare.py), then training."""

import gzip
import io
import json
import tarfile

import numpy as np
import pytest

from g4shape.config import ExperimentConfig
from g4shape.data.g4seq_index import G4SeqIndex
from g4shape.data.genome_store import GenomeStore
from g4shape.data.prepare import CONDITIONS, find_bedgraphs, main, prepare
from g4shape.train import run


def _bedgraph(rows, header=False) -> bytes:
    text = ("track type=bedGraph name=mm\n" if header else "")
    text += "".join(f"{c}\t{s}\t{e}\t{v}\n" for c, s, e, v in rows)
    return gzip.compress(text.encode())


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    """A miniature GSE110582_RAW.tar (same member names) + a gzipped hg19-like FASTA."""
    d = tmp_path_factory.mktemp("raw")
    rng = np.random.default_rng(0)
    chroms = {c: "".join(rng.choice(list("ACGT"), 2000)) for c in ("chr1", "chr2", "chr3", "chr4")}
    chroms["chrUn_gl000220"] = "ACGT" * 50  # extra contig, as in UCSC hg19.fa
    with gzip.open(d / "hg19.fa.gz", "wt") as fh:
        for c, s in chroms.items():
            fh.write(f">{c}\n" + "\n".join(s[i:i + 50] for i in range(0, len(s), 50)) + "\n")
    rows = [(c, st, st + 15, round(float(rng.uniform(0, 60)), 2))
            for c in ("chr1", "chr2", "chr3", "chr4") for st in range(0, 1950, 30)]
    rows += [("chr3", 5, 40, 1.0), ("chr22", 10, 25, 3.0)]  # wrong length / not in FASTA
    with tarfile.open(d / "GSE110582_RAW.tar", "w") as tar:
        def add(name, data):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        add("GSM3003535_Arabidopsis_all_w15_th-1_minus.K.bedGraph.gz", _bedgraph(rows[:3]))
        for cond in CONDITIONS.values():
            add(cond["+"], _bedgraph(rows, header=True))
            add(cond["-"], _bedgraph(rows))
    return d, len(rows)


def test_find_bedgraphs_from_tar_dir_and_files(raw, tmp_path):
    d, _ = raw
    got = find_bedgraphs(d / "GSE110582_RAW.tar", "K", tmp_path / "raw")
    assert got["+"].name == CONDITIONS["K"]["+"] and got["-"].parent == tmp_path / "raw"
    assert sorted(p.name for p in (tmp_path / "raw").iterdir()) == sorted(
        [CONDITIONS["K"]["+"], CONDITIONS["K"]["-"]])  # only the needed members extracted
    assert find_bedgraphs(tmp_path, "K", tmp_path / "x") == got          # directory
    assert find_bedgraphs([got["-"], got["+"]], "K", tmp_path / "x") == got  # files
    with pytest.raises(FileNotFoundError, match="PDS"):
        find_bedgraphs(tmp_path / "raw", "PDS", tmp_path / "x")


def test_prepare_builds_store_indexes_and_report(raw, tmp_path):
    d, n_rows = raw
    out = tmp_path / "data"
    rep = prepare(d / "GSE110582_RAW.tar", d / "hg19.fa.gz", out, conditions=("K", "PDS"))
    g = GenomeStore(out / "hg19")
    assert g.names == ["chr1", "chr2", "chr3", "chr4", "chrUn_gl000220"] and g.lengths["chr1"] == 2000
    for cond in ("K", "PDS"):
        s = rep[cond]["splits"]
        # 2 strands; chr1 -> test, chr2 -> val, chr3 + chr4 -> train; 2 bad rows x 2 strands dropped
        per = 2 * 65
        assert (s["test"]["rows"], s["val"]["rows"], s["train"]["rows"]) == (per, per, 2 * per)
        assert rep[cond]["dropped_rows"] == 4 and 0 < s["train"]["g4_fraction"] < 1
        idx = G4SeqIndex(out / f"g4seq_{cond}", "test")
        assert {g.names[idx.row(i)[0]] for i in range(len(idx))} == {"chr1"}
    assert json.loads((out / "prepare_report.json").read_text())["K"]["splits"] == rep["K"]["splits"]

    # a re-run reuses everything (no re-extraction, no rebuild)
    mtime = (out / "g4seq_K" / "index.json").stat().st_mtime
    main(["--gse", str(d / "GSE110582_RAW.tar"), "--genome", str(d / "hg19.fa.gz"),
          "--out", str(out), "--conditions", "K"])
    assert (out / "g4seq_K" / "index.json").stat().st_mtime == mtime


def test_prepared_data_trains(raw, tmp_path):
    d, _ = raw
    out = tmp_path / "data"
    prepare(d / "GSE110582_RAW.tar", d / "hg19.fa.gz", out)
    cfg = ExperimentConfig().with_overrides([
        f"data.genome_dir={out / 'hg19'}", f"data.index_dir={out / 'g4seq_K'}",
        "data.batch_size=16", "data.eval_batch_size=32", "data.num_workers=0",
        "data.max_train=40", "data.max_val=20",
        "model.hidden_dim=8", "model.graph_encoder_kwargs={num_layers: 1, num_heads: 2}",
        f"trainer.out_dir={tmp_path / 'runs'}", "trainer.run_name=r", "trainer.accelerator=cpu",
        "resources.nice=0"])
    res = run(cfg, trainer_kwargs={"enable_progress_bar": False, "enable_model_summary": False})
    assert res["test"]["test_n_samples"] == 130  # whole test chromosome (max_test unset)
