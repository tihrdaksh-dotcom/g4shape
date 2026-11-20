"""Published-reference table and the guarded comparison (``g4shape.analysis.published``)."""

import json

import pytest

from g4shape.analysis.published import (METRICS, NOT_COMPARABLE, PUBLISHED, ProtocolError, compare,
                                        references)

PDS_TEST_ROWS = 29_923_093


@pytest.fixture
def pds_index(tmp_path):
    """A fake index directory: only index.json, with the complete PDS split counts."""
    (tmp_path / "index.json").write_text(json.dumps(
        {"counts": {"train": 315_437_985, "val": 31_652_070, "test": PDS_TEST_ROWS}}))
    return tmp_path


def _metrics(n, pearson=0.92, mae=1.30, rmse=3.00, r2=0.85):
    return {"test_n_samples": float(n), "test_pearson": pearson, "test_mae": mae,
            "test_rmse": rmse, "test_r2": r2}


def test_table_has_both_conditions_and_all_metrics():
    models = {r.model for r in PUBLISHED}
    assert {"TransG4", "G4mismatch"} <= models
    for cond in ("K", "PDS"):
        for model in models:
            got = {r.metric for r in references(cond) if r.model == model}
            assert got == set(METRICS), (cond, model)
    assert len(PUBLISHED) == len(models) * 2 * len(METRICS)
    vals = {(r.model, r.condition, r.metric): r.value for r in PUBLISHED}
    assert vals["TransG4", "PDS", "pearson"] == 0.941 and vals["G4mismatch", "K", "pearson"] == 0.84


def test_guard_refuses_the_2m_sample(pds_index):
    with pytest.raises(ProtocolError, match="2,000,000"):
        compare(_metrics(2_000_000), "PDS", pds_index)


def test_guard_accepts_the_full_test_set(pds_index):
    out = compare(_metrics(PDS_TEST_ROWS), "PDS", pds_index, label_threshold=35.0)
    assert out["comparable"] and out["note"] is None
    assert len(out["rows"]) == len(references("PDS"))


def test_allow_sample_labels_the_output(pds_index):
    out = compare(_metrics(2_000_000), "PDS", pds_index, allow_sample=True)
    assert not out["comparable"] and out["note"] == NOT_COMPARABLE


def test_delta_and_direction(pds_index):
    rows = compare({"metrics": _metrics(PDS_TEST_ROWS)}, "PDS", pds_index)["rows"]  # eval json
    by = {(r["reference"], r["metric"]): r for r in rows}
    p = by["TransG4", "pearson"]
    assert p["ours"] == 0.92 and p["published"] == 0.941
    assert p["delta"] == pytest.approx(0.92 - 0.941) and not p["better"]
    mae = by["G4mismatch", "mae"]                      # lower is better: 1.30 < 1.47
    assert mae["delta"] == pytest.approx(1.30 - 1.47) and mae["better"]


def test_condition_must_match_label_threshold(pds_index):
    with pytest.raises(ProtocolError, match="label_threshold"):
        compare(_metrics(PDS_TEST_ROWS), "PDS", pds_index, label_threshold=25.0)
    with pytest.raises(ValueError):
        compare(_metrics(PDS_TEST_ROWS), "KPDS", pds_index)


def test_compare_published_script(tmp_path, pds_index):
    from scripts.compare_published import main

    run = tmp_path / "run"
    run.mkdir()
    (run / "config.yaml").write_text("task:\n  label_threshold: 35.0\n")
    (run / "test_metrics.json").write_text(json.dumps(_metrics(PDS_TEST_ROWS)))
    assert main([str(run / "test_metrics.json"), "--condition", "PDS", "--index-dir", str(pds_index)]) == 0
    out = json.loads((run / "published_comparison.json").read_text())
    assert out["comparable"] and len(out["rows"]) == len(references("PDS"))

    ev = tmp_path / "sampled.json"                    # an eval file scored on the 2M sample
    ev.write_text(json.dumps({"protocol": {"label_threshold": 35.0}, "metrics": _metrics(2_000_000)}))
    with pytest.raises(SystemExit):
        main([str(ev), "--condition", "PDS", "--index-dir", str(pds_index)])
    with pytest.raises(SystemExit):                   # wrong condition for the run's threshold
        main([str(run / "test_metrics.json"), "--condition", "K", "--index-dir", str(pds_index)])
