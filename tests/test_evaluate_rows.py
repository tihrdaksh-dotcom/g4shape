"""Row-level evaluation (``g4shape.analysis.evaluate_rows``): synthetic data only."""

import numpy as np
import pytest
import torch
from scipy.stats import spearmanr

from g4shape.analysis import evaluate_rows as ev
from g4shape.lightning.metrics import best_threshold, regression_metrics, score_detection


@pytest.fixture
def scores():
    rng = np.random.default_rng(0)
    y = rng.gamma(0.5, 8.0, 5000)                      # skewed, like mismatch %
    pred = y + rng.normal(0, 4.0, y.size)
    return pred, y


def test_regression_matches_training_metrics(scores):
    pred, y = scores
    ours = ev.regression(pred, y)
    ref = regression_metrics(torch.tensor(pred), torch.tensor(y))
    for k in ("pearson", "mae", "rmse", "r2"):
        assert ours[k] == pytest.approx(float(ref[k]), rel=1e-9), k


def test_spearman_is_rank_correlation():
    x = np.linspace(0.1, 5, 200)
    y = np.exp(3 * x)                                  # monotone but curved
    assert ev.spearman(x, y) == pytest.approx(1.0)
    assert ev.pearson(x, y) < 0.9                      # the old "Spearman" returned this
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=300), rng.normal(size=300)
    assert ev.spearman(a, b) == pytest.approx(spearmanr(a, b).statistic)


def test_within_g4_uses_the_label_threshold():
    y = np.array([1.0, 30.0, 36.0, 40.0, 50.0, 60.0])
    pred = np.array([9.0, 9.0, 36.0, 45.0, 48.0, 70.0])
    pds, k = ev.within_g4(pred, y, 35.0), ev.within_g4(pred, y, 25.0)
    assert pds["n_g4"] == 4 and k["n_g4"] == 5
    assert pds["spearman_g4"] == pytest.approx(spearmanr(pred[2:], y[2:]).statistic)


def test_detection_and_threshold_match_training_metrics(scores):
    pred, y = scores
    grid = [5.0 + 0.5 * i for i in range(60)]
    t = ev.tune_threshold(pred, y, 25.0, grid)
    assert t == best_threshold(torch.tensor(pred), torch.tensor(y) > 25.0, grid)
    ours = ev.detection(pred, y, 25.0, t)
    ref = score_detection(torch.tensor(pred), torch.tensor(y), t, label_threshold=25.0)
    for k in ("precision", "recall", "jaccard", "auprc"):
        assert ours[k] == pytest.approx(float(ref[k]), rel=1e-9), k


def test_classify_windows_first_match_wins():
    seqs = ["TTGGGAGGGAGGGAGGGTT",             # canonical (also matches the other two)
            "TTGGGAAAAAAAAAGGGAGGGAGGGTT",     # one 9-nt loop: long_loop
            "TTGGAGGAGGAGGTT",                 # G2 tracts: two_tetrad
            "ttgggagggaggg",                   # 3 tracts only, lower case: none
            "ACGTACGTACGT"]
    assert list(ev.classify_windows(seqs)) == ["canonical", "long_loop", "two_tetrad", "none", "none"]


def test_per_class_recall_and_residual():
    y = np.array([40.0, 40.0, 10.0, 50.0])
    pred = np.array([38.0, 20.0, 12.0, 45.0])
    cls = np.array(["canonical", "canonical", "canonical", "two_tetrad"])
    out = ev.per_class(pred, y, cls, 35.0, 30.0)
    assert out["canonical"]["n"] == 3 and out["canonical"]["n_g4"] == 2
    assert out["canonical"]["recall"] == 0.5
    assert out["canonical"]["mean_residual"] == pytest.approx((-2 - 20 + 2) / 3)
    assert out["two_tetrad"]["recall"] == 1.0 and out["none"]["n"] == 0


def test_evaluate_rows_needs_identical_rows(scores):
    pred, y = scores
    ids = np.arange(y.size)
    out = ev.evaluate_rows({"a": (ids, pred), "b": (ids, pred + 1)}, y, 35.0,
                           g4_thresholds={"a": 30.0, "b": 30.0},
                           classes=np.full(y.size, "none"))
    assert out["runs"]["a"]["all"]["pearson"] == pytest.approx(out["runs"]["b"]["all"]["pearson"])
    assert "per_class" in out["runs"]["a"] and "detection" in out["runs"]["b"]
    with pytest.raises(ValueError, match="same rows"):
        ev.evaluate_rows({"a": (ids, pred), "b": (ids[::-1], pred)}, y, 35.0)


@pytest.fixture
def chr1_rows():
    """Rows on 40 blocks of 100 kb (bins every 1 kb), a run and a clearly better run."""
    rng = np.random.default_rng(2)
    starts = np.arange(0, 4_000_000, 1000)
    y = rng.gamma(0.5, 8.0, starts.size)
    good = y + rng.normal(0, 2.0, y.size)
    bad = y + rng.normal(0, 8.0, y.size)
    return starts, y, good, bad


def test_bootstrap_identical_runs_give_zero_difference(chr1_rows):
    starts, y, good, _ = chr1_rows
    out = ev.block_bootstrap({"a": good, "b": good.copy()}, y, starts, n_boot=200)
    for m in ev.BOOT_METRICS:
        d = out["diffs"]["a - b"][m]
        assert d["value"] == 0.0 and d["ci95"] == [0.0, 0.0] and d["ci90"] == [0.0, 0.0]


def test_bootstrap_sums_match_direct_recomputation(chr1_rows):
    starts, y, good, _ = chr1_rows
    out = ev.block_bootstrap({"a": good}, y, starts, n_boot=5, keep_samples=True)
    assert out["n_blocks"] == 40
    for m in ev.BOOT_METRICS:                          # the value on all rows
        assert out["runs"]["a"][m]["value"] == pytest.approx(ev.regression(good, y)[m], rel=1e-9)
    blocks = ev.block_ids(starts)
    for b in range(5):                                 # each draw, rebuilt row by row
        idx = np.concatenate([np.flatnonzero(blocks == k) for k, c in
                              enumerate(out["counts"][b].astype(int)) for _ in range(c)])
        direct = ev.regression(good[idx], y[idx])
        for m in ev.BOOT_METRICS:
            assert out["samples"]["a"][m][b] == pytest.approx(direct[m], rel=1e-9), (b, m)


def test_bootstrap_is_seeded_and_detects_a_better_run(chr1_rows):
    starts, y, good, bad = chr1_rows
    a = ev.block_bootstrap({"good": good, "bad": bad}, y, starts, n_boot=300, seed=7)
    b = ev.block_bootstrap({"good": good, "bad": bad}, y, starts, n_boot=300, seed=7)
    assert a["runs"] == b["runs"] and a["diffs"] == b["diffs"]
    d = a["diffs"]["good - bad"]
    assert d["pearson"]["ci95"][0] > 0 and d["rmse"]["ci95"][1] < 0
    lo, hi = a["runs"]["good"]["pearson"]["ci95"]
    assert lo < a["runs"]["good"]["pearson"]["value"] < hi
    lo90, hi90 = a["runs"]["good"]["pearson"]["ci90"]
    assert lo <= lo90 and hi90 <= hi


def test_vs_published(chr1_rows):
    starts, y, good, _ = chr1_rows
    boot = ev.block_bootstrap({"a": good}, y, starts, n_boot=200)
    r = boot["runs"]["a"]["pearson"]
    inside = ev.vs_published(boot, "a", "pearson", r["value"])
    outside = ev.vs_published(boot, "a", "pearson", r["ci95"][1] + 0.05)
    assert inside["contains"] and inside["delta"] == 0.0
    assert not outside["contains"] and outside["delta"] < 0
