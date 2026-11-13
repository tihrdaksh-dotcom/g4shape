"""Weighted subsample index (scripts/make_subset.py): weight column, weighted stats, sample_weight
on samples, weighted losses, and a training run on a weighted toy index."""

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from scripts.make_subset import build_subset
from g4shape.data.g4seq_index import G4SeqIndex
from g4shape.data.window_dataset import G4WindowDataset, subsample_rows

CUTOFF, KEEP_LOW = 10.0, 0.5


@pytest.fixture(scope="module")
def subset(data, tmp_path_factory):
    """Subset of the shared toy index (train scores 0, .5, 30, 70, 148 x 2 strands: 6 rows >= 10,
    4 below); seed chosen so that some low rows are kept and some dropped."""
    d, _, _ = data
    out = tmp_path_factory.mktemp("sub") / "index"
    meta = build_subset(d / "index", out, cutoff=CUTOFF, keep_low=KEEP_LOW, n_val=4, n_test=None, seed=2)
    assert 0 < meta["subset"]["train_kept_lt_cutoff"] < 4
    return d, out, meta


def _write_index(path, scores, chrom=0):
    """A bare index (no genome needed) with the given train scores; val/test = 1 row each."""
    path.mkdir(parents=True)
    n = len(scores)
    cols = {"chrom": (np.uint8, chrom), "start": (np.int32, 0), "strand": (np.int8, 1)}
    for split, k in (("train", n), ("val", 1), ("test", 1)):
        for c, (dt, v) in cols.items():
            np.full(k, v, dt).tofile(path / f"{split}.{c}.bin")
        (scores if split == "train" else np.zeros(1)).astype(np.float32).tofile(path / f"{split}.score.bin")
    (path / "index.json").write_text(json.dumps({
        "window": 15, "counts": {"train": n, "val": 1, "test": 1}, "genome": ["chr3"],
        "test_chroms": ["chr1"], "val_chroms": ["chr2"]}))


def test_subset_rows_weights_and_meta(data, subset):
    d, out, meta = subset
    full, sub = G4SeqIndex(d / "index", "train"), G4SeqIndex(out, "train")
    assert sub.weighted and not G4SeqIndex(out, "val").weighted and not full.weighted
    s_full = np.asarray(full.col("score"))
    s_sub, w = np.asarray(sub.col("score")), np.asarray(sub.col("weight"))
    assert w.dtype == np.float32 and len(w) == len(sub) == meta["counts"]["train"]
    # every row >= cutoff kept with weight 1, the kept low rows weigh 1 / keep_low
    assert (s_sub >= CUTOFF).sum() == (s_full >= CUTOFF).sum() == meta["subset"]["train_kept_ge_cutoff"]
    np.testing.assert_array_equal(w, np.where(s_sub >= CUTOFF, 1.0, 1 / KEEP_LOW).astype(np.float32))
    assert (s_sub < CUTOFF).sum() == meta["subset"]["train_kept_lt_cutoff"]
    # kept rows are copies of source rows, in source order
    src = [full.row(i) for i in range(len(full))]
    got = [sub.row(i) for i in range(len(sub))]
    pos = [src.index(r) for r in got]
    assert pos == sorted(pos)
    assert [sub.weight(i) for i in range(len(sub))] == pytest.approx(w.tolist())
    # val: uniform sample = the rows load_window_bundle(max_val=4, seed=2) would pick; test: whole split
    val_full, val_sub = G4SeqIndex(d / "index", "val"), G4SeqIndex(out, "val")
    assert [val_sub.row(i) for i in range(4)] == [val_full.row(int(i)) for i in subsample_rows(len(val_full), 4, 2)]
    assert meta["counts"]["test"] == len(G4SeqIndex(d / "index", "test"))
    on_disk = json.loads((out / "index.json").read_text())
    assert on_disk == meta and on_disk["genome"] == json.loads((d / "index" / "index.json").read_text())["genome"]
    assert on_disk["subset"]["full_score_stats"]["train"] == pytest.approx(
        dict(zip(("mean", "std"), full.score_stats())))
    with pytest.raises(FileExistsError):
        build_subset(d / "index", out)


def test_unweighted_split_weight_api(subset):
    _, out, _ = subset
    val = G4SeqIndex(out, "val")
    assert val.weight(0) == 1.0
    with pytest.raises(KeyError, match="no weight column"):
        val.col("weight")


def test_weighted_score_stats_match_manual(subset):
    _, out, _ = subset
    idx = G4SeqIndex(out, "train")
    s = np.asarray(idx.col("score"), np.float64)
    w = np.asarray(idx.col("weight"), np.float64)
    mean = (w * s).sum() / w.sum()
    std = np.sqrt((w * (s - mean) ** 2).sum() / w.sum())
    assert np.allclose(idx.score_stats(chunk=3), (mean, std))
    rows = np.array([3, 0, 1])
    m = (w[rows] * s[rows]).sum() / w[rows].sum()
    assert np.allclose(idx.score_stats(rows=rows, chunk=2),
                       (m, np.sqrt((w[rows] * (s[rows] - m) ** 2).sum() / w[rows].sum())))


def test_weighted_subset_estimates_full_stats(tmp_path):
    """Horvitz-Thompson: the weighted subset's mean/std are close to the full split's."""
    rng = np.random.default_rng(123)  # not the builder's seed: its draws must be independent of y
    scores = np.where(rng.random(400_000) < 0.02, rng.gamma(2, 15, 400_000), rng.exponential(1.0, 400_000))
    _write_index(tmp_path / "full", scores)
    meta = build_subset(tmp_path / "full", tmp_path / "sub", cutoff=10, keep_low=0.05,
                        n_val=None, n_test=None, chunk=50_000)
    full = G4SeqIndex(tmp_path / "full", "train").score_stats()
    sub = G4SeqIndex(tmp_path / "sub", "train").score_stats()
    assert meta["counts"]["train"] < 0.1 * len(scores)
    assert sub == pytest.approx(full, rel=0.02)
    assert meta["subset"]["full_score_stats"]["train"] == pytest.approx(dict(zip(("mean", "std"), full)))


def test_sample_weight_only_on_weighted_splits(data, subset):
    d, out, _ = subset
    train = G4WindowDataset(d / "genome", out, "train")
    assert all(train[i].sample_weight.shape == (1,) for i in range(len(train)))
    assert [train[i].sample_weight.item() for i in range(len(train))] == pytest.approx(
        np.asarray(train.index.col("weight")).tolist())
    assert not hasattr(G4WindowDataset(d / "genome", out, "val")[0], "sample_weight")
    assert not hasattr(G4WindowDataset(d / "genome", d / "index", "train")[0], "sample_weight")
    b = next(iter(DataLoader(train, batch_size=len(train))))
    assert b.sample_weight.shape == (b.num_graphs,) and b.sample_weight.dtype == torch.float32


def _lit(**kw):
    from g4shape.lightning.build import build_lit_model

    return build_lit_model(node_dim=5, edge_dim=3, global_dim=8, hidden_dim=8,
                           global_encoder_kwargs={"num_radii": 2}, y_mean=1.0, y_std=4.0, **kw)


def test_weighted_regression_loss():
    out = torch.tensor([0.0, 8.0, 9.0, -1.0])
    target = torch.tensor([0.5, 7.0, 10.0, 0.0])   # z(35) = 8.5 -> third is a G4
    w = torch.tensor([50.0, 1.0, 1.0, 50.0])
    plain = _lit()
    assert torch.equal(plain.loss(out, target), F.mse_loss(out, target))
    assert torch.allclose(plain.loss(out, target, torch.ones(4)), F.mse_loss(out, target))
    assert torch.allclose(plain.loss(out, target, w), (w * (out - target) ** 2).sum() / w.sum())
    aux = _lit(aux_g4_weight=0.5, aux_g4_pos_weight=10.0)
    bce = F.binary_cross_entropy_with_logits(out - 8.5, (target > 8.5).float(),
                                             pos_weight=torch.tensor(10.0), reduction="none")
    assert torch.allclose(aux.loss(out, target), F.mse_loss(out, target) + 0.5 * bce.mean())
    assert torch.allclose(aux.loss(out, target, torch.ones(4)), aux.loss(out, target))
    assert torch.allclose(aux.loss(out, target, w),
                          ((w * (out - target) ** 2).sum() + 0.5 * (w * bce).sum()) / w.sum())


def test_train_step_uses_sample_weight(data, subset):
    """The training step passes ``batch.sample_weight`` to the loss."""
    from g4shape.data.window_dataset import load_window_bundle
    from g4shape.lightning.build import build_lit_model

    d, out, _ = subset
    bundle = load_window_bundle(d / "genome", out)
    model = build_lit_model(bundle["node_dim"], bundle["edge_dim"], bundle["global_dim"], hidden_dim=8,
                            y_mean=20.0, y_std=30.0, global_layout=bundle["global_layout"],
                            graph_encoder_kwargs={"num_layers": 1, "num_heads": 2})
    seen, orig = [], model.loss
    model.loss = lambda o, t, weight=None: seen.append(weight) or orig(o, t, weight)
    b = next(iter(DataLoader(bundle["train_dataset"], batch_size=len(bundle["train_dataset"]))))
    loss = model.training_step(b, 0)
    assert torch.isfinite(loss) and torch.equal(seen[-1], b.sample_weight)
    b = next(iter(DataLoader(bundle["val_dataset"], batch_size=4)))
    model.validation_step(b, 0)
    assert seen[-1] is None


def test_run_end_to_end_on_weighted_index(data, subset, tmp_path):
    import yaml

    from g4shape.config import ExperimentConfig
    from g4shape.train import run

    d, out, _ = subset
    cfg = ExperimentConfig().with_overrides([
        "model.hidden_dim=8", "model.graph_encoder_kwargs={num_layers: 1, num_heads: 2}",
        "data.batch_size=4", "data.eval_batch_size=8", "resources.nice=0", "data.num_workers=0",
        f"data.genome_dir={d / 'genome'}", f"data.index_dir={out}",
        f"trainer.out_dir={tmp_path}", "trainer.run_name=weighted", "trainer.accelerator=cpu"])
    res = run(cfg, trainer_kwargs={"enable_progress_bar": False, "enable_model_summary": False})
    assert np.isfinite(res["test"]["test_mae"]) and np.isfinite(res["test"]["test_loss"])
    saved = yaml.safe_load((tmp_path / "weighted" / "config.yaml").read_text())
    mean, std = G4SeqIndex(out, "train").score_stats()   # weighted stats of the subset
    assert np.isclose(saved["task"]["y_mean"], mean) and np.isclose(saved["task"]["y_std"], std)
