"""The two task modules share one training/evaluation loop (``LitG4Model``) and differ
only in their task hooks; analysis and inference look the same for both."""

import pytorch_lightning as pl
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from g4shape.analysis.embeddings import collect_embeddings
from g4shape.lightning.base import LitG4Model
from g4shape.lightning.build import build_lit_model
from g4shape.lightning.classification_module import LitClassificationModel
from g4shape.lightning.regression_module import LitRegressionModel

KW = dict(node_dim=6, edge_dim=4, global_dim=8, hidden_dim=16, global_encoder_kwargs={"num_radii": 2})
HOOKS = {"__init__", "target", "loss", "postprocess", "g4_labels", "step_metrics", "epoch_metrics"}


def _data(n=40):
    g = torch.Generator().manual_seed(0)
    out = []
    for i in range(n):
        k = int(torch.randint(2, 6, (1,), generator=g))
        d = Data(x=torch.randn(k, 6, generator=g), edge_index=torch.randint(0, k, (2, 3 * k), generator=g),
                 edge_attr=torch.randn(3 * k, 4, generator=g),
                 y=torch.tensor([60.0 if i % 4 == 0 else 10.0]) + torch.rand(1, generator=g))
        d.global_features = torch.randn(1, 8, generator=g)
        out.append(d)
    return out


def _trainer():
    return pl.Trainer(max_epochs=1, accelerator="cpu", logger=False, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False)


def test_subclasses_only_define_task_hooks():
    for cls in (LitRegressionModel, LitClassificationModel):
        assert issubclass(cls, LitG4Model)
        own = {k for k, v in vars(cls).items() if callable(v)}
        assert own <= HOOKS, (cls.__name__, own - HOOKS)


def test_same_analysis_api_for_both_tasks():
    b = next(iter(DataLoader(_data(), batch_size=8)))
    reg = build_lit_model(task="regression", y_mean=25.0, y_std=14.0, **KW)
    clf = build_lit_model(task="classification", **KW)
    r, c = reg.characterize(b), clf.characterize(b)
    assert r.keys() == c.keys() == {"output", "prediction", "is_g4", "details"}
    torch.testing.assert_close(r["prediction"], r["output"].view(-1) * 14.0 + 25.0)
    torch.testing.assert_close(c["prediction"], torch.sigmoid(c["output"].view(-1)))
    for m in (reg, clf):  # inference is deterministic even while the module is in train mode
        assert m.training and torch.equal(m.predict(b), m.predict(b))
        assert torch.equal(m.predict_g4(b), m.predict(b) > m.detection_threshold)
        assert not m.predict(b).requires_grad and m.training


def test_embeddings_use_the_models_own_postprocess():
    """A bare network has no task: its prediction is its raw output (no implicit sigmoid)."""
    reg = build_lit_model(task="regression", y_mean=25.0, y_std=14.0, **KW)
    dl = DataLoader(_data(8), batch_size=8)
    lit, bare = collect_embeddings(reg, dl), collect_embeddings(reg.model, dl)
    torch.testing.assert_close(torch.from_numpy(lit["prediction"]),
                               torch.from_numpy(lit["output"]) * 14.0 + 25.0)
    assert (bare["prediction"] == bare["output"]).all() and "is_g4" not in bare
    assert lit["is_g4"].dtype == bool


def test_classifier_tunes_probability_threshold_on_val():
    dl = DataLoader(_data(), batch_size=8)
    clf = build_lit_model(task="classification", **KW)
    tr = _trainer()
    tr.fit(clf, dl, dl)
    thr = float(clf.detection_threshold)
    assert 0.01 - 1e-6 <= thr <= 0.99 + 1e-6 and "detection_threshold" in clf.state_dict()
    res = tr.test(clf, dl, verbose=False)[0]
    assert {"test_loss", "test_acc", "test_precision", "test_recall", "test_jaccard", "test_auprc"} <= set(res)
    assert abs(res["test_g4_threshold"] - thr) < 1e-6

    m = clf.model
    fixed = LitClassificationModel(m.graph_encoder, m.global_encoder, m.fusion, m.classifier,
                                   threshold_grid=(0.5,))
    _trainer().fit(fixed, dl, dl)
    assert float(fixed.detection_threshold) == 0.5


def test_trainer_predict_and_hparams():
    dl = DataLoader(_data(16), batch_size=8)
    reg = build_lit_model(task="regression", y_mean=25.0, y_std=14.0, **KW)
    out = _trainer().predict(reg, dl)
    assert len(out) == 2 and out[0]["prediction"].shape == out[0]["is_g4"].shape == (8,)
    hp = reg.hparams
    assert hp["y_mean"] == 25.0 and hp["label_threshold"] == 35.0 and "threshold_grid" in hp
    assert not {"graph_encoder", "global_encoder", "fusion", "classifier"} & set(hp)
    assert hp["architecture"]["global_encoder"] == "topo_curve"
