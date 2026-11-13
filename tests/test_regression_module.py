import numpy as np
import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from g4shape.lightning.metrics import regression_metrics
from g4shape.lightning.build import build_lit_model


def test_metrics_match_numpy():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200) * 5 + 20
    p = y + rng.normal(size=200)
    m = regression_metrics(torch.tensor(p), torch.tensor(y))
    assert m["mae"].item() == np.mean(np.abs(p - y)).item() or abs(m["mae"] - np.abs(p - y).mean()) < 1e-9
    assert abs(m["rmse"] - np.sqrt(((p - y) ** 2).mean())) < 1e-9
    assert abs(m["pearson"] - np.corrcoef(p, y)[0, 1]) < 1e-9
    assert abs(m["r2"] - (1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())) < 1e-9


def _data(n=40, dn=6, de=4):
    g = torch.Generator().manual_seed(0)
    out = []
    for _ in range(n):
        k = int(torch.randint(2, 6, (1,), generator=g))
        e = torch.randint(0, k, (2, 3 * k), generator=g)
        d = Data(x=torch.randn(k, dn, generator=g), edge_index=e, edge_attr=torch.randn(3 * k, de, generator=g),
                 y=torch.rand(1, generator=g) * 50)
        out.append(d)
    return out


def test_regression_fit_and_test_run():
    model = build_lit_model(node_dim=6, edge_dim=4, hidden_dim=16, y_mean=25.0, y_std=14.0)
    dl = DataLoader(_data(), batch_size=8)
    tr = pl.Trainer(max_epochs=2, accelerator="cpu", logger=False, enable_checkpointing=False,
                    enable_progress_bar=False, enable_model_summary=False)
    tr.fit(model, dl, dl)
    res = tr.test(model, dl, verbose=False)[0]
    assert {"test_loss", "test_mae", "test_rmse", "test_r2", "test_pearson"} <= set(res)
    b = next(iter(dl))
    assert model.predict(b).shape == (b.num_graphs,)


def test_detection_metrics_match_sklearn():
    from sklearn.metrics import average_precision_score, jaccard_score, precision_score, recall_score

    from g4shape.lightning.metrics import average_precision, best_threshold, score_detection

    rng = np.random.default_rng(1)
    y = rng.gamma(2.0, 10.0, size=500)
    p = np.round(y + rng.normal(scale=5, size=500), 1)  # rounding -> tied scores
    lab = y > 35
    m = score_detection(torch.tensor(p), torch.tensor(y), threshold=33.1)
    assert abs(m["precision"] - precision_score(lab, p > 33.1)) < 1e-12
    assert abs(m["recall"] - recall_score(lab, p > 33.1)) < 1e-12
    assert abs(m["jaccard"] - jaccard_score(lab, p > 33.1)) < 1e-12
    assert abs(average_precision(torch.tensor(p), torch.tensor(lab)) - average_precision_score(lab, p)) < 1e-12
    t = best_threshold(torch.tensor(p), torch.tensor(lab))
    best = max(jaccard_score(lab, p > g) for g in np.arange(300, 350) / 10)
    assert 30 <= t <= 34.9 and abs(jaccard_score(lab, p > t) - best) < 1e-12


def test_regression_tunes_threshold_on_val_and_reuses_it_on_test():
    model = build_lit_model(node_dim=6, edge_dim=4, hidden_dim=16, y_mean=25.0, y_std=14.0)
    dl = DataLoader(_data(), batch_size=8)
    tr = pl.Trainer(max_epochs=1, accelerator="cpu", logger=False, enable_checkpointing=False,
                    enable_progress_bar=False, enable_model_summary=False)
    tr.fit(model, dl, dl)
    thr = float(model.detection_threshold)
    assert 30.0 <= thr <= 34.9 and "detection_threshold" in model.state_dict()
    res = tr.test(model, dl, verbose=False)[0]
    assert {"test_precision", "test_recall", "test_jaccard", "test_auprc"} <= set(res)
    assert float(model.detection_threshold) == thr and abs(res["test_g4_threshold"] - thr) < 1e-5
