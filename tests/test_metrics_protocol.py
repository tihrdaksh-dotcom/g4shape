"""Corrected G4 detection protocol: per-condition label, wide threshold grid from a config spec,
and re-scoring a finished checkpoint (scripts.evaluate --retune) without retraining."""

import json

import numpy as np
import pytest

from g4shape.config import ExperimentConfig
from g4shape.lightning.build import build_lit_model
from g4shape.lightning.metrics import (G4_LABEL_THRESHOLDS, TRANSG4_THRESHOLD_GRID, WIDE_GRID,
                                   resolve_threshold_grid)


def test_resolve_threshold_grid():
    assert resolve_threshold_grid(None) is None
    assert resolve_threshold_grid([30, 31.5]) == (30.0, 31.5)
    g = resolve_threshold_grid(WIDE_GRID)
    assert len(g) == 401 and g[0] == 5.0 and g[-1] == 45.0       # both ends included
    assert 19.0 in g and 24.1 in g and 30.0 in g                  # exact decimals, no float drift
    assert G4_LABEL_THRESHOLDS == {"K": 25.0, "PDS": 35.0}
    for bad in ({"start": 5, "stop": 45}, {"start": 5, "stop": 1, "step": 1},
                {"start": 0, "stop": 1, "step": 0}, []):
        with pytest.raises(ValueError):
            resolve_threshold_grid(bad)


def test_grid_spec_from_config_reaches_the_model():
    cfg = ExperimentConfig().with_overrides(["task.threshold_grid={start: 5.0, stop: 45.0, step: 0.1}"])
    small = dict(node_dim=4, edge_dim=3, global_dim=6, hidden_dim=8, global_encoder="mlp")
    model = build_lit_model(**small, threshold_grid=cfg.task.threshold_grid)
    assert model.threshold_grid == resolve_threshold_grid(WIDE_GRID)
    default = build_lit_model(**small)
    assert default.threshold_grid == tuple(TRANSG4_THRESHOLD_GRID)  # default unchanged


def test_evaluate_retune_rescores_without_retraining(data, tmp_path):
    from scripts.evaluate import evaluate
    from g4shape.train import run

    d, _, _ = data
    base = ["model.hidden_dim=8", "model.graph_encoder_kwargs={num_layers: 1, num_heads: 2}",
            "data.batch_size=4", "data.eval_batch_size=8", "resources.nice=0", "data.num_workers=0",
            f"data.genome_dir={d / 'genome'}", f"data.index_dir={d / 'index'}",
            f"trainer.out_dir={tmp_path}", "trainer.run_name=r", "trainer.accelerator=cpu"]
    res = run(ExperimentConfig().with_overrides(base),   # label 35, TransG4 grid
              trainer_kwargs={"enable_progress_bar": False, "enable_model_summary": False})
    ckpt = res["best_checkpoint"]

    new = tmp_path / "corrected.yaml"
    ExperimentConfig().with_overrides(
        base + ["task.label_threshold=25.0", "task.threshold_grid={start: 0.0, stop: 160.0, step: 0.5}"]
    ).to_yaml(new)
    old = evaluate(ckpt, new, tmp_path / "old.json", cpu=True)
    re = evaluate(ckpt, new, tmp_path / "re.json", cpu=True, retune=True)

    assert old["protocol"]["label_threshold"] == 35.0 and "val_metrics" not in old
    assert re["protocol"]["label_threshold"] == 25.0
    assert re["protocol"]["threshold_grid"] == [0.0, 160.0, 321]
    assert re["metrics"]["test_g4_threshold"] == re["val_metrics"]["val_g4_threshold"]  # tuned on val
    assert re["metrics"]["test_g4_threshold"] in resolve_threshold_grid(
        {"start": 0.0, "stop": 160.0, "step": 0.5})
    for k in ("test_mae", "test_rmse", "test_r2", "test_pearson"):   # regression is protocol-free
        assert np.isclose(re["metrics"][k], old["metrics"][k])
    assert json.loads((tmp_path / "re.json").read_text())["protocol"]["label_threshold"] == 25.0
