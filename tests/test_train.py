"""Experiment config (TransG4 defaults, strict keys) + the training entry point end to end."""

import json

import numpy as np
import pytest
import torch
import yaml
from torch_geometric.loader import DataLoader

from g4shape.config import ExperimentConfig
from g4shape.data.g4seq_index import G4SeqIndex
from g4shape.data.sequence_dataset import SequenceDataset
from g4shape.lightning.optim import TRANSG4_OPTIM, configure_optimizers, resolve_optim
from g4shape.train import load_trained, main, run

FAST = ["model.hidden_dim=8", "model.graph_encoder_kwargs={num_layers: 1, num_heads: 2}",
        "data.batch_size=4", "data.eval_batch_size=8", "resources.nice=0", "data.num_workers=0"]


def test_defaults_are_transg4():
    cfg = ExperimentConfig()
    assert cfg.task.label_threshold == 35.0
    assert cfg.data.flank == 100 and cfg.data.reverse_strand == "+"
    assert cfg.data.batch_size == 256 and cfg.trainer.max_epochs == 1 and cfg.trainer.seed == 42
    assert cfg.optim["eps"] == 1e-7 and cfg.optim["weight_decay"] == 0.0
    assert cfg.optim["scheduler"] == "exponential" and cfg.optim["decay_rate"] == 0.99
    assert cfg.optim["decay_steps"] == 10000


def test_yaml_roundtrip_and_overrides(tmp_path):
    cfg = ExperimentConfig().with_overrides(["model.hidden_dim=32", "features.edge_radius=40",
                                             "optim.scheduler=plateau", "data.max_train=1000"])
    cfg.to_yaml(tmp_path / "c.yaml")
    back = ExperimentConfig.from_yaml(tmp_path / "c.yaml")
    assert back.to_dict() == cfg.to_dict()
    assert back.model.hidden_dim == 32 and back.features.edge_radius == 40
    assert back.optim["scheduler"] == "plateau" and back.data.max_train == 1000


@pytest.mark.parametrize("bad, match", [
    ({"modle": {}}, "did you mean 'model'"),
    ({"model": {"hiden_dim": 8}}, "did you mean 'hidden_dim'"),
    ({"optim": {"learning_rate": 1}}, "unknown optimizer setting"),
    ({"features": {"edge_radus": 3}}, "unknown config key"),
    ({"task": {"task": "regression"}}, "unknown key"),  # task.task was removed
])
def test_config_typos_raise(bad, match):
    with pytest.raises(ValueError, match=match):
        ExperimentConfig.from_dict(bad)


def test_exponential_schedule_matches_keras():
    w = torch.nn.Parameter(torch.zeros(1))
    out = configure_optimizers([w], resolve_optim({"decay_steps": 10}))
    opt, sched = out["optimizer"], out["lr_scheduler"]["scheduler"]
    assert out["lr_scheduler"]["interval"] == "step" and opt.defaults["eps"] == 1e-7
    for _ in range(25):
        opt.step()
        sched.step()
    assert np.isclose(opt.param_groups[0]["lr"], 1e-3 * 0.99 ** 2.5)  # staircase=False
    assert resolve_optim() == {**TRANSG4_OPTIM, "betas": (0.9, 0.999)}


def test_score_stats_streaming(data):
    d, _, _ = data
    idx = G4SeqIndex(d / "index", "train")
    s = np.asarray(idx.col("score"), dtype=np.float64)
    assert np.allclose(idx.score_stats(chunk=3), (s.mean(), s.std()))
    rows = np.array([4, 0, 7])
    assert np.allclose(idx.score_stats(rows=rows, chunk=2), (s[rows].mean(), s[rows].std()))


def test_run_end_to_end_and_reload(data, tmp_path):
    d, _, _ = data
    cfg = ExperimentConfig().with_overrides(FAST + [
        f"data.genome_dir={d / 'genome'}", f"data.index_dir={d / 'index'}",
        f"trainer.out_dir={tmp_path}", "trainer.run_name=smoke", "trainer.accelerator=cpu"])
    out = run(cfg, trainer_kwargs={"enable_progress_bar": False, "enable_model_summary": False})

    run_dir = tmp_path / "smoke"
    assert out["run_dir"] == str(run_dir)
    for name in ("config.yaml", "metrics.csv", "test_metrics.json", "checkpoints/best.ckpt"):
        assert (run_dir / name).exists(), name
    table1_fig3 = {"test_mae", "test_rmse", "test_r2", "test_pearson",
                   "test_precision", "test_recall", "test_jaccard", "test_auprc"}
    assert table1_fig3 <= set(out["test"])
    assert json.loads((run_dir / "test_metrics.json").read_text()) == out["test"]

    saved = yaml.safe_load((run_dir / "config.yaml").read_text())
    mean, std = G4SeqIndex(d / "index", "train").score_stats()
    assert np.isclose(saved["task"]["y_mean"], mean) and np.isclose(saved["task"]["y_std"], std)

    model, cfg2 = load_trained(out["best_checkpoint"])
    assert cfg2.to_dict() == ExperimentConfig.from_dict(saved).to_dict()
    assert (model.y_mean, model.y_std) == (cfg2.task.y_mean, cfg2.task.y_std)
    ds = SequenceDataset(["GGGTTAGGGTTAGGGTTAGGG" * 10 + "ACGTA"], feature_config=cfg2.features)
    b = next(iter(DataLoader(ds, batch_size=1)))
    assert torch.isfinite(model.predict(b)).all()


def test_cli_print_config(capsys):
    assert main(["--print-config", "--set", "model.hidden_dim=12"]) == 0
    printed = yaml.safe_load(capsys.readouterr().out)
    assert printed["model"]["hidden_dim"] == 12
    assert ExperimentConfig.from_dict(printed).optim == ExperimentConfig().optim
