"""Stop / resume: rank-independent batches, pause checkpoints, resume on another world size."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

import src.data.sampler as sampler_mod
import src.train as train_mod
from src.config import ExperimentConfig
from src.data.sampler import ResumableBatchSampler, ShardSampler
from src.lightning.resume import latest_checkpoint, resume_position
from src.resources import HANDOFF_ENV, GPUWatcher, ResourceConfig, ResourcePlan, gpus_available
from src.train import EXIT_PAUSED, EXIT_RESCALE, RESUME_OVERRIDABLE, main, run

ROOT = Path(__file__).resolve().parents[1]
FAST = ["model.hidden_dim=8", "model.dropout=0.0",
        "model.graph_encoder_kwargs={num_layers: 1, num_heads: 2}",
        "data.batch_size=4", "data.eval_batch_size=8", "resources.nice=0", "data.num_workers=0",
        "trainer.accelerator=cpu", "trainer.max_epochs=2"]
QUIET = {"enable_progress_bar": False, "enable_model_summary": False}


def _ranks(monkeypatch, s, world):
    out = []
    for r in range(world):
        monkeypatch.setattr(sampler_mod, "rank_world", lambda r=r: (r, world))
        out.append(list(s))
    return out


def test_global_batches_do_not_depend_on_world_size(monkeypatch):
    s = ResumableBatchSampler(103, 10, seed=3)
    assert len(s) == 10 and s.sampler is s  # Lightning calls set_epoch on batch_sampler.sampler
    (one,) = _ranks(monkeypatch, s, 1)
    four = _ranks(monkeypatch, s, 4)
    for b in range(len(s)):
        assert sorted(sum((r[b] for r in four), [])) == sorted(one[b])
        assert len({len(r[b]) for r in four}) <= 2  # shares differ by at most one sample
    assert len(set(sum(one, []))) == 100  # last partial batch dropped, no repeats

    s.set_epoch(1)
    (other,) = _ranks(monkeypatch, s, 1)
    assert other != one  # a new shuffle every epoch


def test_resume_skips_trained_batches(monkeypatch):
    full = ResumableBatchSampler(50, 5, seed=0)
    resumed = ResumableBatchSampler(50, 5, seed=0, start=(1, 4))
    for e in (0, 1, 2):
        full.set_epoch(e)
        resumed.set_epoch(e)
        a, b = list(full), list(resumed)
        assert b == (a[4:] if e == 1 else a)  # only the stopped epoch is shortened
        assert len(resumed) == len(full)  # Lightning counts the skipped batches as done
    tiny = ResumableBatchSampler(3, 8)
    assert list(tiny) == [list(tiny)[0]] and sorted(list(tiny)[0]) == [0, 1, 2]


def test_shard_sampler_splits_without_padding(monkeypatch):
    s = ShardSampler(11)
    shards = _ranks(monkeypatch, s, 3)
    assert sum(shards, []) == list(range(11)) and [len(x) for x in shards] == [3, 4, 4]


def _cfg(data, tmp_path, name, *extra):
    d, _, _ = data
    return ExperimentConfig().with_overrides(FAST + [
        f"data.genome_dir={d / 'genome'}", f"data.index_dir={d / 'index'}",
        f"trainer.out_dir={tmp_path}", f"trainer.run_name={name}", *extra])


def _weights(ckpt):
    return torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]


def test_stop_and_resume_matches_uninterrupted_run(data, tmp_path):
    run(_cfg(data, tmp_path, "ref"), trainer_kwargs=QUIET)

    cfg = _cfg(data, tmp_path, "paused", "trainer.checkpoint_every_minutes=0")
    run_dir = tmp_path / "paused"
    # stop request from the STOP file, after batch 1 of epoch 0
    stopper = _StopAfter(1)
    out = _run_with(stopper, cfg)
    assert out["paused"] == "stop" and out["step"] == 1
    ckpt = latest_checkpoint(run_dir / "checkpoints")
    assert ckpt.name == "resume.ckpt" and resume_position(ckpt) == (0, 1)
    assert not (run_dir / "test_metrics.json").exists()

    out = run(train_mod._resume_config(run_dir, []), trainer_kwargs=QUIET, resume=run_dir)
    assert "test" in out and (run_dir / "test_metrics.json").exists()
    sessions = [json.loads(l) for l in (run_dir / "sessions.jsonl").read_text().splitlines()]
    assert [s["step"] for s in sessions] == [0, 1]

    a, b = _weights(tmp_path / "ref/checkpoints/last.ckpt"), _weights(run_dir / "checkpoints/last.ckpt")
    assert a.keys() == b.keys() and all(torch.allclose(a[k], b[k], atol=1e-6) for k in a)
    assert torch.load(run_dir / "checkpoints/last.ckpt", weights_only=False)["global_step"] == 4


class _StopAfter:
    """Create the run's STOP file once ``n`` training batches are done (test helper)."""

    def __init__(self, n):
        self.n = n


def _run_with(stopper, cfg):
    import pytorch_lightning as pl

    class Touch(pl.Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if trainer.global_step >= stopper.n:
                (Path(trainer.default_root_dir) / "STOP").touch()

    orig = pl.Trainer.__init__

    def init(self, *a, callbacks=None, **kw):
        orig(self, *a, callbacks=[Touch(), *(callbacks or [])], **kw)

    pl.Trainer.__init__ = init
    try:
        return run(cfg, trainer_kwargs=QUIET)
    finally:
        pl.Trainer.__init__ = orig


def test_resume_only_allows_resource_changes(tmp_path):
    cfg = ExperimentConfig()
    cfg.to_yaml(tmp_path / "config.yaml")
    ok = train_mod._resume_config(tmp_path, ["trainer.devices=2", "resources.max_gpus=3"])
    assert ok.trainer.devices == 2 and ok.resources.max_gpus == 3
    with pytest.raises(SystemExit, match="only resource settings"):
        train_mod._resume_config(tmp_path, ["model.hidden_dim=4"])
    assert all(o.endswith((".", "=")) for o in RESUME_OVERRIDABLE)
    with pytest.raises(SystemExit):
        main(["--resume", str(tmp_path), "--config", "x.yaml"])


def test_ddp_stop_then_resume_on_one_device(data, tmp_path):
    """2 CPU ranks stopped by the STOP file, resumed through the CLI on ONE device: same
    weights as an uninterrupted single-device run (equal shares of each global batch)."""
    run(_cfg(data, tmp_path, "ref"), trainer_kwargs=QUIET)
    cfg = tmp_path / "ddp.yaml"
    _cfg(data, tmp_path, "ddp", "trainer.devices=2").to_yaml(cfg)
    run_dir = tmp_path / "ddp"
    env = {k: v for k, v in os.environ.items() if k not in (HANDOFF_ENV, "LOCAL_RANK")}
    stop_early = tmp_path / "stop_early.py"
    stop_early.write_text(
        "import sys, pytorch_lightning as pl\n"
        "from pathlib import Path\n"
        "class Touch(pl.Callback):\n"
        "    def on_train_batch_end(self, trainer, m, o, b, i):\n"
        "        if trainer.global_step >= 1 and trainer.is_global_zero:\n"
        "            (Path(trainer.default_root_dir) / 'STOP').touch()\n"
        "orig = pl.Trainer.__init__\n"
        "def init(self, *a, callbacks=None, **kw):\n"
        "    orig(self, *a, callbacks=[Touch(), *(callbacks or [])], **kw)\n"
        "pl.Trainer.__init__ = init\n"
        "from src.train import main\n"
        "sys.exit(main(sys.argv[1:]))\n")
    res = subprocess.run([sys.executable, str(stop_early), "--config", str(cfg)], cwd=ROOT,
                         env={**env, "PYTHONPATH": str(ROOT)}, capture_output=True, text=True,
                         timeout=300)
    assert res.returncode == EXIT_PAUSED, res.stderr[-3000:]
    assert resume_position(run_dir / "checkpoints/resume.ckpt") == (0, 1)

    res = subprocess.run([sys.executable, "-m", "src.train", "--resume", str(run_dir),
                          "--set", "trainer.devices=1"], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr[-3000:]
    sessions = [json.loads(l) for l in (run_dir / "sessions.jsonl").read_text().splitlines()]
    assert [(s["devices"], s["step"]) for s in sessions] == [(2, 0), (1, 1)]
    a, b = _weights(tmp_path / "ref/checkpoints/last.ckpt"), _weights(run_dir / "checkpoints/last.ckpt")
    assert all(torch.allclose(a[k], b[k], atol=1e-5) for k in a)
    assert json.loads((run_dir / "test_metrics.json").read_text())["test_n_samples"] == 10


def test_gpu_watcher_needs_two_checks_in_a_row():
    seen = iter([1, 3, 1, 3, 3])
    plan = ResourcePlan("gpu", 1, "auto", "0", gpus=[{"uuid": "GPU-a"}])
    w = GPUWatcher(ResourceConfig(), plan, available=lambda: next(seen), log=lambda m: None)
    streak = 0
    for expect in (False, False, False, False, True):
        streak = w.check(streak)
        assert w.triggered is expect


def test_gpus_available_counts_ours_plus_free_others():
    smi_out = ("0, GPU-a, X, 24000, 20000, 95\n"   # ours (busy because of us)
               "1, GPU-b, X, 24000, 100, 0\n"      # free
               "2, GPU-c, X, 24000, 20000, 90\n"   # someone else's
               "3, GPU-d, X, 24000, 200, 3\n")     # free
    plan = ResourcePlan("gpu", 1, "auto", "0", gpus=[{"uuid": "GPU-a"}])
    rc = ResourceConfig(sample_seconds=1)
    assert gpus_available(rc, plan, smi=lambda a: smi_out) == 3
    assert gpus_available(ResourceConfig(sample_seconds=1, max_gpus=2), plan, smi=lambda a: smi_out) == 2
    plan.allowed = ["0", "1"]
    assert gpus_available(rc, plan, smi=lambda a: smi_out) == 2
    assert gpus_available(rc, plan, "cpu", smi=lambda a: smi_out) == 0


def test_supervisor_restarts_on_rescale_only(monkeypatch, tmp_path):
    calls, codes = [], iter([EXIT_RESCALE, EXIT_RESCALE, EXIT_PAUSED])

    class FakeProc:
        def __init__(self, cmd, env, preexec_fn):
            calls.append(cmd[3:])
            assert env[train_mod.SUPERVISED_ENV] == "1" and HANDOFF_ENV not in env

        def wait(self):
            return next(codes)

        def send_signal(self, s):
            pass

    monkeypatch.setattr(train_mod.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(train_mod.signal, "signal", lambda *a: None)
    cfg = ExperimentConfig().with_overrides([f"trainer.out_dir={tmp_path}", "trainer.run_name=r"])
    code = train_mod._supervise(cfg, ["trainer.run_name=r", "model.hidden_dim=8",
                                      "trainer.devices=2"], "c.yaml", None)
    assert code == EXIT_PAUSED and len(calls) == 3
    assert calls[0][:2] == ["--config", "c.yaml"] and "model.hidden_dim=8" in calls[0]
    resume = ["--resume", str(tmp_path / "r"), "--set", "trainer.devices=2"]
    assert calls[1] == resume and calls[2] == resume


@pytest.mark.parametrize("sig, devices", [("SIGTERM", 1), ("SIGINT", 2)])
def test_signal_pauses_running_training(data, tmp_path, sig, devices):
    """kill / Ctrl-C (SIGINT to the whole process group, as a terminal does) during training
    -> checkpoint + exit code 75; the run then resumes."""
    import signal
    import time

    cfg = tmp_path / "long.yaml"
    _cfg(data, tmp_path, "long", "trainer.max_epochs=100000", f"trainer.devices={devices}").to_yaml(cfg)
    env = {k: v for k, v in os.environ.items() if k not in (HANDOFF_ENV, "LOCAL_RANK")}
    proc = subprocess.Popen([sys.executable, "-m", "src.train", "--config", str(cfg)], cwd=ROOT,
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    metrics = tmp_path / "long" / "metrics.csv"
    deadline = time.time() + 120
    while not (metrics.exists() and metrics.stat().st_size > 0) and time.time() < deadline:
        time.sleep(0.2)  # a few epochs logged: training is running
    if sig == "SIGINT":
        os.killpg(proc.pid, signal.SIGINT)
    else:
        proc.send_signal(signal.SIGTERM)
    out, err = proc.communicate(timeout=120)
    assert proc.returncode == EXIT_PAUSED, err[-3000:]
    ckpt = tmp_path / "long/checkpoints/resume.ckpt"
    step = torch.load(ckpt, weights_only=False)["global_step"]
    assert step > 0 and '"paused": "stop"' in out
    epoch, done = resume_position(ckpt)
    assert step == 2 * epoch + done  # 2 global batches per epoch


def test_stop_on_last_batch_keeps_that_epochs_validation(data, tmp_path):
    import csv

    run(_cfg(data, tmp_path, "ref"), trainer_kwargs=QUIET)
    run_dir = tmp_path / "p"
    out = _run_with(_StopAfter(2), _cfg(data, tmp_path, "p", "trainer.checkpoint_every_minutes=0"))
    assert out["paused"] == "stop" and out["step"] == 2  # after epoch 0's validation
    run(train_mod._resume_config(run_dir, []), trainer_kwargs=QUIET, resume=run_dir)
    rows = [r for f in (run_dir / "metrics.csv", run_dir / "session2" / "metrics.csv")
            for r in csv.DictReader(open(f))]
    val_epochs = sorted({r["epoch"] for r in rows if r.get("val_loss")})
    assert val_epochs == ["0", "1"]
    a, b = _weights(tmp_path / "ref/checkpoints/last.ckpt"), _weights(run_dir / "checkpoints/last.ckpt")
    assert all(torch.allclose(a[k], b[k], atol=1e-6) for k in a)
