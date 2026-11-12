"""Train + test one experiment from an ``ExperimentConfig`` (YAML).

    python -m src.train --config configs/transg4_k.yaml
    python -m src.train --config configs/transg4_k.yaml --set model.hidden_dim=32 --set data.max_train=1000000
    python -m src.train --print-config > my.yaml        # the defaults, to edit
    python -m src.train --resume runs/<run>             # continue a stopped run
    python -m src.train --config X --auto-resume        # also grow onto GPUs freed later

Stop / resume: Ctrl-C, ``kill <pid>`` or ``touch <run_dir>/STOP`` saves
``checkpoints/resume.ckpt`` and stops (exit code 75); a checkpoint is also written every
``trainer.checkpoint_every_minutes``. ``--resume <run_dir>`` re-plans the resources (the
GPUs free *now*, possibly more or fewer than before) and continues at the same step and
batch: the data order does not depend on the number of GPUs (``ResumableBatchSampler``) and
``data.batch_size`` is global, so the result matches an uninterrupted run up to dropout
randomness. Only resource settings may be changed on resume (``RESUME_OVERRIDABLE``).
``--auto-resume`` supervises the run: it restarts it whenever ``GPUWatcher`` finds more
free GPUs (checked every ``resources.rescale_check_minutes``), and stops on Ctrl-C/STOP.

``run(cfg)``: genome store + G4-seq index -> datasets (``load_window_bundle``) -> model
(``build_lit_model``, dims from ``feature_dims(cfg.features)``) -> ``Trainer.fit`` on
train/val -> ``Trainer.test`` on the best checkpoint. The regression target is
standardised with mean/std of the train split's scores unless ``task.y_mean/y_std`` are set.

Run directory ``<trainer.out_dir>/<run_name>/``: ``config.yaml`` (the resolved config,
incl. y_mean/y_std), ``metrics.csv`` (CSVLogger), ``checkpoints/{best,last}.ckpt``,
``test_metrics.json`` (Table 1 + Fig. 3 metrics on the test split), ``resources.json`` (the
resource plan), ``resources_rank<k>_<stage>.json`` (per-GPU memory cap and peak use),
``sessions.jsonl`` (one line per start / resume: time, step, plan),
``session<k>/metrics.csv`` (the metrics of the k-th session, k >= 2) and
``checkpoints/resume.ckpt`` (the latest stop / periodic checkpoint).

Shared machines: ``src.resources`` picks only free GPUs (and narrows
``CUDA_VISIBLE_DEVICES`` to them), checks that NCCL works between them, caps our GPU
memory, takes a share of the idle CPU cores at raised niceness, and waits if no GPU is
free (config section ``resources``).
With > 1 GPU the run uses DDP; ``data.batch_size`` stays the global batch, and the final
test runs on one device for exact metrics.

``load_trained(ckpt)`` rebuilds a trained model from a checkpoint alone (architecture and
feature settings come from the config stored in it) and returns it with that config, so
new data is featurized exactly as at training time.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch_geometric.loader import DataLoader

from g4shape.config import ExperimentConfig
from g4shape.data.featurize import feature_dims
from g4shape.data.g4seq_index import G4SeqIndex
from g4shape.data.sampler import ResumableBatchSampler, ShardSampler
from g4shape.data.window_dataset import load_window_bundle
from g4shape.lightning.base import LitG4Model
from g4shape.lightning.build import build_lit_model
from g4shape.lightning.resume import PauseCheckpoint, TrainingPaused, latest_checkpoint, resume_position
from g4shape.resources import (HANDOFF_ENV, GPUMemoryCap, GPUWatcher, apply_process_settings,
                           exit_with_parent, handoff, inherited, plan_resources, prepare_worker_fork)

__all__ = ["build_model", "run", "load_trained", "main", "RESUME_OVERRIDABLE",
           "EXIT_PAUSED", "EXIT_RESCALE"]

EXIT_PAUSED = 75   # stopped on request; resume with --resume <run_dir>
EXIT_RESCALE = 76  # paused to restart on more GPUs (--auto-resume restarts it)
SUPERVISED_ENV = "G4GNN_SUPERVISED"
# settings that may change between sessions of one run (anything else would change the run)
RESUME_OVERRIDABLE = ("resources.", "trainer.accelerator=", "trainer.devices=",
                      "trainer.strategy=", "trainer.checkpoint_every_minutes=",
                      "trainer.log_every_n_steps=", "data.num_workers=", "data.eval_batch_size=")


def build_model(cfg: ExperimentConfig) -> LitG4Model:
    """The Lightning model described by ``cfg`` (``task.y_mean/y_std`` must be resolved for
    regression); the full config is stored in ``hparams["experiment"]``."""
    m, t = cfg.model, cfg.task
    model = build_lit_model(
        **feature_dims(cfg.features),
        hidden_dim=m.hidden_dim, dropout=m.dropout,
        graph_encoder=m.graph_encoder, global_encoder=m.global_encoder,
        fusion=m.fusion, classifier=m.classifier,
        graph_encoder_kwargs=m.graph_encoder_kwargs, global_encoder_kwargs=m.global_encoder_kwargs,
        fusion_kwargs=m.fusion_kwargs, classifier_kwargs=m.classifier_kwargs,
        task=t.task, y_mean=t.y_mean if t.y_mean is not None else 0.0,
        y_std=t.y_std if t.y_std is not None else 1.0,
        label_threshold=t.label_threshold, threshold_grid=t.threshold_grid,
        aux_g4_weight=t.aux_g4_weight, aux_g4_pos_weight=t.aux_g4_pos_weight,
        optim=cfg.optim,
    )
    model.hparams["experiment"] = cfg.to_dict()
    return model


def _resolve_target_stats(cfg: ExperimentConfig) -> ExperimentConfig:
    """Fill ``task.y_mean/y_std`` from the train split (regression only, when unset)."""
    t = cfg.task
    if t.task != "regression" or (t.y_mean is not None and t.y_std is not None):
        return cfg
    mean, std = G4SeqIndex(cfg.data.index_dir, "train").score_stats()
    return cfg.with_overrides([f"task.y_mean={t.y_mean if t.y_mean is not None else mean}",
                               f"task.y_std={t.y_std if t.y_std is not None else std}"])


def _run_name(cfg: ExperimentConfig) -> str:
    if cfg.trainer.run_name:
        name = cfg.trainer.run_name
    else:
        name = (f"{time.strftime('%Y%m%d-%H%M%S')}_{cfg.task.task}_"
                f"{cfg.model.graph_encoder}-{cfg.model.global_encoder}")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def _log_session(run_dir: Path, plan, checkpoint: Optional[Path]) -> None:
    entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"),
             "resumed_from": str(checkpoint) if checkpoint else None,
             "step": int(torch.load(checkpoint, map_location="cpu", weights_only=False)["global_step"])
             if checkpoint else 0,
             "accelerator": plan.accelerator, "devices": plan.devices,
             "gpus": [g["index"] for g in plan.gpus], "num_workers": plan.num_workers,
             "torch_threads": plan.torch_threads}
    with open(run_dir / "sessions.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def run(cfg: ExperimentConfig, trainer_kwargs: Optional[dict] = None,
        resume: Optional[str | Path] = None) -> dict:
    """Fit + test. Returns ``{"run_dir", "best_checkpoint", "test": {metric: value}}`` on rank 0
    (``{"run_dir", "rank"}`` on other DDP ranks; ``{"run_dir", "paused": reason,
    "checkpoint", "step"}`` when the run was stopped). ``resume``: the directory of a run to
    continue from its latest checkpoint (``cfg`` = its ``config.yaml``, see ``main``).
    ``trainer_kwargs``: extra ``pl.Trainer`` arguments (e.g. ``limit_train_batches``)."""
    d, t = cfg.data, cfg.trainer
    child = inherited()  # DDP rank > 0, re-launched by Lightning: reuse rank 0's decisions
    if child is None and (not d.genome_dir or not d.index_dir):
        raise ValueError("data.genome_dir and data.index_dir must be set")
    cfg = _resolve_target_stats(cfg)  # deterministic: identical on every rank

    if child is None:
        if resume is not None:
            run_dir = Path(resume)
            if latest_checkpoint(run_dir / "checkpoints") is None:
                raise FileNotFoundError(f"no checkpoint to resume from in {run_dir / 'checkpoints'}")
        plan = plan_resources(cfg.resources, t.accelerator, t.devices, t.strategy, d.num_workers)
        if resume is None:
            run_dir = Path(t.out_dir) / _run_name(cfg)
            run_dir.mkdir(parents=True, exist_ok=False)
        cfg.to_yaml(run_dir / "config.yaml")
        (run_dir / "resources.json").write_text(json.dumps(plan.to_dict(), indent=1))
        handoff(run_dir, plan)
    else:
        run_dir, plan = child
        exit_with_parent()  # rank 0 gone -> stop now, not at the 30 min DDP timeout
    apply_process_settings(plan)
    pl.seed_everything(t.seed, workers=True)
    ckpt_dir = run_dir / "checkpoints"
    resume_ckpt = latest_checkpoint(ckpt_dir) if resume is not None else None
    if child is None:
        _log_session(run_dir, plan, resume_ckpt)

    bundle = load_window_bundle(d.genome_dir, d.index_dir, flank=d.flank,
                                feature_config=cfg.features, max_train=d.max_train,
                                max_val=d.max_val, max_test=d.max_test, seed=d.subsample_seed,
                                reverse_strand=d.reverse_strand, use_topology=d.use_topology)
    ebs = plan.per_device_batch(d.eval_batch_size)
    loader = dict(num_workers=plan.num_workers, persistent_workers=plan.num_workers > 0,
                  pin_memory=plan.accelerator == "gpu")
    # same global batches whatever the number of ranks; skips what resume_ckpt already trained
    n_train = len(bundle["train_dataset"])
    batches = ResumableBatchSampler(n_train, d.batch_size, seed=t.seed,
                                    drop_last=n_train > d.batch_size,
                                    start=resume_position(resume_ckpt))
    train_dl = DataLoader(bundle["train_dataset"], batch_sampler=batches, **loader)
    val_dl = DataLoader(bundle["val_dataset"], batch_size=ebs,
                        sampler=ShardSampler(len(bundle["val_dataset"])), **loader)

    model = build_model(cfg)
    ckpt = ModelCheckpoint(dirpath=ckpt_dir, filename="best", monitor="val_loss",
                           mode="min", save_top_k=1, save_last=True)
    watcher = None
    if child is None and os.environ.get(SUPERVISED_ENV) and cfg.resources.rescale_check_minutes > 0:
        watcher = GPUWatcher(cfg.resources, plan, t.accelerator, t.devices).start()
    pause = PauseCheckpoint(ckpt_dir, t.checkpoint_every_minutes, stop_file=run_dir / "STOP",
                            rescale=(lambda: watcher.triggered) if watcher else None)
    common = dict(accelerator=plan.accelerator, precision=t.precision,
                  log_every_n_steps=t.log_every_n_steps, default_root_dir=run_dir,
                  **(trainer_kwargs or {}))
    trainer = pl.Trainer(max_epochs=t.max_epochs, devices=plan.devices, strategy=plan.strategy,
                         gradient_clip_val=t.gradient_clip_val, use_distributed_sampler=False,
                         logger=CSVLogger(run_dir, name="", version=_log_version(run_dir, resume)),
                         callbacks=[ckpt, pause, GPUMemoryCap(plan.memory_fraction, run_dir)],
                         **common)
    try:
        trainer.fit(model, train_dl, val_dl, ckpt_path=resume_ckpt)
        if pause.stop_requested or (run_dir / "STOP").exists():  # asked during the last validation
            (run_dir / "STOP").unlink(missing_ok=True)
            raise TrainingPaused(str(latest_checkpoint(ckpt_dir)), trainer.global_step)
    except TrainingPaused as p:
        _end_distributed(trainer)
        if trainer.is_global_zero:
            print(f"[train] paused ({p.reason}) at step {p.step}; continue with: "
                  f"python -m src.train --resume {run_dir}", flush=True)
        return {"run_dir": str(run_dir), "paused": p.reason, "checkpoint": p.checkpoint,
                "step": p.step}
    finally:
        if watcher is not None:
            watcher.stop()

    # Test on ONE device: exact metrics over the whole test split (no DDP sample padding)
    if trainer.world_size > 1:
        _end_distributed(trainer)
        if not trainer.is_global_zero:
            return {"run_dir": str(run_dir), "rank": trainer.global_rank}
        tester = pl.Trainer(devices=1, logger=CSVLogger(run_dir, name="", version="test"),
                            callbacks=[GPUMemoryCap(plan.memory_fraction[:1], run_dir)], **common)
    else:
        tester = trainer
    os.environ.pop(HANDOFF_ENV, None)
    best = ckpt.best_model_path or None
    if best is None and resume_ckpt is not None:  # the resumed session saved no better model
        best = str(ckpt_dir / "best.ckpt") if (ckpt_dir / "best.ckpt").exists() else None
    # the other ranks are gone: the single test process may use all their loader workers
    test_workers = plan.num_workers * max(1, trainer.world_size)
    test_dl = DataLoader(bundle["test_dataset"], batch_size=d.eval_batch_size,
                         num_workers=test_workers, persistent_workers=False,
                         pin_memory=plan.accelerator == "gpu")
    prepare_worker_fork()  # no CUDA-owning garbage may reach the forked test workers
    test = tester.test(model, test_dl, ckpt_path=best, verbose=False)[0]
    test = {k: float(v) for k, v in test.items()}
    (run_dir / "test_metrics.json").write_text(json.dumps(test, indent=1))
    return {"run_dir": str(run_dir), "best_checkpoint": best, "test": test}


def _log_version(run_dir: Path, resume) -> str:
    """CSVLogger sub-directory: the run dir for the first session, ``session<k>/`` for the
    k-th (Lightning's CSV writer would drop the earlier rows of a shared metrics.csv)."""
    if resume is None:
        return ""
    k = len((run_dir / "sessions.jsonl").read_text().splitlines())
    return f"session{k}"


def _end_distributed(trainer) -> None:
    if trainer.world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def load_trained(checkpoint: str | Path, map_location="cpu") -> tuple[LitG4Model, ExperimentConfig]:
    """Rebuild a trained model from its checkpoint alone; returns ``(model in eval mode,
    config)``. Featurize new data with ``config.features`` (e.g. ``SequenceDataset(...,
    feature_config=config.features)``)."""
    state = torch.load(checkpoint, map_location=map_location, weights_only=False)
    cfg = ExperimentConfig.from_dict(state["hyper_parameters"]["experiment"])
    model = build_model(cfg)
    model.load_state_dict(state["state_dict"])
    return model.eval(), cfg


def _resume_config(run_dir: Path, overrides: list[str]) -> ExperimentConfig:
    bad = [o for o in overrides if not str(o).startswith(RESUME_OVERRIDABLE)]
    if bad:
        raise SystemExit(f"--resume: only resource settings may change ({', '.join(RESUME_OVERRIDABLE)}); "
                         f"not {bad}")
    return ExperimentConfig.from_yaml(run_dir / "config.yaml").with_overrides(overrides)


def _supervise(cfg: ExperimentConfig, overrides: list[str], config: Optional[str],
               resume: Optional[str]) -> int:
    """``--auto-resume``: run the training in a child process and restart it with
    ``--resume`` whenever it pauses to grow onto more GPUs (exit ``EXIT_RESCALE``). Ctrl-C
    reaches the child (it saves and exits ``EXIT_PAUSED``); SIGTERM is passed on."""
    if resume is None:  # fix the run directory now, so restarts can find it
        name = _run_name(cfg)
        overrides = [*overrides, f"trainer.run_name={name}"]
        run_dir = Path(cfg.trainer.out_dir) / name
    else:
        run_dir = Path(resume)
    env = {**os.environ, SUPERVISED_ENV: "1"}
    env.pop(HANDOFF_ENV, None)
    sets = [a for o in overrides for a in ("--set", o)]
    first = [*(["--config", config] if config else []), *sets] if resume is None else None
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the child handles Ctrl-C
    session = 0
    while True:
        if first is not None and session == 0:
            args = first
        else:
            args = ["--resume", str(run_dir), *[a for o in overrides if o.startswith(RESUME_OVERRIDABLE)
                                                for a in ("--set", o)]]
        session += 1
        print(f"[auto-resume] session {session}: python -m src.train {' '.join(args)}", flush=True)
        child = subprocess.Popen([sys.executable, "-m", "src.train", *args], env=env,
                                 preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
        signal.signal(signal.SIGTERM, lambda s, f: child.send_signal(signal.SIGTERM))
        code = child.wait()
        if code != EXIT_RESCALE:
            print(f"[auto-resume] finished with exit code {code}"
                  + (f"; continue with: python -m src.train --resume {run_dir} --auto-resume"
                     if code == EXIT_PAUSED else ""), flush=True)
            return code


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", help="experiment YAML (default: built-in defaults)")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="override one config value (repeatable)")
    p.add_argument("--print-config", action="store_true",
                   help="print the resolved config as YAML and exit")
    p.add_argument("--resume", metavar="RUN_DIR",
                   help="continue a stopped run from its latest checkpoint (re-plans the resources)")
    p.add_argument("--auto-resume", action="store_true",
                   help="restart the run on more GPUs whenever they become free")
    args = p.parse_args(argv)
    if args.resume and args.config:
        p.error("--resume uses the run's own config.yaml; do not pass --config")
    if args.resume:
        cfg = _resume_config(Path(args.resume), args.set)
    else:
        cfg = ExperimentConfig.from_yaml(args.config) if args.config else ExperimentConfig()
        cfg = cfg.with_overrides(args.set)
    if args.print_config:
        sys.stdout.write(cfg.to_yaml())
        return 0
    if args.auto_resume and inherited() is None and not os.environ.get(SUPERVISED_ENV):
        return _supervise(cfg, args.set, args.config, args.resume)
    out = run(cfg, resume=args.resume)
    print(json.dumps(out, indent=1))
    if out.get("paused"):
        return EXIT_RESCALE if out["paused"] == "rescale" else EXIT_PAUSED
    return 0


if __name__ == "__main__":
    sys.exit(main())
