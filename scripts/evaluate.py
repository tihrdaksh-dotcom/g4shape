"""Score a trained checkpoint on the test split of an index, without training. Run from the repo
root, in env g4gnn:

    python -m scripts.evaluate runs/fast_K/checkpoints/best.ckpt --config configs/fast_k.yaml
    python -m scripts.evaluate ~/G4_GNN/runs/transg4_K_run1/checkpoints/best.ckpt \\
        --config configs/fast_k.yaml --name run1_K          # run1 on the same 2M chr1 rows

* model, its features, flank, strand handling and topology switch: from the checkpoint;
* data (``genome_dir``, ``index_dir``, ``max_test``, ``subsample_seed``, ``eval_batch_size``) and
  the resource limits: from ``--config``, so every model is scored on identical rows;
* G4 calls use the checkpoint's ``detection_threshold`` (tuned on its own validation run), as
  ``src.train`` does after fitting;
* ``--retune``: G4 label (``task.label_threshold``) and threshold grid (``task.threshold_grid``) from
  ``--config`` instead, the threshold re-tuned on its validation rows (max Jaccard), then the test
  split scored with it: re-scores a finished run under a corrected protocol without retraining
  (``project/docs/Metrics validation.md``). Regression metrics do not change.

Writes ``runs/eval/<name>.json`` (metrics + provenance); refuses to overwrite it without ``--force``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from src.config import ExperimentConfig
from src.data.window_dataset import load_window_bundle
from src.lightning.metrics import resolve_threshold_grid
from src.resources import apply_process_settings, plan_resources, prepare_worker_fork
from src.train import load_trained


def evaluate(checkpoint: str | Path, config: str | Path, out: str | Path, cpu: bool = False,
             overrides: tuple[str, ...] = (), retune: bool = False) -> dict:
    data_cfg = ExperimentConfig.from_yaml(config).with_overrides(list(overrides))
    model, mcfg = load_trained(checkpoint)
    d, m = data_cfg.data, mcfg.data
    plan = plan_resources(data_cfg.resources, "cpu" if cpu else data_cfg.trainer.accelerator, 1,
                          "auto", d.num_workers)
    apply_process_settings(plan)
    bundle = load_window_bundle(d.genome_dir, d.index_dir, flank=m.flank, feature_config=mcfg.features,
                                max_val=d.max_val if retune else 1, max_test=d.max_test,
                                seed=d.subsample_seed,
                                reverse_strand=m.reverse_strand, use_topology=m.use_topology)
    test_ds = bundle["test_dataset"]
    print(f"[evaluate] {checkpoint}\n[evaluate] {len(test_ds):,} test rows of {d.index_dir}, "
          f"{plan.accelerator}, {plan.num_workers} workers", flush=True)
    dl = DataLoader(test_ds, batch_size=d.eval_batch_size, num_workers=plan.num_workers,
                    persistent_workers=False, shuffle=False)
    trainer = pl.Trainer(accelerator=plan.accelerator, devices=1, logger=False,
                         precision=mcfg.trainer.precision, enable_model_summary=False)
    t0 = time.time()
    protocol = {"label_threshold": model.label_threshold, "threshold_grid": "checkpoint",
                "threshold_tuned_on": "checkpoint's own validation run"}
    val_metrics = None
    if retune:
        t = data_cfg.task
        model.label_threshold = None if t.label_threshold is None else float(t.label_threshold)
        grid = resolve_threshold_grid(t.threshold_grid)
        if grid is not None:
            model.threshold_grid = grid
        val_ds = bundle["val_dataset"]
        print(f"[evaluate] re-tuning on {len(val_ds):,} val rows: label > {model.label_threshold}, "
              f"grid {model.threshold_grid[0]}..{model.threshold_grid[-1]} ({len(model.threshold_grid)})",
              flush=True)
        val_dl = DataLoader(val_ds, batch_size=d.eval_batch_size, num_workers=plan.num_workers,
                            persistent_workers=False, shuffle=False)
        prepare_worker_fork()
        val_metrics = {k: float(v) for k, v in trainer.validate(model, val_dl, verbose=False)[0].items()}
        protocol = {"label_threshold": model.label_threshold,
                    "threshold_grid": [model.threshold_grid[0], model.threshold_grid[-1],
                                       len(model.threshold_grid)],
                    "threshold_tuned_on": f"val, {len(val_ds)} rows"}
    prepare_worker_fork()
    metrics = {k: float(v) for k, v in trainer.test(model, dl, verbose=False)[0].items()}
    result = {"checkpoint": str(Path(checkpoint).resolve()), "config": str(config),
              "overrides": list(overrides), "index_dir": d.index_dir, "split": "test", "rows": len(test_ds),
              "seconds": round(time.time() - t0, 1), "protocol": protocol, "metrics": metrics}
    if val_metrics is not None:
        result["val_metrics"] = val_metrics
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("checkpoint")
    p.add_argument("--config", required=True, help="experiment YAML giving the data (e.g. configs/fast_k.yaml)")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="override a value of --config (e.g. data.max_test=5000 for a quick check)")
    p.add_argument("--name", help="output name (default: the checkpoint's run directory name)")
    p.add_argument("--out-dir", default="runs/eval")
    p.add_argument("--cpu", action="store_true", help="force CPU")
    p.add_argument("--retune", action="store_true",
                   help="label + threshold grid from --config, threshold re-tuned on its val rows")
    p.add_argument("--force", action="store_true", help="overwrite an existing result")
    a = p.parse_args(argv)
    name = a.name or Path(a.checkpoint).resolve().parent.parent.name
    out = Path(a.out_dir) / f"{name}.json"
    if out.exists() and not a.force:
        p.error(f"{out} exists (pass --force to overwrite it)")
    res = evaluate(a.checkpoint, a.config, out, a.cpu, tuple(a.set), a.retune)
    print(json.dumps(res["metrics"], indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
