"""Optimizer + learning-rate schedule of the Lightning modules, from one settings dict.

Default = TransG4's recipe (``G4/train.ipynb`` cell 9): Adam(lr=1e-3, beta1=0.9,
beta2=0.999, eps=1e-7, no weight decay) with Keras ``ExponentialDecay(0.001,
decay_steps=10000, decay_rate=0.99)``, i.e. ``lr(step) = lr * 0.99 ** (step / 10000)``,
updated every optimizer step (Keras' default ``staircase=False``).

``scheduler="plateau"``: ReduceLROnPlateau on ``val_loss`` (``plateau_factor``,
``plateau_patience``), the schedule used before; ``scheduler=None``: constant lr.

Keras applies ``eps`` to the not-yet-bias-corrected second moment, PyTorch to the corrected
one; with ``eps=1e-7`` the difference only matters in the first few steps.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch

__all__ = ["TRANSG4_OPTIM", "SCHEDULERS", "resolve_optim", "configure_optimizers"]

TRANSG4_OPTIM: dict[str, Any] = {
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "eps": 1e-7,
    "weight_decay": 0.0,
    "scheduler": "exponential",
    "decay_rate": 0.99,
    "decay_steps": 10000,
    "plateau_factor": 0.5,
    "plateau_patience": 3,
}
SCHEDULERS = ("exponential", "plateau", None)


def resolve_optim(optim: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """``TRANSG4_OPTIM`` updated with ``optim``; unknown keys or schedulers raise."""
    optim = dict(optim or {})
    unknown = sorted(set(optim) - set(TRANSG4_OPTIM))
    if unknown:
        raise ValueError(f"unknown optimizer setting(s) {unknown}; known: {sorted(TRANSG4_OPTIM)}")
    out = {**TRANSG4_OPTIM, **optim}
    out["betas"] = tuple(float(b) for b in out["betas"])
    if out["scheduler"] not in SCHEDULERS:
        raise ValueError(f"scheduler must be one of {SCHEDULERS}, got {out['scheduler']!r}")
    return out


def configure_optimizers(params, optim: Mapping[str, Any]):
    """Lightning ``configure_optimizers`` return value for already-resolved ``optim``."""
    opt = torch.optim.Adam(params, lr=optim["lr"], betas=optim["betas"], eps=optim["eps"],
                           weight_decay=optim["weight_decay"])
    if optim["scheduler"] is None:
        return opt
    if optim["scheduler"] == "exponential":
        rate, steps = float(optim["decay_rate"]), float(optim["decay_steps"])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: rate ** (step / steps))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=optim["plateau_factor"], patience=optim["plateau_patience"])
    return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}
