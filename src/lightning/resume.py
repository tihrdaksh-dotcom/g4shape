"""Stop a training run at any time and resume it later (possibly on other GPUs).

``PauseCheckpoint`` (a Lightning callback) writes ``<dir>/resume.ckpt`` after a training
batch, every ``every_minutes`` and when a stop is requested; a stop then ends ``fit`` with
``TrainingPaused``. A stop is requested by

* Ctrl-C (first press; a second press aborts at once, without saving),
* ``kill <pid>`` (SIGTERM) of any rank, or
* creating the file ``<stop_file>`` (``touch <run_dir>/STOP``; removed when the run pauses).

Every rank takes part in the decision (one tiny all-reduce per batch), so all ranks save the
same step and stop together. The checkpoint also stores where the run is in the epoch
(``checkpoint["resume"] = {"epoch", "batches_done"}``), which ``ResumableBatchSampler``
uses to skip the batches already trained on. A stop requested during validation or test
takes effect at the next training batch (or right after ``fit`` returns).

``rescale`` (rank 0 only, e.g. ``GPUWatcher.triggered``): when it returns True the run pauses
the same way, with ``reason="rescale"``, so a supervisor can restart it on more GPUs.
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path
from typing import Callable, Optional

import pytorch_lightning as pl
import torch

__all__ = ["PauseCheckpoint", "TrainingPaused", "RESUME_KEY", "latest_checkpoint", "resume_position"]

RESUME_KEY = "resume"
_SIGNALS = (signal.SIGINT, signal.SIGTERM)


class TrainingPaused(Exception):
    """Raised out of ``Trainer.fit`` after a stop request; the run can be resumed."""

    def __init__(self, checkpoint: Optional[str], step: int, reason: str = "stop"):
        super().__init__(f"training paused ({reason}) at step {step}; checkpoint: {checkpoint}")
        self.checkpoint, self.step, self.reason = checkpoint, step, reason


class PauseCheckpoint(pl.Callback):
    def __init__(self, dirpath: str | Path, every_minutes: Optional[float] = 30.0,
                 stop_file: Optional[str | Path] = None,
                 rescale: Optional[Callable[[], bool]] = None, clock=time.monotonic):
        self.dirpath = Path(dirpath)
        self.every = None if not every_minutes or every_minutes <= 0 else every_minutes * 60.0
        self.stop_file = Path(stop_file) if stop_file else None
        self.rescale, self.clock = rescale, clock
        self.stop_requested = False
        self.path = self.dirpath / "resume.ckpt"
        self._last_save = clock()
        self._epoch, self._batches_done = 0, 0
        self._old_handlers: dict = {}

    # ---- stop requests -------------------------------------------------------------------
    def _on_signal(self, signum, frame):
        if self.stop_requested and signum == signal.SIGINT:  # second Ctrl-C: abort now
            raise KeyboardInterrupt
        self.stop_requested = True
        print(f"\n[pause] {signal.Signals(signum).name} received: saving a checkpoint and stopping "
              "after the current batch (Ctrl-C again to abort without saving)", flush=True)

    def _install(self):
        if not self._old_handlers and _main_thread():
            for s in _SIGNALS:  # replaces Lightning's SIGTERM handling for the training loop
                self._old_handlers[s] = signal.signal(s, self._on_signal)

    def _restore(self):
        for s, h in self._old_handlers.items():
            signal.signal(s, h)
        self._old_handlers = {}

    def on_train_start(self, trainer, pl_module):
        self._install()
        self._last_save = self.clock()

    def on_train_end(self, trainer, pl_module):
        self._restore()

    def on_exception(self, trainer, pl_module, exception):
        self._restore()

    # ---- periodic / requested saves ------------------------------------------------------
    def _agree(self, trainer, pl_module, *flags: bool) -> list[bool]:
        """Flags OR-ed over all ranks (every rank must call this at the same batch)."""
        t = torch.tensor([float(f) for f in flags], device=pl_module.device)
        if trainer.world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
        return [bool(v) for v in t.tolist()]

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._epoch, self._batches_done = trainer.current_epoch, batch_idx + 1
        if batch_idx + 1 < trainer.num_training_batches:
            self._checkpoint_or_stop(trainer, pl_module)
        # last batch: this epoch's validation runs first (Lightning would skip it on a resume
        # from a checkpoint taken here), so decide in on_train_epoch_end

    def on_train_epoch_end(self, trainer, pl_module):
        self._checkpoint_or_stop(trainer, pl_module)

    def _checkpoint_or_stop(self, trainer, pl_module):
        stop = self.stop_requested or (trainer.is_global_zero and self.stop_file is not None
                                       and self.stop_file.exists())
        rescale = trainer.is_global_zero and self.rescale is not None and bool(self.rescale())
        due = self.every is not None and self.clock() - self._last_save >= self.every
        stop, rescale, due = self._agree(trainer, pl_module, stop, rescale, due)
        if stop or rescale or due:
            self.save(trainer)
        if stop or rescale:
            if trainer.is_global_zero and self.stop_file is not None and self.stop_file.exists():
                self.stop_file.unlink()
            raise TrainingPaused(str(self.path), trainer.global_step,
                                 "stop" if stop else "rescale")

    def save(self, trainer) -> None:
        """Write ``resume.ckpt`` atomically (every rank must call this; rank 0 writes)."""
        tmp = self.dirpath / "resume.ckpt.tmp"
        trainer.save_checkpoint(tmp)
        if trainer.is_global_zero:
            os.replace(tmp, self.path)
            print(f"[pause] step {trainer.global_step}: saved {self.path}", flush=True)
        self._last_save = self.clock()

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # position in the epoch, for every checkpoint (also ModelCheckpoint's best/last)
        checkpoint[RESUME_KEY] = {"epoch": self._epoch, "batches_done": self._batches_done,
                                  "global_step": trainer.global_step}


def _main_thread() -> bool:
    import threading
    return threading.current_thread() is threading.main_thread()


def _step(path: Path) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)
    return int(state.get("global_step", -1))


def latest_checkpoint(ckpt_dir: str | Path) -> Optional[Path]:
    """The checkpoint to resume from: the furthest of ``resume.ckpt`` and ``last.ckpt``
    (``last.ckpt`` on a tie: it was written after that step's validation)."""
    ckpt_dir = Path(ckpt_dir)
    found = [p for p in (ckpt_dir / "last.ckpt", ckpt_dir / "resume.ckpt") if p.exists()]
    if not found:
        return None
    return max(found, key=lambda p: (_step(p), p.name == "last.ckpt"))


def resume_position(checkpoint: Optional[str | Path]) -> Optional[tuple[int, int]]:
    """``(epoch, batches_done)`` stored by ``PauseCheckpoint`` in ``checkpoint`` (None if absent)."""
    if checkpoint is None:
        return None
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    pos = state.get(RESUME_KEY)
    return (int(pos["epoch"]), int(pos["batches_done"])) if pos else None
