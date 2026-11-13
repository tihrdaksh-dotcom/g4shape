"""Shared training/evaluation loop of the G4 Lightning modules.

The network (``MultiModalModel``) is judged at the end by "is this window a G4?". The
objective is defined by four hooks, implemented by ``LitRegressionModel``:

* ``target(batch)``      what the head output is compared to by the loss,
* ``loss(out, target)``  the objective,
* ``postprocess(out)``   head output -> prediction in task units (score / probability),
* ``epoch_metrics``      extra task metrics on the whole epoch (e.g. Pearson R),

and those hooks are all the subclass defines. Everything else lives here, once:
building the network, the train/val/test steps and logging, collecting a whole epoch of
predictions (Pearson, R2, AUPRC are not averages of per-batch values), the G4-detection
protocol, analysis (``characterize``), inference (``predict``, ``predict_step``) and
the optimizer (``optim`` settings, default = TransG4's recipe, see ``optim.py``).

G4 detection protocol (TransG4, see ``src.lightning.metrics``): true label = ``batch.y``
> ``label_threshold`` (35); the prediction threshold is a buffer, ``detection_threshold``,
re-tuned at the end of every real validation epoch (max Jaccard over
``threshold_grid``), saved in checkpoints, and reused unchanged on the test split.

Multi-GPU (DDP): the per-epoch predictions are gathered from all ranks first, so the
epoch metrics and the tuned threshold describe the whole split, identically on every rank.

Logged per split: ``{split}_loss``; per val/test epoch: ``{split}_{precision, recall,
jaccard, auprc, g4_threshold, n_samples}`` plus the task's ``epoch_metrics``.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import pytorch_lightning as pl
import torch

from g4shape.lightning.metrics import best_threshold, score_detection
from g4shape.lightning.optim import configure_optimizers, resolve_optim
from g4shape.models.multimodel import MultiModalModel

__all__ = ["LitG4Model", "to_cpu"]

COMPONENTS = ["graph_encoder", "global_encoder", "fusion", "classifier"]


def to_cpu(obj):
    """Detach + move every tensor of a (nested dict / list) structure to the CPU."""
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_cpu(v) for v in obj]
    return obj


class LitG4Model(pl.LightningModule):
    """Abstract base; use ``LitRegressionModel``."""

    progress_metric: Optional[str] = None  # epoch metric shown in the progress bar

    def __init__(
        self,
        graph_encoder,
        global_encoder,
        fusion,
        classifier,
        optim: Optional[Mapping[str, Any]],
        label_threshold: Optional[float],
        threshold_grid: Sequence[float],
        initial_threshold: float,
    ):
        super().__init__()
        self.model = MultiModalModel(graph_encoder=graph_encoder, global_encoder=global_encoder,
                                     fusion=fusion, classifier=classifier)
        self.optim = resolve_optim(optim)  # TransG4 recipe unless overridden (lightning/optim.py)
        self.label_threshold = None if label_threshold is None else float(label_threshold)
        self.threshold_grid = tuple(float(t) for t in threshold_grid)
        # prediction threshold for "is a G4", in postprocess() units; tuned on validation
        self.register_buffer("detection_threshold",
                             torch.tensor(float(initial_threshold), dtype=torch.float32))
        self._buf: dict[str, list] = {"val": [], "test": []}

    # ---- task hooks -------------------------------------------------------------------
    def target(self, batch) -> torch.Tensor:
        """[B] tensor the loss compares the head output to."""
        raise NotImplementedError

    def loss(self, out: torch.Tensor, target: torch.Tensor,
             weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Batch loss; ``weight`` [B] = per-sample weights (``batch.sample_weight`` of a weighted
        index split), None = all 1. Weighted losses are self-normalised: sum(w*l) / sum(w)."""
        raise NotImplementedError

    def postprocess(self, out: torch.Tensor) -> torch.Tensor:
        """Head output [B] -> prediction [B] in task units."""
        raise NotImplementedError

    def g4_labels(self, y: torch.Tensor) -> Optional[torch.Tensor]:
        """True G4 labels [B] (bool) from raw ``batch.y``; None disables detection metrics."""
        return None if self.label_threshold is None else y > self.label_threshold

    def epoch_metrics(self, pred: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
        """Optional task metrics on a whole val/test epoch (``pred`` = postprocess units)."""
        return {}

    # ---- network, inference, analysis -------------------------------------------------
    def forward(self, batch, return_details: bool = False):
        return self.model(batch, return_details=return_details)  # head output [B, 1] (+ details)

    @torch.no_grad()
    def _eval_forward(self, batch, return_details: bool = False):
        """Deterministic forward (eval mode: no dropout, BatchNorm running stats, no grad);
        the caller's train/eval mode is restored afterwards."""
        was_training = self.training
        self.eval()
        try:
            return self(batch, return_details=return_details)
        finally:
            self.train(was_training)

    def predict(self, batch) -> torch.Tensor:
        """Prediction in task units, shape [B] (eval mode, no grad)."""
        return self.postprocess(self._eval_forward(batch).view(-1))

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        """``trainer.predict``: {"prediction", "is_g4"} per batch."""
        pred = self.predict(batch)
        return {"prediction": pred, "is_g4": pred > self.detection_threshold}

    def characterize(self, batch):
        """Eval-mode forward for analysis, on CPU: {"output" [B, 1] raw head output,
        "prediction" [B], "is_g4" [B], "details"}; details["graph"] holds per-layer/head/hop
        attention, diffusion coefficients, node embeddings and readout weights (see
        DiffusionGATEncoder). Eval mode, no grad, like ``predict``."""
        out, details = self._eval_forward(batch, return_details=True)
        pred = self.postprocess(out.view(-1))
        return to_cpu({"output": out, "prediction": pred,
                       "is_g4": pred > self.detection_threshold, "details": details})

    # ---- training loop ----------------------------------------------------------------
    def _step(self, batch, prefix: str):
        out = self(batch).view(-1)
        target = self.target(batch)
        loss = self.loss(out, target, getattr(batch, "sample_weight", None))
        bs = out.numel()
        on_step = prefix == "train"
        self.log(f"{prefix}_loss", loss, on_step=on_step, on_epoch=True, prog_bar=True, batch_size=bs,
                 sync_dist=self._distributed)
        if prefix != "train":
            self._buf[prefix].append((self.postprocess(out).detach().cpu(),
                                      batch.y.float().view(-1).detach().cpu()))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    @property
    def _distributed(self) -> bool:
        trainer = getattr(self, "_trainer", None)
        return trainer is not None and trainer.world_size > 1

    def _gather(self, t: torch.Tensor) -> torch.Tensor:
        """Concatenate a 1-D per-rank tensor over all DDP ranks (sizes may differ by rank)."""
        if not self._distributed:
            return t
        t = t.to(self.device)
        n = self.all_gather(torch.tensor([t.numel()], device=self.device)).view(-1)
        padded = torch.zeros(int(n.max()), dtype=t.dtype, device=self.device)
        padded[:t.numel()] = t
        allp = self.all_gather(padded).reshape(len(n), -1)
        return torch.cat([allp[r, :int(k)] for r, k in enumerate(n)]).cpu()

    def _epoch_end(self, prefix: str):
        chunks, self._buf[prefix] = self._buf[prefix], []
        pred = torch.cat([p for p, _ in chunks]) if chunks else torch.zeros(0)
        y = torch.cat([t for _, t in chunks]) if chunks else torch.zeros(0)
        # DDP: every rank must join the gather; metrics + threshold then use the whole split
        # (identical on all ranks; DistributedSampler may repeat a few samples to even out)
        pred, y = self._gather(pred), self._gather(y)
        if pred.numel() == 0:
            return
        self.log(f"{prefix}_n_samples", float(pred.numel()))  # what the epoch metrics cover
        for name, value in self.epoch_metrics(pred, y).items():
            self.log(f"{prefix}_{name}", value.float(), prog_bar=name == self.progress_metric)

        label = self.g4_labels(y)
        if label is None:
            return
        trainer = getattr(self, "_trainer", None)
        if prefix == "val" and not (trainer is not None and trainer.sanity_checking):
            self.detection_threshold.fill_(best_threshold(pred, label, self.threshold_grid))
        thr = float(self.detection_threshold)
        for name, value in score_detection(pred, label.float(), thr, label_threshold=None).items():
            self.log(f"{prefix}_{name}", value.float(), prog_bar=name == "jaccard")
        self.log(f"{prefix}_g4_threshold", thr)

    def on_validation_start(self):
        self._buf["val"] = []

    def on_validation_epoch_end(self):
        self._epoch_end("val")

    def on_test_epoch_end(self):
        self._epoch_end("test")

    def configure_optimizers(self):
        return configure_optimizers(self.parameters(), self.optim)
