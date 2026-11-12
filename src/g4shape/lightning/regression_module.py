"""Score regression (G4-seq mismatch score), MSE loss: the default task, as TransG4.

Training loop, G4-detection protocol, analysis and optimizer: ``LitG4Model`` (base.py).
This class only defines the regression objective:

* target = ``batch.y`` standardised with ``y_mean`` / ``y_std`` (the head predicts in
  z-space); ``postprocess`` maps it back, so predictions, metrics and the G4 threshold
  are all in original score units;
* epoch metrics: MAE, RMSE, R2, Pearson R (``{val,test}_{mae,rmse,r2,pearson}``), once
  per epoch from all predictions, because Pearson / R2 are not averages of batch values;
* G4 calls: predicted score > ``detection_threshold``, tuned on validation over
  ``threshold_grid`` (TransG4's 30.0..34.9); ``label_threshold=None`` disables them;
* per-sample weights (``batch.sample_weight``, from a subsampled index; ``scripts/make_subset.py``):
  weighted MSE sum(w*e^2) / sum(w), an estimate of the full-split MSE (same for the aux term);
* optional auxiliary G4 term (``aux_g4_weight`` > 0; off by default = TransG4's plain MSE):
  loss = MSE + w * BCE(out - z(label_threshold), y > label_threshold) with ``pos_weight``
  ``aux_g4_pos_weight``, on the same single output (1 z-unit = 1 logit unit). G4 windows
  are 0.3 % of the data, so plain MSE pulls peak predictions down; this term pushes the
  output across the label threshold for true G4s without a second head.

``batch.y`` must be the float score, as produced by ``G4WindowDataset``.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from src.lightning.base import COMPONENTS, LitG4Model
from src.lightning.metrics import G4_LABEL_THRESHOLD, TRANSG4_THRESHOLD_GRID, regression_metrics

__all__ = ["LitRegressionModel"]


class LitRegressionModel(LitG4Model):
    progress_metric = "pearson"

    def __init__(
        self,
        graph_encoder,
        global_encoder,
        fusion,
        classifier,
        optim: Optional[Mapping[str, Any]] = None,
        y_mean: float = 0.0,
        y_std: float = 1.0,
        label_threshold: Optional[float] = G4_LABEL_THRESHOLD,
        threshold_grid: Sequence[float] = TRANSG4_THRESHOLD_GRID,
        aux_g4_weight: float = 0.0,
        aux_g4_pos_weight: float = 1.0,
    ):
        super().__init__(graph_encoder, global_encoder, fusion, classifier, optim,
                         label_threshold, threshold_grid,
                         initial_threshold=label_threshold or 0.0)  # until tuned: label threshold
        self.save_hyperparameters(ignore=COMPONENTS)
        self.y_mean, self.y_std = float(y_mean), float(y_std)
        self.aux_g4_weight, self.aux_g4_pos_weight = float(aux_g4_weight), float(aux_g4_pos_weight)
        if self.aux_g4_weight and self.label_threshold is None:
            raise ValueError("aux_g4_weight needs a label_threshold")

    def target(self, batch) -> torch.Tensor:
        return (batch.y.float().view(-1) - self.y_mean) / self.y_std

    def loss(self, out, target, weight=None):
        w = None if weight is None else weight.to(out.dtype).view(-1)
        mse = F.mse_loss(out, target) if w is None else (w * (out - target) ** 2).sum() / w.sum()
        if not self.aux_g4_weight:
            return mse
        z_thr = (self.label_threshold - self.y_mean) / self.y_std
        bce = F.binary_cross_entropy_with_logits(
            out - z_thr, (target > z_thr).to(out.dtype),
            pos_weight=out.new_tensor(self.aux_g4_pos_weight), reduction="none")
        return mse + self.aux_g4_weight * (bce.mean() if w is None else (w * bce).sum() / w.sum())

    def postprocess(self, out):
        return out * self.y_std + self.y_mean

    def epoch_metrics(self, pred, y):
        return regression_metrics(pred, y)
