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
  weighted MSE sum(w*e^2) / sum(w), an estimate of the full-split MSE.

``batch.y`` must be the float score, as produced by ``G4WindowDataset``.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from g4shape.lightning.base import COMPONENTS, LitG4Model
from g4shape.lightning.metrics import G4_LABEL_THRESHOLD, TRANSG4_THRESHOLD_GRID, regression_metrics

__all__ = ["LitRegressionModel"]


class LitRegressionModel(LitG4Model):
    progress_metric = "pearson"

    def __init__(
        self,
        graph_encoder,
        classifier,
        optim: Optional[Mapping[str, Any]] = None,
        y_mean: float = 0.0,
        y_std: float = 1.0,
        label_threshold: Optional[float] = G4_LABEL_THRESHOLD,
        threshold_grid: Sequence[float] = TRANSG4_THRESHOLD_GRID,
    ):
        super().__init__(graph_encoder, classifier, optim,
                         label_threshold, threshold_grid,
                         initial_threshold=label_threshold or 0.0)  # until tuned: label threshold
        self.save_hyperparameters(ignore=COMPONENTS)
        self.y_mean, self.y_std = float(y_mean), float(y_std)

    def target(self, batch) -> torch.Tensor:
        return (batch.y.float().view(-1) - self.y_mean) / self.y_std

    def loss(self, out, target, weight=None):
        w = None if weight is None else weight.to(out.dtype).view(-1)
        return F.mse_loss(out, target) if w is None else (w * (out - target) ** 2).sum() / w.sum()

    def postprocess(self, out):
        return out * self.y_std + self.y_mean

    def epoch_metrics(self, pred, y):
        return regression_metrics(pred, y)
