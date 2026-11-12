"""Direct binary G4 classification, BCE loss: the alternative to TransG4's regression.

TransG4 does not train a classifier: it regresses the mismatch score and thresholds it
(``regression_module.py``, the default task). This class trains the same network on the
G4 label directly. Training loop, G4-detection protocol, analysis and optimizer:
``LitG4Model`` (base.py). Only the classification objective is defined here:

* target = G4 label: ``batch.y > label_threshold`` (35, TransG4's G4 definition;
  ``G4WindowDataset`` yields the float score), or ``batch.y`` itself when
  ``label_threshold=None`` (labels already 0/1);
* ``postprocess`` = sigmoid, so predictions are probabilities;
* per-batch ``{split}_acc`` at probability 0.5;
* G4 calls: probability > ``detection_threshold``, tuned on validation over
  ``threshold_grid`` (0.01..0.99) exactly like the regressor's score threshold, so both
  tasks are judged by the same protocol (G4 windows are rare, and a fixed 0.5 cut-off is
  rarely the best one under class imbalance). ``threshold_grid=(0.5,)`` fixes it at 0.5.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from src.lightning.base import COMPONENTS, LitG4Model
from src.lightning.metrics import G4_LABEL_THRESHOLD

__all__ = ["LitClassificationModel", "PROBABILITY_GRID"]

PROBABILITY_GRID = tuple(i / 100 for i in range(1, 100))  # 0.01 .. 0.99


class LitClassificationModel(LitG4Model):
    def __init__(
        self,
        graph_encoder,
        global_encoder,
        fusion,
        classifier,
        optim: Optional[Mapping[str, Any]] = None,
        label_threshold: Optional[float] = G4_LABEL_THRESHOLD,
        threshold_grid: Sequence[float] = PROBABILITY_GRID,
    ):
        super().__init__(graph_encoder, global_encoder, fusion, classifier, optim,
                         label_threshold, threshold_grid, initial_threshold=0.5)
        self.save_hyperparameters(ignore=COMPONENTS)

    def g4_labels(self, y: torch.Tensor) -> torch.Tensor:
        return y > 0.5 if self.label_threshold is None else y > self.label_threshold

    def target(self, batch) -> torch.Tensor:
        return self.g4_labels(batch.y.float().view(-1)).float()

    def loss(self, out, target, weight=None):
        if weight is None:
            return F.binary_cross_entropy_with_logits(out, target)
        w = weight.to(out.dtype).view(-1)
        return (w * F.binary_cross_entropy_with_logits(out, target, reduction="none")).sum() / w.sum()

    def postprocess(self, out):
        return torch.sigmoid(out)

    def step_metrics(self, out, target):
        return {"acc": ((out > 0).float() == target).float().mean()}
