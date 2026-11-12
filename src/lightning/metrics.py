"""Epoch-level metrics shared by the Lightning modules (all on 1-D tensors, original units).

Regression: MAE, RMSE, R2, Pearson R (TransG4 Table 1).

G4 detection (TransG4 Fig. 3, ``G4/train.ipynb`` cells 13-14): a window is a G4 when its
G4-seq mismatch score is ``> label_threshold`` (35). The regressor's predicted score is
binarised with a second threshold, chosen by sweeping a grid (TransG4: 30.0..34.9, step
0.1) for the best Jaccard index; precision / recall / Jaccard are reported at that
threshold. TransG4 swept on its test chromosome; here the threshold is chosen on the
validation split and then applied unchanged to the test split. AUPRC is threshold-free
(the predicted score ranks the windows).
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = ["G4_LABEL_THRESHOLD", "G4_LABEL_THRESHOLDS", "TRANSG4_THRESHOLD_GRID", "WIDE_GRID",
           "resolve_threshold_grid", "regression_metrics", "detection_metrics", "best_threshold",
           "average_precision", "score_detection"]

G4_LABEL_THRESHOLD = 35.0
G4_LABEL_THRESHOLDS = {"K": 25.0, "PDS": 35.0}  # per condition (TransG4 p. 8892, G4mismatch)
TRANSG4_THRESHOLD_GRID = tuple(30.0 + 0.1 * i for i in range(50))  # 30.0 .. 34.9
WIDE_GRID = {"start": 5.0, "stop": 45.0, "step": 0.1}  # tuned on val; contains both labels


def resolve_threshold_grid(spec) -> Optional[tuple[float, ...]]:
    """Threshold grid from a config value: None (the task's default), a list of thresholds, or
    ``{"start", "stop", "step"}`` (both ends included)."""
    if spec is None:
        return None
    if isinstance(spec, dict):
        if set(spec) != {"start", "stop", "step"}:
            raise ValueError(f"threshold grid dict needs exactly start, stop, step; got {sorted(spec)}")
        start, stop, step = float(spec["start"]), float(spec["stop"]), float(spec["step"])
        if step <= 0 or stop < start:
            raise ValueError(f"bad threshold grid {spec}")
        n = int(round((stop - start) / step)) + 1
        return tuple(round(start + step * i, 6) for i in range(n))
    grid = tuple(float(t) for t in spec)
    if not grid:
        raise ValueError("threshold grid is empty")
    return grid


def regression_metrics(pred: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
    """MAE, RMSE, R2 and Pearson R of two 1-D tensors (original units)."""
    pred, y = pred.double().flatten(), y.double().flatten()
    err = pred - y
    ss_res = (err ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    pc, yc = pred - pred.mean(), y - y.mean()
    denom = (pc.pow(2).sum() * yc.pow(2).sum()).sqrt()
    return {
        "mae": err.abs().mean(),
        "rmse": (ss_res / y.numel()).sqrt(),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else torch.zeros(()),
        "pearson": (pc * yc).sum() / denom if denom > 0 else torch.zeros(()),
    }


def _ratio(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    return num / den if den > 0 else torch.zeros((), dtype=torch.float64)


def detection_metrics(pred_pos: torch.Tensor, label: torch.Tensor) -> dict[str, torch.Tensor]:
    """Precision, recall and Jaccard of boolean predictions vs boolean labels (0 when undefined,
    as sklearn's ``zero_division=0``)."""
    pred_pos, label = pred_pos.bool().flatten(), label.bool().flatten()
    tp = (pred_pos & label).sum().double()
    fp = (pred_pos & ~label).sum().double()
    fn = (~pred_pos & label).sum().double()
    return {"precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
            "jaccard": _ratio(tp, tp + fp + fn)}


def best_threshold(score: torch.Tensor, label: torch.Tensor, grid=TRANSG4_THRESHOLD_GRID) -> float:
    """Grid threshold on ``score`` maximising the Jaccard index (first one on ties)."""
    best_t, best_j = float(grid[0]), -1.0
    for t in grid:
        j = float(detection_metrics(score > t, label)["jaccard"])
        if j > best_j:
            best_t, best_j = float(t), j
    return best_t


def average_precision(score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """AUPRC as sklearn's ``average_precision_score``: sum over distinct thresholds of
    (recall step) x precision; 0 when there is no positive."""
    score, label = score.double().flatten(), label.bool().flatten()
    n_pos = label.sum()
    if n_pos == 0:
        return torch.zeros((), dtype=torch.float64)
    order = torch.argsort(score, descending=True)
    score, label = score[order], label[order].double()
    # last index of every run of equal scores = one threshold
    last = torch.nonzero(torch.diff(score, append=score[-1:] - 1) != 0).flatten()
    tps = torch.cumsum(label, 0)[last]
    precision = tps / (last + 1).double()
    recall = tps / n_pos
    return (torch.diff(recall, prepend=recall.new_zeros(1)) * precision).sum()


def score_detection(score: torch.Tensor, y: torch.Tensor, threshold: float,
                    label_threshold: Optional[float] = G4_LABEL_THRESHOLD) -> dict[str, torch.Tensor]:
    """Precision / recall / Jaccard at ``score > threshold`` and AUPRC, against
    ``y > label_threshold`` (``label_threshold=None``: ``y`` is already a 0/1 label)."""
    label = y.bool() if label_threshold is None else y > label_threshold
    out = detection_metrics(score > threshold, label)
    out["auprc"] = average_precision(score, label)
    return out
