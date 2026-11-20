"""Row-level evaluation of one or more of our runs scored on identical rows (Gates 1 and 2,
``docs/GATES_AGENT_PROMPT.md``). Pure numpy / scipy / sklearn on arrays the caller loads: no
data loading here (the analysis layer imports no other layer).

* all-row regression: Pearson, Spearman, MAE, RMSE, R² (original units);
* within true G4s (``y > label_threshold``: 35 K+PDS, 25 K+): Pearson and Spearman;
* per sequence class (:func:`classify_windows`, regexes on the 215-nt window as the model sees
  it, '+' rows reverse-complemented; first match wins): rows, true G4s, recall, mean residual;
* detection: precision / recall / Jaccard at a G4 call threshold tuned on the chr2 validation
  rows (max Jaccard, :func:`tune_threshold`), and AUPRC. **For comparisons between our own runs
  only**: never against TransG4's detection numbers (its notebook tunes on test labels).

The metric definitions match ``g4shape.lightning.metrics`` (the training-time metrics), which
the tests check. The paired genomic block bootstrap is not here yet.
"""

from __future__ import annotations

import re
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

# (class, regex on the window); first match wins; else "none". From
# G4_GNN_fast/analysis_scripts/scripts/analyze_run_forward.py.
MOTIFS = (
    ("canonical", re.compile(r"(?=(G{3,}[ACGTN]{1,7}G{3,}[ACGTN]{1,7}G{3,}[ACGTN]{1,7}G{3,}))")),
    ("long_loop", re.compile(r"(?=(G{3,}[ACGTN]{1,12}G{3,}[ACGTN]{1,12}G{3,}[ACGTN]{1,12}G{3,}))")),
    ("two_tetrad", re.compile(r"(?=(G{2,}[ACGTN]{1,7}G{2,}[ACGTN]{1,7}G{2,}[ACGTN]{1,7}G{2,}))")),
)
CLASSES = tuple(name for name, _ in MOTIFS) + ("none",)


def _f64(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).ravel()


def align_runs(runs: Mapping[str, tuple]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """``{name: (row_ids, pred)}`` -> (row_ids, {name: pred}); every run must cover the same rows
    in the same order (paired comparisons need identical rows)."""
    if not runs:
        raise ValueError("no runs given")
    ids, preds = None, {}
    for name, (rid, pred) in runs.items():
        rid, pred = np.asarray(rid).ravel(), _f64(pred)
        if rid.shape != pred.shape:
            raise ValueError(f"{name}: {rid.size} row ids but {pred.size} predictions")
        if ids is None:
            ids = rid
        elif not np.array_equal(rid, ids):
            raise ValueError(f"{name} was not scored on the same rows (same order) as the first run")
        preds[name] = pred
    return ids, preds


def pearson(pred, y) -> float:
    p, t = _f64(pred), _f64(y)
    pc, tc = p - p.mean(), t - t.mean()
    den = np.sqrt((pc ** 2).sum() * (tc ** 2).sum())
    return float((pc * tc).sum() / den) if den > 0 else 0.0


def spearman(pred, y) -> float:
    """Spearman rank correlation (ties: average ranks); 0 when undefined."""
    p, t = _f64(pred), _f64(y)
    if p.size < 2 or np.all(p == p[0]) or np.all(t == t[0]):
        return 0.0
    return float(spearmanr(p, t).statistic)


def regression(pred, y) -> dict[str, float]:
    """Pearson, Spearman, MAE, RMSE, R² (as ``lightning.metrics.regression_metrics``, + Spearman)."""
    p, t = _f64(pred), _f64(y)
    err = p - t
    ss_res, ss_tot = float((err ** 2).sum()), float(((t - t.mean()) ** 2).sum())
    return {"pearson": pearson(p, t), "spearman": spearman(p, t), "mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt(ss_res / t.size)), "r2": 1 - ss_res / ss_tot if ss_tot > 0 else 0.0,
            "n": int(t.size)}


def within_g4(pred, y, label_threshold: float) -> dict[str, float]:
    """Pearson and Spearman on the true G4s only (``y > label_threshold``)."""
    p, t = _f64(pred), _f64(y)
    m = t > label_threshold
    return {"pearson_g4": pearson(p[m], t[m]) if m.sum() > 1 else 0.0,
            "spearman_g4": spearman(p[m], t[m]), "n_g4": int(m.sum())}


def classify_windows(seqs: Sequence[str]) -> np.ndarray:
    """Class of each window (upper-cased; first matching motif in :data:`MOTIFS`, else "none")."""
    out = np.empty(len(seqs), dtype=object)
    for i, s in enumerate(seqs):
        s = s.upper()
        out[i] = next((name for name, rx in MOTIFS if rx.search(s)), "none")
    return out.astype(str)


def _prf(pred_pos: np.ndarray, label: np.ndarray) -> dict[str, float]:
    tp = float((pred_pos & label).sum())
    fp = float((pred_pos & ~label).sum())
    fn = float((~pred_pos & label).sum())
    r = lambda a, b: a / b if b > 0 else 0.0  # noqa: E731  (sklearn's zero_division=0)
    return {"precision": r(tp, tp + fp), "recall": r(tp, tp + fn), "jaccard": r(tp, tp + fp + fn)}


def tune_threshold(val_pred, val_y, label_threshold: float, grid: Sequence[float]) -> float:
    """Grid threshold maximising the Jaccard index on validation rows (first one on ties), as
    ``lightning.metrics.best_threshold``."""
    p, label = _f64(val_pred), _f64(val_y) > label_threshold
    best_t, best_j = float(grid[0]), -1.0
    for t in grid:
        j = _prf(p > t, label)["jaccard"]
        if j > best_j:
            best_t, best_j = float(t), j
    return best_t


def detection(pred, y, label_threshold: float, g4_threshold: float) -> dict[str, float]:
    """Precision / recall / Jaccard at ``pred > g4_threshold`` and AUPRC, against
    ``y > label_threshold``. Our own runs only (see the module docstring)."""
    p, label = _f64(pred), _f64(y) > label_threshold
    out = _prf(p > g4_threshold, label)
    out["auprc"] = float(average_precision_score(label, p)) if label.any() else 0.0
    out["g4_threshold"] = float(g4_threshold)
    return out


def per_class(pred, y, classes, label_threshold: float, g4_threshold: float) -> dict[str, dict]:
    """Per class: rows, true G4s, recall of the true G4s at ``g4_threshold``, mean residual
    (pred - y; negative = under-predicted)."""
    p, t, c = _f64(pred), _f64(y), np.asarray(classes).ravel()
    out = {}
    for name in CLASSES:
        m = c == name
        g4 = m & (t > label_threshold)
        out[name] = {"n": int(m.sum()), "n_g4": int(g4.sum()),
                     "recall": float((p[g4] > g4_threshold).mean()) if g4.any() else 0.0,
                     "mean_residual": float((p[m] - t[m]).mean()) if m.any() else 0.0}
    return out


def evaluate_rows(runs: Mapping[str, tuple], y, label_threshold: float,
                  g4_thresholds: Optional[Mapping[str, float]] = None,
                  classes: Optional[Sequence[str]] = None) -> dict:
    """All metrics for each run on identical rows.

    ``runs``: ``{name: (row_ids, pred)}``; ``y``: true scores of those rows; ``g4_thresholds``:
    each run's G4 call threshold, tuned on chr2 validation (:func:`tune_threshold`); without it
    detection and per-class recall are skipped. ``classes``: :func:`classify_windows` of the rows.
    """
    ids, preds = align_runs(runs)
    t = _f64(y)
    if t.size != ids.size:
        raise ValueError(f"{t.size} labels for {ids.size} rows")
    if classes is not None and len(classes) != ids.size:
        raise ValueError(f"{len(classes)} classes for {ids.size} rows")
    out = {"n_rows": int(ids.size), "label_threshold": float(label_threshold), "runs": {}}
    for name, p in preds.items():
        res = {"all": regression(p, t), "true_g4": within_g4(p, t, label_threshold)}
        g4_t = None if g4_thresholds is None else g4_thresholds.get(name)
        if g4_t is not None:
            res["detection"] = detection(p, t, label_threshold, g4_t)
            if classes is not None:
                res["per_class"] = per_class(p, t, classes, label_threshold, g4_t)
        out["runs"][name] = res
    return out
