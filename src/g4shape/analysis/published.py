"""Reported regression metrics of the published G4-seq mismatch models, and a guarded comparison
of one of our runs against them (Gate 1, ``docs/GATES_AGENT_PROMPT.md``).

The published values were measured on **every chr1 bin** (held-out test chromosome; chr2
validation; trained on all other chromosomes), on the raw mismatch-% scale. They are read from
the papers and fixed here; no published model is rebuilt or loaded.

* regression metrics only (Pearson R primary; MAE / RMSE / R² secondary): detection metrics are
  not comparable (TransG4's released notebook tunes its threshold on test labels);
* the values carry no seed spread or interval: they are fixed references;
* protocol guard: :func:`compare` refuses a run scored on fewer than 0.999 x the test rows of
  its index (e.g. the 2M-row chr1 sample), unless ``allow_sample=True``, which labels the
  output "NOT comparable: sampled test set".

Typical use::

    rows = compare(json.load(open("runs/x/test_metrics.json")), "PDS", "data/g4seq_PDS",
                   label_threshold=35.0)["rows"]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

CONDITIONS = {"K": "K+", "PDS": "K+PDS"}           # --condition name -> paper name
LABEL_THRESHOLDS = {"K": 25.0, "PDS": 35.0}         # G4 = score > threshold
METRICS = ("pearson", "mae", "rmse", "r2")          # keys of our test_metrics (``test_<metric>``)
HIGHER_IS_BETTER = {"pearson": True, "r2": True, "mae": False, "rmse": False}
MIN_TEST_FRACTION = 0.999
NOT_COMPARABLE = "NOT comparable: sampled test set"

_TRANSG4 = "Yuan et al. 2026 PCCP, doi:10.1039/d6cp00173d"
_G4MM = ("TransG4 (Yuan et al. 2026 PCCP, doi:10.1039/d6cp00173d); R also in Barshai et al. "
         "2023 PLoS CB, doi:10.1371/journal.pcbi.1010948")


@dataclass(frozen=True)
class Reference:
    model: str
    condition: str          # "K" or "PDS"
    metric: str             # one of METRICS
    value: float
    source: str
    table: str


def _rows(model, source, tables, k, pds):
    """``k`` / ``pds`` = (MAE, RMSE, R², Pearson R), the order of the papers' tables."""
    out = []
    for cond, vals, table in (("K", k, tables[0]), ("PDS", pds, tables[1])):
        for metric, v in zip(("mae", "rmse", "r2", "pearson"), vals):
            out.append(Reference(model, cond, metric, v, source, table))
    return out


PUBLISHED: tuple[Reference, ...] = tuple(
    _rows("TransG4", _TRANSG4, ("Table 2", "Table 3"),
          (0.837, 1.762, 0.761, 0.873), (1.228, 2.767, 0.885, 0.941))
    + _rows("G4mismatch", _G4MM, ("TransG4 Table 1", "TransG4 Table 1"),
            (0.92, 1.97, 0.70, 0.84), (1.47, 3.29, 0.84, 0.91))
    + _rows("CNN-BiGRU (TransG4 ablation)", _TRANSG4, ("Table 2", "Table 3"),
            (0.868, 1.811, 0.748, 0.867), (1.276, 2.878, 0.876, 0.936))
    + _rows("BiGRU (TransG4 ablation)", _TRANSG4, ("Table 2", "Table 3"),
            (0.852, 1.773, 0.759, 0.871), (1.275, 2.865, 0.877, 0.937))
)


class ProtocolError(ValueError):
    """The run cannot be put next to the published values (sampled test set, wrong condition)."""


def references(condition: str) -> list[Reference]:
    """Published values of one condition ("K" or "PDS")."""
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {sorted(CONDITIONS)}, got {condition!r}")
    return [r for r in PUBLISHED if r.condition == condition]


def index_test_rows(index_dir: str | Path) -> int:
    """Number of test rows (every chr1 bin) recorded in ``<index_dir>/index.json``."""
    return int(json.loads((Path(index_dir) / "index.json").read_text())["counts"]["test"])


def _flat_metrics(test_metrics: dict) -> dict:
    """Accept a run's ``test_metrics.json`` or a ``runs/eval/<name>.json`` (metrics nested)."""
    return test_metrics["metrics"] if isinstance(test_metrics.get("metrics"), dict) else test_metrics


def compare(test_metrics: dict, condition: str, index_dir: str | Path,
            label_threshold: Optional[float] = None, allow_sample: bool = False) -> dict:
    """One row per (published model, metric): ours, published, ``delta`` = ours - published,
    ``better`` (ours better, by the metric's direction).

    Raises :class:`ProtocolError` if the run's ``test_n_samples`` is below 0.999 x the test
    rows of ``index_dir`` (unless ``allow_sample``) or if ``label_threshold`` (the run's
    ``task.label_threshold``, when known) does not match ``condition``.
    """
    refs = references(condition)
    m = _flat_metrics(test_metrics)
    if label_threshold is not None and float(label_threshold) != LABEL_THRESHOLDS[condition]:
        raise ProtocolError(f"condition {condition!r} expects label_threshold "
                            f"{LABEL_THRESHOLDS[condition]}, the run used {label_threshold}")
    n_ours = int(m["test_n_samples"])
    n_index = index_test_rows(index_dir)
    comparable = n_ours >= MIN_TEST_FRACTION * n_index
    if not comparable and not allow_sample:
        raise ProtocolError(f"run scored on {n_ours:,} test rows, {index_dir} has {n_index:,}: "
                            f"not the published protocol (every chr1 bin); pass allow_sample=True "
                            f"to compare anyway, labelled {NOT_COMPARABLE!r}")
    rows = []
    for r in refs:
        ours = float(m[f"test_{r.metric}"])
        delta = ours - r.value
        rows.append({"reference": r.model, "metric": r.metric, "ours": ours, "published": r.value,
                     "delta": delta, "better": delta > 0 if HIGHER_IS_BETTER[r.metric] else delta < 0,
                     "source": r.source, "table": r.table})
    return {"condition": condition, "paper_condition": CONDITIONS[condition],
            "comparable": comparable, "note": None if comparable else NOT_COMPARABLE,
            "test_n_samples": n_ours, "index_test_rows": n_index, "index_dir": str(index_dir),
            "rows": rows}
