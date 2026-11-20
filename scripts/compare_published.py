"""Compare one of our test results with the reported TransG4 / G4mismatch values (Gate 1). Run
from the repo root, in env g4gnn:

    python -m scripts.compare_published runs/pds_r0/test_metrics.json --condition PDS
    python -m scripts.compare_published runs/eval/fast_K_gatv2_pos_fullchr1.json --condition K

* the protocol guard checks the run's test rows against the **complete** index of the condition
  (``data/g4seq_PDS`` / ``data/g4seq_K``), never against the index the run recorded: a run on a
  ``_fast`` subset would otherwise pass (its own test split is the 2M sample);
* the label threshold is read from the run's ``config.yaml`` (next to ``test_metrics.json``) or
  from an eval file's ``protocol``, and must match the condition;
* writes ``<run>/published_comparison.json``, or ``runs/eval/<name>.published_comparison.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import yaml

from g4shape.analysis.published import ProtocolError, compare

COMPLETE_INDEX = {"K": "data/g4seq_K", "PDS": "data/g4seq_PDS"}


def _label_threshold(path: Path, result: dict) -> Optional[float]:
    if isinstance(result.get("protocol"), dict) and result["protocol"].get("label_threshold") is not None:
        return float(result["protocol"]["label_threshold"])
    cfg = path.parent / "config.yaml"
    if cfg.exists():
        t = (yaml.safe_load(cfg.read_text()) or {}).get("task", {}).get("label_threshold")
        return None if t is None else float(t)
    return None


def _out_path(path: Path) -> Path:
    if path.name == "test_metrics.json":
        return path.parent / "published_comparison.json"
    return path.with_name(f"{path.stem}.published_comparison.json")


def _table(out: dict) -> str:
    lines = [f"{'reference':<30} {'metric':<8} {'ours':>8} {'published':>9} {'delta':>8}  better",
             "-" * 74]
    for r in out["rows"]:
        lines.append(f"{r['reference']:<30} {r['metric']:<8} {r['ours']:>8.4f} {r['published']:>9.3f} "
                     f"{r['delta']:>+8.4f}  {'yes' if r['better'] else 'no'}")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("result", help="runs/<run>/test_metrics.json or runs/eval/<name>.json")
    p.add_argument("--condition", required=True, choices=sorted(COMPLETE_INDEX))
    p.add_argument("--index-dir", help="complete index to check the test rows against "
                                       "(default: data/g4seq_<condition>)")
    p.add_argument("--allow-sample", action="store_true",
                   help="compare a sampled test set anyway, labelled 'NOT comparable'")
    a = p.parse_args(argv)
    path = Path(a.result)
    result = json.loads(path.read_text())
    index_dir = a.index_dir or COMPLETE_INDEX[a.condition]
    try:
        out = compare(result, a.condition, index_dir, label_threshold=_label_threshold(path, result),
                      allow_sample=a.allow_sample)
    except ProtocolError as e:
        p.error(str(e))
    out["result"] = str(path)
    print(f"{path}  ({out['paper_condition']}, {out['test_n_samples']:,} of "
          f"{out['index_test_rows']:,} test rows)")
    if out["note"]:
        print(f"*** {out['note']} ***")
    print(_table(out))
    dest = _out_path(path)
    dest.write_text(json.dumps(out, indent=1))
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
