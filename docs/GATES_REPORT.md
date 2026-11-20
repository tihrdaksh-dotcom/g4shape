# Gates report

The two go/no-go gates of `docs/GATES_AGENT_PROMPT.md` (version 2, 2026-11-20). Predictions and
the pre-registered decision rules are in `docs/RESULTS.md` (committed 2026-11-20, before any gate
run). Every number here must trace to a file under `runs/` or `runs/eval/`.

## Status (2026-11-20)

**Nothing has been measured yet.** Gate 1 tooling is in place and tested; no checkpoint has been
scored on every chr1 row. Gate 2 has not started.

| Gate 1 item (prompt step) | State | Where |
|---|---|---|
| 1. configs on the complete dataset | done | `configs/pds_r0.yaml`, `configs/k_r0.yaml`: `data/g4seq_{PDS,K}`, all train rows, every chr1 row, fixed 1M chr2 validation subset |
| data links | done (not in git) | `data/g4seq_PDS`, `data/g4seq_K` on laptop and workstation; `index.json` counts match the prompt's table on both |
| 2. published-reference module + guard | done | `src/g4shape/analysis/published.py`, `tests/test_published.py` |
| 3. comparison script | done | `scripts/compare_published.py` (guard always against the complete index) |
| 4. row-level evaluation | done | `src/g4shape/analysis/evaluate_rows.py`, `tests/test_evaluate_rows.py` |
| 4. paired genomic block bootstrap | done | `block_bootstrap` in `evaluate_rows.py` (100-kb chr1 blocks, 1,000 draws; ~4 s on 30M rows) |
| 4. per-row predictions | done | `scripts/evaluate.py --save-predictions` → `runs/eval/<name>.predictions.npy` |
| 5. score existing checkpoints on every chr1 row | **to do** (workstation GPU) | commands below |
| 6. Gate 1 table | **to do** | section 1 |

## 1. Gate 1: protocol parity

Our checkpoints, scored on **every chr1 bin**, against the values **reported** in the TransG4 and
G4mismatch papers (regression metrics only; the published values carry no interval). Δ = ours −
published.

Commands (prompt step 5; workstation, in `tmux`, env `g4gnn`; `--retune` tunes the G4 call
threshold on the 1M chr2 validation subset of the config, regression metrics do not depend on it):

```bash
python -m scripts.evaluate ~/G4_GNN_fast/runs/fast_PDS_posonly/checkpoints/best.ckpt \
    --config configs/pds_r0.yaml --name fast_PDS_posonly_fullchr1 --retune --save-predictions
python -m scripts.evaluate ~/G4_GNN_fast/runs/fast_PDS_gatv2_pos_graphonly/checkpoints/best.ckpt \
    --config configs/pds_r0.yaml --name fast_PDS_gatv2_pos_graphonly_fullchr1 --retune --save-predictions
python -m scripts.evaluate ~/G4_GNN_fast/runs/fast_K_gatv2_pos/checkpoints/best.ckpt \
    --config configs/k_r0.yaml --name fast_K_gatv2_pos_fullchr1 --retune --save-predictions
python -m scripts.compare_published runs/eval/<name>_fullchr1.json --condition {PDS|K}
```

Intervals: `block_bootstrap` on `runs/eval/<name>_fullchr1.predictions.npy`; `vs_published` for
whether each published value lies inside our 95 % interval.

### K+PDS

| Checkpoint | R [95 % CI] | MAE [95 % CI] | RMSE [95 % CI] | R² [95 % CI] | Δ R TransG4 (0.941) | Δ R G4mismatch (0.91) |
|---|---|---|---|---|---|---|
| `fast_PDS_posonly` | | | | | | |
| `fast_PDS_gatv2_pos_graphonly` | | | | | | |

### K+

| Checkpoint | R [95 % CI] | MAE [95 % CI] | RMSE [95 % CI] | R² [95 % CI] | Δ R TransG4 (0.873) | Δ R G4mismatch (0.84) |
|---|---|---|---|---|---|---|
| `fast_K_gatv2_pos` | | | | | | |

Measured gap: *pending*.

## 2. Gate 2: input fidelity and learning curves

**Not started.** Order of work (prompt, Gate 2): `cnn_g4mm` (matched 1-D CNN; must land within
±0.01 of the reported G4mismatch R before any graph arm is trained) → `r0_plus` input fix, one
opt-in setting per commit (lattice tokenizer, learned nucleotide encoder, position at readout) →
throughput measurement and time estimate (needs the user's go-ahead) → learning-curve runs → K+
replication.

Tables to fill: all rows; true G4s; per class (canonical / long_loop / two_tetrad / none);
learning curves (R and within-G4 Spearman vs fraction); leave-one-out attributions.

## 3. Verdict

*Pending.* Applied mechanically from the pre-registered rules in `docs/RESULTS.md`.

## 4. Deviations from the prompt

1. **Commit `9c79cad` (workstation, 2026-11-20)** bundled four changes in one commit, left two
   tests failing (`tests/test_layering.py`) and contained placeholder code: a "Spearman" that
   returned Pearson, a bootstrap that returned fixed intervals (−0.01, +0.01), a true-G4 cutoff of
   `y > 0`, and `scripts/score_reference.py`, which aimed to load the published models (the prompt
   uses the reported values instead). It was repaired step by step in `be1b347` … `8dad7bd`; the
   placeholder bootstrap was removed and replaced, never used for a number.
2. **Predictions replaced before any run.** `9c79cad` wrote predictions to `docs/RESULTS.md` that
   targeted reproducing the published models and compared across conditions. They were replaced in
   `71a1791` (2026-11-20). No gate run had started at either commit, so pre-registration holds.
3. **Per-row predictions file.** One structured `.npy` per run (fields `row`, `start`, `y`,
   `pred`) rather than separate arrays: the fields cannot drift out of alignment.

## 5. Threats to validity

- **Single test chromosome** (chr1): the intervals cover resampling within chr1, not variation
  between chromosomes.
- **Seed count**: existing checkpoints are one seed each; Gate 2 uses 2 (3 where compute allows).
- **Published values have no interval** and come from the authors' own pipelines (TF 2.3); our
  windows, strand handling and labels follow the papers but are reimplemented.
- **Featurization differences between arms** (graph vs one-hot) in Gate 2.
- **Compute caps**: the fixed 1M-row chr2 validation subset (not all of chr2) for early stopping
  and threshold tuning; the 20-epoch cap for small fractions.
- **Existing checkpoints were trained on the weighted `_fast` subset**, not on all training rows:
  Gate 1 measures them as they are; full-data training is Gate 2's 100 % fraction.
