# Results

Rule: fill in the **Predictions** table and commit it *before* the runs start. Fill in the
**Results** table from `runs/<run_name>/test_metrics.json` afterwards; never edit a prediction
after its run has started.

## R0 baseline (phase 6)

Configs: `configs/pds_r0.yaml`, `configs/k_r0.yaml` (30,442 trainable parameters). Two seeds
each (`--set trainer.seed=<seed> --set trainer.run_name=<run>`). Test split: 2,000,000 chr1
windows (2M sample).

### Predictions (commit before the runs start)

| Run | Condition | Seed | Test Pearson | Test Jaccard | Reasoning |
|---|---|---|---|---|---|
| `pds_r0_s42` | K+ + PDS | 42 | | | |
| `pds_r0_s43` | K+ + PDS | 43 | | | |
| `k_r0_s42` | K+ | 42 | | | |
| `k_r0_s43` | K+ | 43 | | | |

Date committed:

### Results

| Run | Test Pearson | Test R2 | Test RMSE | Test Jaccard | Test AUPRC | G4 threshold | Commit |
|---|---|---|---|---|---|---|---|
| `pds_r0_s42` | | | | | | | |
| `pds_r0_s43` | | | | | | | |
| `k_r0_s42` | | | | | | | |
| `k_r0_s43` | | | | | | | |

## Earlier checkpoints (2M sample)

Scored on the 2,000,000-row uniform chr1 sample, one seed, from
`~/G4_GNN_fast/runs/<run>/test_metrics.json`. **Not comparable with published values** (sampled
test set). Gate 1 re-scores these on every chr1 row; the new numbers go in the Gate 1 results below.

| Run | Condition | Pearson R | MAE | RMSE | R² | Test rows |
|---|---|---|---|---|---|---|
| `fast_PDS_posonly` | K+PDS | 0.9115 | 1.481 | 3.375 | 0.830 | 2,000,000 |
| `fast_PDS_gatv2_pos_graphonly` | K+PDS | 0.8993 | 1.603 | 3.575 | 0.809 | 2,000,000 |
| `fast_K_gatv2_pos` | K+ | 0.8395 | 0.946 | 1.984 | 0.701 | 2,000,000 |

## Gates (`docs/GATES_AGENT_PROMPT.md`)

### Gate 1 predictions (commit before any checkpoint is scored on every chr1 row)

Gate 1 has no pass/fail; it measures the gap to the **reported** values (TransG4: Yuan et al. 2026
PCCP, Tables 2–3; G4mismatch: TransG4 Table 1 / Barshai et al. 2023). Reasoning: run1 moved by
≤ 0.002 between the 2M sample and all chr1 bins, so each checkpoint is expected within ±0.005 of
its 2M-sample R. Δ = ours − published; negative means we are behind.

| Checkpoint | Condition | Predicted R (all chr1) | vs TransG4 (R) | Predicted Δ | vs G4mismatch (R) | Predicted Δ |
|---|---|---|---|---|---|---|
| `fast_PDS_posonly` | K+PDS | 0.911 ± 0.005 | 0.941 | −0.030 | 0.91 | +0.001 |
| `fast_PDS_gatv2_pos_graphonly` | K+PDS | 0.899 ± 0.005 | 0.941 | −0.042 | 0.91 | −0.011 |
| `fast_K_gatv2_pos` | K+ | 0.840 ± 0.005 | 0.873 | −0.033 | 0.84 | 0.000 |

Expected MAE / RMSE / R² follow the same pattern: roughly at G4mismatch's level, behind TransG4's.

Date committed: 2026-11-20

### Gate 1 results

Fill from `runs/eval/<name>_fullchr1.json` and `published_comparison.json`, with 95 % intervals
from the paired genomic block bootstrap.

| Checkpoint | Condition | R [95 % CI] | MAE | RMSE | R² | Test rows | Δ R vs TransG4 | Δ R vs G4mismatch |
|---|---|---|---|---|---|---|---|---|
| `fast_PDS_posonly` | K+PDS | | | | | | | |
| `fast_PDS_gatv2_pos_graphonly` | K+PDS | | | | | | | |
| `fast_K_gatv2_pos` | K+ | | | | | | | |

### Gate 2 pre-registered decision rules (do not change after seeing results)

1. **Parity at full data.** Let Δ = R(r0_plus) − R(cnn_g4mm) on all chr1 rows at 100 %, with the
   interval from the paired block bootstrap combined with seed variance.
   - "Comparable" if the 90 % interval of Δ lies above −0.010 (one-sided equivalence).
   - "Worse" if the 95 % interval lies below 0.
2. **Small-data advantage.** At 0.1 % or 1 %, r0_plus beats cnn_g4mm on all-row R or on
   within-true-G4 Spearman, with the 95 % interval excluding 0.
3. **Verdict.**
   - **GO** if (1) is comparable or better in PDS and holds in the K+ replication, **and** (2) holds.
   - **NO-GO** if cnn_g4mm ≥ r0_plus at every fraction (Δ ≤ 0 with the interval not excluding 0 in
     r0_plus's favour anywhere). The graph-as-better-predictor claim is then dropped.
   - **INCONCLUSIVE** otherwise. The only allowed remedy is more seeds, not new arms.
4. Also reported, not used in the verdict: r0_plus vs r0; r0_plus and cnn_g4mm at 100 % vs the
   reported TransG4 and G4mismatch values in both conditions; per-class recall (expectation: the
   largest gain is in two_tetrad); the leave-one-out attributions.

### Gate 2 predictions (commit before any Gate 2 training starts)

| Quantity | Prediction | Reasoning |
|---|---|---|
| cnn_g4mm, 100 %, PDS: R | 0.91 ± 0.01 | It reimplements G4mismatch (reported 0.91); this is its validation check |
| cnn_g4mm, 100 %, K+: R | 0.84 ± 0.01 | Same, reported 0.84 |
| r0_plus, 100 %, PDS: R | 0.905 | Graph-only R0 was 0.899 (2M sample); bulge-aware tokens and learned loop encoding should add a little |
| Δ at 100 %, PDS | −0.005 | Close to parity, so rule 1 is likely borderline |
| Δ at 0.1 % / 1 %, PDS | > 0 | The graph's built-in G-run structure should help most when data is scarce |
| Largest per-class recall gain (r0_plus vs r0) | two_tetrad | Bulged tracts are invisible to R0 (lone G's dropped) |
| Verdict | INCONCLUSIVE | Parity is expected to be borderline at full data |

Date committed: 2026-11-20
