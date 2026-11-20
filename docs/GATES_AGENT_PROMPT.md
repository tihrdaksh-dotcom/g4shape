# Prompt for the coding agent: run the two go/no-go gates in ~/g4shape and reach a conclusion

Paste everything below the line into Claude Code, started in `~/g4shape`.

Version 2 (2026-11-20): Gate 1 now uses the complete dataset (every chr1 row) and the values reported in the TransG4
and G4mismatch papers as baselines. Reproducing or retraining the published models is no longer required.

---

You are working in the repository `~/g4shape` (GitHub `tihrdaksh-dotcom/g4shape`, HEAD `bac6b5c`, tag `v1-core`).
Read `CLAUDE.md`, `README.md`, `docs/FOCUS.md`, `docs/DECISIONS.md` and `docs/RESULTS.md` first, and follow
`CLAUDE.md` exactly:
- teaching mode: say what you will do and why, show the command, wait for my go-ahead;
- one change per commit;
- `python -m pytest -q` before and after every change;
- never edit outside the repo; no force-push, no hard reset;
- one comprehension question after each step.

The golden tests (`tests/golden/`) must keep passing: every new behaviour is an **opt-in setting whose default
reproduces today's R0**. Never regenerate the golden files.

## Goal

Two gates decide whether the G-run graph project continues. Run them and write a conclusion, with numbers, in
`docs/GATES_REPORT.md`. Do not tune anything to make a gate pass. Write the predictions into `docs/RESULTS.md` and
commit them **before** any gate run starts.

- **Gate 1 (protocol parity):** what is the real gap between our model and the published models? Our checkpoints are
  scored on **every chr1 bin** of the complete dataset and compared with the values **reported** in the TransG4 and
  G4mismatch papers. The published models are **not** rebuilt or retrained.
- **Gate 2 (input fidelity + learning curves):** does fixing the graph's *input* (tokens and loop/tract encoding)
  bring the graph model to parity with a matched 1-D CNN at full data, and does the graph lead at small data?

## Facts you can rely on (verified; do not re-derive)

### Data and splits
- Full G4-seq indexes (GSE110582, hg19, 15-nt bins, window = bin ± 100 nt = 215 nt; chr1 = test, chr2 = val, rest
  = train) on the laptop at `/home/daksh/G4_GNN/data/g4seq_K` and `/home/daksh/G4_GNN/data/g4seq_PDS`. Genome store:
  `/home/daksh/G4_GNN/data/hg19`.

  | | train | val | test |
  |---|---|---|---|
  | K+ | 311,734,698 | 31,356,341 | 29,534,138 |
  | PDS | 315,437,985 | 31,652,070 | 29,923,093 |

- **The complete dataset replaces the sampled one.** I (the user) will add the full indexes to `data/` as new links,
  `data/g4seq_PDS` and `data/g4seq_K`, on the laptop and on the workstation.
  - Before using them, confirm they exist and that `index.json` counts match the table above. If they are missing,
    stop and ask me.
  - Keep the existing `data/g4seq_*_fast` links: the real-data golden test (`tests/test_golden.py`, golden A) reads
    `data/g4seq_PDS_fast`, and it must not change.
  - Up to now every number in this repo was measured on the `_fast` subsets: a weighted training subsample, and a
    2,000,000-row uniform chr1 test sample. From now on the test set is every chr1 row.
  - Sampling check already done: run1 (old baseline) scored R 0.8585 / 0.8111 (PDS / K+) on the 2M sample and
    0.859 / 0.809 on all chr1 bins. The sample effect is ≤ 0.002 for that model, but our current models have never
    been scored on all bins.
- Labels: G4 = score > 35 (PDS), > 25 (K+). '+' rows are reverse-complemented, as in TransG4 / G4mismatch.

### Earlier results, all on the 2M-row sample, not full chr1, one seed unless stated
PDS posonly (with topology branch) R 0.9115; gatv2_pos graph-only 0.8993; GATv2-only 0.8012; baseline 0.8545. A
full-data baseline (run1) scored on all chr1 bins gave R 0.859.

### Published references: the Gate 1 baselines, read from the papers; take these values, do not re-derive them

Every chr1 bin, held-out test chromosome; chr2 validation; trained on all other chromosomes, 1 epoch, MSE, Adam
lr 1e-3; mismatch score in % (raw scale).

| Model | Condition | MAE | RMSE | R² | Pearson R | Source |
|---|---|---|---|---|---|---|
| TransG4 | K+ | 0.837 | 1.762 | 0.761 | 0.873 | Yuan et al. 2026 PCCP, 10.1039/d6cp00173d, Table 2 |
| TransG4 | K+PDS | 1.228 | 2.767 | 0.885 | 0.941 | same, Table 3 |
| G4mismatch | K+ | 0.92 | 1.97 | 0.70 | 0.84 | TransG4 Table 1; R also in Barshai et al. 2023 PLoS CB, 10.1371/journal.pcbi.1010948 |
| G4mismatch | K+PDS | 1.47 | 3.29 | 0.84 | 0.91 | same |
| CNN-BiGRU (TransG4 ablation) | K+ / K+PDS | 0.868 / 1.276 | 1.811 / 2.878 | 0.748 / 0.876 | 0.867 / 0.936 | TransG4 Tables 2–3 |
| BiGRU (TransG4 ablation) | K+ / K+PDS | 0.852 / 1.275 | 1.773 / 2.865 | 0.759 / 0.877 | 0.871 / 0.937 | TransG4 Tables 2–3 |

Rules for using them:
- Compare **regression metrics only**. Pearson R is primary; MAE / RMSE / R² are secondary. Our `test_metrics.json`
  already reports them on the raw mismatch-% scale (`RegressionModule.postprocess` undoes the standardisation): check
  this once.
- Do **not** compare detection metrics (precision / recall / Jaccard) with TransG4's. Its paper says predictions are
  thresholded at 25 / 35, but its released notebook was found earlier to sweep the threshold on test labels.
- The published values have no reported seed spread or interval. Report our value with its interval and the
  published value as a fixed reference.
- G4-Attention's mismatch-regression numbers are only in its PCCP 2026 version, which is not in `ref_papers/`. Leave
  it out unless I provide that PDF.

### Released checkpoints (TensorFlow 2.3): OPTIONAL, only if I ask for it later
- `github.com/OrensteinLab/G4mismatch`: `models/model_K.h5`, `models/model_KPDS.h5`. Architecture: Conv1D(256, k = 95,
  ReLU) → GlobalMaxPool → Dense(32, ReLU) → Dense(1, linear); 105,793 parameters. N = 0.25 per channel. Read
  `mm_utils.py` for the one-hot channel order and `mm_gen.py` for the strand flip (`seq[::-1, ::-1]` on '+' rows).
- `github.com/M-nianjj/TransG4-deep-learning-approach-for-sequence-based-G-quadruplex-prediction`:
  `G4/models/Transformer_kernelsize_15_K` (SavedModel, K+ only). Architecture: Conv1D(64, 15) → BN → transformer block
  (64-d, 4 heads, FF 256) → BiGRU(64) → Dense(32) → Dense(1); 108,097 trainable parameters.
- raw.githubusercontent.com may return a proxy 502 here. The GitHub contents API (`api.github.com/repos/.../contents/
  <path>` with header `Accept: application/vnd.github.raw`) works.

### Measured facts motivating the input fix (PDS chr1)
- Lone G's are dropped (`min_run_len 2`), so bulged tracts are invisible to R0.
- Among two-tetrad-class windows, a bulge-tolerant three-tetrad reading is associated with label > 35: 9.4 % vs 2.0 %.
  Odds ratio 2.4–3.3 after adjusting for G count and G-run count.
- Loops between G3+ runs: median 4 nt, 25 % are 1 nt.
- Loop features are four base fractions, so base order is lost. Literature: an A at loop position 1 is destabilizing
  (Guédin 2009, NAR).
- The encoder's value path is linear between layers (`activation: null`, `residual: false`, no norm).

**Gate 2 changes the input only; it leaves the operator as is.**

## Gate 1: protocol parity (complete dataset vs reported values)

No reference model is rebuilt, retrained or loaded: the baselines are the reported values above.

1. **Configs on the complete dataset** (one commit). In `configs/pds_r0.yaml` and `configs/k_r0.yaml`:
   - `data.index_dir: data/g4seq_PDS` / `data/g4seq_K` (the complete indexes);
   - `max_train: null` (all training rows);
   - `max_test: null` (every chr1 row);
   - `max_val: 1000000` with `subsample_seed: 0`. Validation, used for early stopping and threshold tuning, is a fixed
     1M-row uniform chr2 subset, because a full chr2 pass every validation is too costly. Write the reason in the
     config comment.

   Rewrite the comment block that describes the weighted `_fast` subset. Keep `task.y_mean / y_std`: they are already
   the full-train-split statistics. The full index has no weight column, so training is unweighted MSE, as in
   TransG4 / G4mismatch; confirm that the loader and loss handle an unweighted index. Run the tests.
2. **Published-reference module** (one commit), e.g. `src/g4shape/analysis/published.py`. A constant table holding
   exactly the values above, with model, condition, metric, value, source DOI and table number. A function
   `compare(test_metrics, condition, index_dir)` returns one row per (reference, metric): ours, published, and Δ.
   - **Protocol guard:** it refuses (raises) unless the run's `test_n_samples` is at least 0.999 × the test count in
     `<index_dir>/index.json`. So a run scored on the 2M sample can never be presented next to published numbers.
     An explicit `allow_sample=True` labels the output "NOT comparable: sampled test set".
   - **Condition check:** the condition must match the run's `task.label_threshold` (35 = K+PDS, 25 = K+) when the
     run's `config.yaml` is available.
   - **Tests** (no real data): the table has both conditions and all four metrics; the guard refuses
     n = 2,000,000 against a fake index of 29,923,093 and accepts the full count; Δ is computed correctly.
3. **Script** `scripts/compare_published.py <runs/<run>/test_metrics.json | runs/eval/<name>.json> --condition
   {K,PDS}`. It prints a table and writes `<run>/published_comparison.json`.
4. **Row-level evaluation module and paired genomic block bootstrap** (one or two commits), e.g.
   `src/g4shape/analysis/evaluate_rows.py`. It takes (row ids, predictions) for one or more of our runs on identical
   rows and reports:
   - all-row Pearson, Spearman, MAE, RMSE, R²;
   - Pearson and Spearman within true G4s (y > 35 K+PDS, y > 25 K+);
   - per-class recall and mean residual. Classes are regexes on the 215-nt window, first match wins, taken from
     `G4_GNN_fast/analysis_scripts/scripts/analyze_run_forward.py`:

     | Class | Regex |
     |---|---|
     | canonical | `(?=(G{3,}[ACGTN]{1,7}G{3,}[ACGTN]{1,7}G{3,}[ACGTN]{1,7}G{3,}))` |
     | long_loop | the same with `{1,12}` loops |
     | two_tetrad | `G{2,}` tracts with `{1,7}` loops |
     | none | otherwise |
   - detection precision / recall / Jaccard (threshold tuned on the chr2 validation subset, max Jaccard) and AUPRC,
     **for comparisons between our own runs only**.

   For the bootstrap, `evaluate.py` gets an option to save per-row predictions with row ids (`.npy`). `evaluate.py` gets an option to save per-row
   predictions with row ids (`.npy`). The bootstrap resamples 100-kb blocks on chr1 by bin start, 1,000 times, and
   gives 95 % intervals for our R, RMSE, MAE and R², and for differences between our runs on identical rows.
   - Against a published value, report our interval and whether it contains the reference value; there is no paired
     test, since we have no per-row predictions for the published models.
   - Unit test on synthetic data: identical predictions give Δ = 0 with zero width.
5. **Score existing checkpoints on every chr1 row**, with no retraining:
   - `G4_GNN_fast/runs/fast_PDS_posonly`, `fast_PDS_gatv2_pos_graphonly`, `fast_K_gatv2_pos`, via
     `python -m scripts.evaluate <ckpt> --config <pds_r0|k_r0>.yaml --name <name>_fullchr1`;
   - then `scripts/compare_published.py` on each.

   These numbers replace the 2M-sample ones in `docs/RESULTS.md`; keep the old ones, labelled "2M sample".
6. **Gate 1 output:** a table of our best existing checkpoint vs TransG4 and G4mismatch (R, MAE, RMSE, R², each with
   our 95 % interval and Δ to the published value), per condition. This is the measured gap. There is no pass/fail:
   it is the starting point for Gate 2.

## Gate 2: input fidelity and learning curves (PDS first; K+ replication at the end)

### Arms
All in PyTorch in this repo, same training recipe, parameter counts reported. Every new feature is an opt-in config
setting, default off.

1. **`cnn_g4mm`: the matched 1-D model.** A PyTorch reimplementation of the G4mismatch CNN (Conv1d(4 → 256, k = 95,
   padding 'same'), ReLU, global max-pool, Linear(256 → 32), ReLU, Linear(32 → 1); about 105.8k parameters). Same
   one-hot, N = 0.25. Add it as a new model in the factory with its own dataset path (one-hot windows); it does not
   touch the graph path.
   - **Check:** trained on all PDS training rows (and later K+), it lands within ±0.01 of the **reported**
     G4mismatch Pearson on every chr1 row: 0.91 K+PDS, 0.84 K+. This validates our training recipe and data pipeline
     against the paper without loading their checkpoint. If it misses, stop and find the cause before training the
     graph arms.
2. **`r0`:** today's `configs/pds_r0.yaml`, unchanged.
3. **`r0_plus`: the input fix.** Three changes, each its own setting and its own commit, introduced in this order with
   tests:
   1. **Lattice tokenizer** (`tokenizer: lattice`; default `strict`). Nodes are:
      - maximal G-runs of length ≥ 2;
      - **plus** one-bulge readings G{a}XG{b} (a, b ≥ 1, a + b ≥ 3, X a single non-G);
      - **plus** split readings of long runs, G{p} G G{q} with p, q ≥ 3; a run may split more than once.

      Overlapping readings are linked by an edge type `overlap`. Node columns: n_G, is_bulged, bulge position
      (5′ / mid / 3′), is_split.

      Tests:
      - on `TGAGGGTGGGTAGGGTGGGTAA`, the strict runs are kept, so the 1-nt loops survive;
      - on `TTGTGGTGGGTGGGTGGGT`, the reading GTGG appears;
      - with the setting off, features are byte-identical to golden.
   2. **Learned local nucleotide encoder** (`nuc_encoder: cnn`; default `none`). A small dilated Conv1d stack over the
      one-hot window, receptive field ≤ 21 nt, about 32 channels.
      - Node feature: mean and max of the per-nucleotide embedding over the tract span, plus the embeddings at the two
        tract boundaries.
      - Edge feature for consecutive tracts: mean and max over the loop span, plus the first and last loop nucleotide
        embeddings; empty loop gives zeros.
      - Keep the existing hand features.

      This makes loop base order and flanks learnable without hand features. It runs on the GPU inside the model:
      pass the one-hot window and the span indices from the featurizer.
   3. **Position only at the readout** (`position_at_readout: true`). Node and edge features carry no absolute offset;
      the weighted-sum gate gets an additive term k(offset) from the 16 RBFs of the node's signed offset from the bin.
      Gap (relative distance) stays on edges.

   Leave the diffusion-GAT operator unchanged in this gate. Target about 100–110k parameters, by `hidden_dim` and
   encoder width, to match `cnn_g4mm`.
4. **Leave-one-out arms at the 10 % fraction only:** `r0_plus` minus lattice; minus nucleotide encoder; minus
   position-at-readout. They attribute any gain.

### Learning curves
- Training fractions: uniform random subsets of the **full** PDS training rows, at 0.1 %, 1 %, 10 % and 100 %, with a
  fixed seed per fraction shared by all arms.
- Validation: a fixed 1,000,000-row uniform chr2 subset, used for early stopping (patience 3 evaluations) and threshold
  tuning.
- Fractions below 100 % train up to convergence, capped at 20 epochs; 100 % trains 1 epoch, as the comparators did.
  Report the steps actually taken.
- Seeds: 2 per arm and fraction; 3 for `cnn_g4mm` and `r0_plus` at 100 % if compute allows.
- Test: **every chr1 row**, scored with the Gate 1 evaluation module, paired.
- Matrix: `cnn_g4mm` and `r0_plus` at all four fractions; `r0` at 1 % and 10 %; leave-one-out arms at 10 %.
- K+ replication: `cnn_g4mm` and `r0_plus` at 10 % and 100 %, 2 seeds.

### Pre-registered decision rules
Commit these to `docs/RESULTS.md` before training.

1. **Parity at full data.** Let Δ = R(r0_plus) − R(cnn_g4mm) on all chr1 rows at 100 %, with the interval from the
   paired block bootstrap combined with seed variance.
   - "Comparable" if the 90 % interval of Δ lies above −0.010 (one-sided equivalence).
   - "Worse" if the 95 % interval lies below 0.
2. **Small-data advantage.** At 0.1 % or 1 %, r0_plus beats cnn_g4mm on all-row R or on within-true-G4 Spearman, with
   the 95 % interval excluding 0.
3. **Verdict.**
   - **GO** if (1) is comparable or better in PDS and holds in the K+ replication, **and** (2) holds.
   - **NO-GO** if cnn_g4mm ≥ r0_plus at every fraction (Δ ≤ 0 with the interval not excluding 0 in r0_plus's favour
     anywhere). The graph-as-better-predictor claim is then dropped.
   - **INCONCLUSIVE** otherwise. The only allowed remedy is more seeds, not new arms.
4. Also report, without using them in the verdict: r0_plus vs r0 (does the input fix help?); r0_plus (and cnn_g4mm) at 100 % vs the **reported** TransG4 and
   G4mismatch values in both conditions (`scripts/compare_published.py`); per-class recall (expectation: the largest gain is in two_tetrad); and the leave-one-out
   attributions.

## Engineering rules

- Throughput first: before any full run, measure windows/s for featurization and training, on the workstation, for
  each arm. Report estimated hours per run and the total, and wait for my go-ahead.
- Long runs go in `tmux` on the workstation (`scripts/sync.sh push|pull|status`, as in the README). Outputs go to
  `runs/<name>/` with config, git commit, seed, parameter count, wall time and per-row test predictions.
- Use the existing `--set section.key=value` overrides. Unknown keys must stay errors.
- Figures as **SVG** in `docs/figures/gates/`:
  1. Gate 1 bar chart of R with intervals, ours vs references;
  2. learning curves: R and within-G4 Spearman vs fraction (log x), per arm, with seed points and intervals;
  3. per-class recall at 100 %.

## Deliverable: `docs/GATES_REPORT.md`

1. Gate 1 table and the measured gap.
2. Gate 2 tables: all rows, true G4s, classes; learning curves; leave-one-out attributions.
3. The verdict (GO / NO-GO / INCONCLUSIVE), applying the pre-registered rules mechanically, with every number traced
   to a `runs/` or `runs/eval/` file.
4. Deviations from this prompt, if any, each with a reason.
5. Threats to validity: single test chromosome, seed count, featurization differences between arms, compute caps.

Do not change the decision rules after seeing results. If something forces a deviation, write it down before
proceeding and ask me.
