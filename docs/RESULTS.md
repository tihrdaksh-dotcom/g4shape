# Results

Rule: fill in the **Predictions** table and commit it *before* the runs start. Fill in the
**Results** table from `runs/<run_name>/test_metrics.json` afterwards; never edit a prediction
after its run has started.

## R0 baseline (phase 6)

Configs: `configs/pds_r0.yaml`, `configs/k_r0.yaml` (30,442 trainable parameters). Two seeds
each (`--set trainer.seed=<seed> --set trainer.run_name=<run>`). Test split: 2,000,000 chr1
windows.

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
