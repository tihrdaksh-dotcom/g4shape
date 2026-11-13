# Decisions

One line per cut of the cleaning (docs/CLEANING_PLAN.md, now in git history), with the reason
and the commit. Goldens were frozen at tag `v0.1-golden`; none was regenerated and `RENAME` in
`tests/test_golden.py` stayed empty (no weight was renamed).

## Cuts

- 2026-11-13 Removed the classification task (BCE module, `task.task`, the classifier-only `step_metrics` hook): regression on the mismatch score is the only task. `29b8c58`
- 2026-11-13 Removed the auxiliary G4 loss (`aux_g4_weight`, `aux_g4_pos_weight`): the objective is plain, optionally weighted, MSE. `9910347`
- 2026-11-13 Removed `models/utils.py` and `predict_g4`: one helper with no caller, and a duplicate of the `is_g4` field that `predict_step` and `characterize` return. `characterize` stays for the interpretability analyses. `d399130`
- 2026-11-13 Removed the `seq_cnn` and `mlp` global encoders and the one-hot window path (`sequence_branch`, `seq_window`): neither serves the G-run graph model. `86d2fd3`
- 2026-11-13 Removed the anchor node and `anchor_gat`: target-bin position is carried by `position_encoding`. `6f96154`
- 2026-11-13 Removed the one-hot run context (`onehot_context`): run flanks are summarised by `context_bases`; `gap_encoding` and `loop_end_bases` stay as the loop-class features of the tolerance experiment. `0ab50d8`
- 2026-11-13 Removed GATv2 attention (`attention`, `attention_dim`): the encoder keeps DADGNN's GATv1 scoring; this also removed the 3 known test failures. `490b43c`
- 2026-11-13 Added `GraphModel` (encoder -> LayerNorm -> head), the single-branch R0 network, 30,442 trainable parameters. `b2a332b`
- 2026-11-13 Switched the Lightning module, `build_lit_model`, `build_model` and `ModelConfig` to `GraphModel`; dropped `global_encoder`, `fusion` and their kwargs. `02e998a`
- 2026-11-13 Removed the topology branch from the data path (`features/topology.py`, `topo_clouds`, `step_size`, `max_step`, `use_topology`, `global_features`), with the KmerTopology dependency; renamed the configs to `pds_r0` / `k_r0` and turned on `position_encoding` in both. `23e4b5c`
- 2026-11-13 Deleted the multi-branch stage machinery (`multimodel.py`, `fusion/`, `global_encoders/`); the factory has two registries. `6d1fa46`
- 2026-11-13 Reduced the embeddings interface to `x_graph`, `penultimate` and `prediction`. `fa19a0d`
- 2026-11-13 Wrote every setting out in the R0 configs (identical model and initial weights). `a60d74c`
- 2026-11-13 Added `environment.yml` with only the imported packages. `10d26cf`

## Choices made during the cleaning

- 2026-11-13 The head keeps its class name `BinaryMLPClassifier` and the stage name `classifier` (config key, factory registry, checkpoints); the optional rename to `MLPHead` was skipped.
- 2026-11-13 `GraphModel`'s head argument is named `classifier`, not `head`, so `GraphModel(**build_components(...))` works and checkpoint keys stay `model.classifier.*`.
- 2026-11-13 The docstring of `scripts/make_golden.py` still mentions `global_features` (it explains why golden A does not hash it); kept on purpose, so the phase 4 grep shows that one line.
