# g4shape: aggressive cleaning plan, for execution with Claude Code

## The end state

One model, regression only:

```
G-run graph  ->  diffusion-attention encoder  ->  weighted-sum readout  ->  LayerNorm  ->  MLP head
```

The finished repository has no topology vector, no second branch, no fusion stage, no classification task and no
GATv2. Every file in `src/g4shape` can be justified in one line against `docs/FOCUS.md`.

## Rules for every phase

1. **One phase per branch.** Start with `git switch main && git pull && git switch -c <branch>`.
2. **Scope fence.** Each phase lists the files it may touch. If Claude Code wants to edit anything else, it must stop
   and ask.
3. **Tests before and after.** Run `python -m pytest -q` before the first edit and after every commit. Write down the
   summary line each time.
4. **Small commits.** One commit per sub-step. Read `git diff --stat` before each one.
5. **Golden files are frozen.** After phase 1, nobody regenerates anything in `tests/golden/`. If a golden test fails,
   the code is wrong. The only exception is a change you decided in advance and recorded in `docs/DECISIONS.md`.
6. **Finish a phase** with:
   ```bash
   git switch main && git merge --no-ff <branch> && git push && git branch -d <branch>
   ```
7. **Keep the two configs loading.** The config loader rejects unknown keys, so a phase that removes a setting also
   removes that key from `configs/*.yaml` in the same commit.
8. **Stop and ask** if a test that passed before now fails, if the parameter count changes, or if a deleted name
   still turns up in a `grep`.

Start every Claude Code session with:
> Read CLAUDE.md, docs/FOCUS.md and docs/CLEANING_PLAN.md. We are doing phase N. Explain your plan for this phase and
> list the files you will touch before changing anything.

---

## Phase 0: housekeeping (done; one leftover)

Phase 0 is merged and pushed. The only leftover is the merged branch:

```bash
git branch -d chore/housekeeping
```

Also check that the GitHub repository `tihrdaksh-dotcom/g4shape` is private.

## Phase 1: golden tests

The old phase 1 (porting the temporary single-branch modules) is dropped: phase 4 would delete them again, and no
trained model exists whose behaviour must be kept.

**Branch:** `test/golden`. **May touch:** `tests/golden/` (new), `tests/test_golden.py` (new),
`scripts/make_golden.py` (new).

**What to pin.** These three things do not change during the cleaning, so freezing them catches accidental changes:

| Golden | Contents | Why it survives the cleaning |
|---|---|---|
| A. Features | SHA-256 of `x`, `edge_index`, `edge_attr` and `y` for (i) the synthetic genome in `tests/conftest.py`, and (ii) 64 fixed chr1 test windows of `data/g4seq_PDS_fast`, using `FeatureConfig(min_run_len=2, edge_radius=50, context_bases=2, flank_window=20, position_encoding=True)` written out in the script (not read from a config file, because the configs change during cleaning) | The G-run graph is kept. `global_features` is not hashed, because it is removed |
| B. Encoder | A `diffusion_gat` encoder built with the posonly settings: its `state_dict`, one fixed input batch, and its output embedding | The encoder class is kept (only the GATv2 option goes) |
| C. Head | The `binary_mlp` head: its `state_dict`, a fixed input tensor of shape [8, 64], and its output | The head class is kept |

**Prompt:**
> Write scripts/make_golden.py, which builds goldens A, B and C as described in phase 1 of docs/CLEANING_PLAN.md and
> saves them under tests/golden/ (hashes as JSON, tensors and state_dicts as .pt, each file under 1 MB). Then write
> tests/test_golden.py, which recomputes A, B and C and compares them: hashes exactly, tensors with atol 1e-6. Load the
> saved state_dicts by parameter name through a RENAME = {} table at the top of the test, and never rely on a random
> seed. Skip the real-data part of A with a clear message when data/ is missing. Run make_golden.py once, show me the
> hashes, then run the tests.

**Checks:**
1. `python -m pytest -q tests/test_golden.py` passes.
2. Prove the test can fail. Change `_MAX_LEN_SCALE = 12` to `13` in `features/g_run_graph.py`. Golden A must fail.
   Undo the change with `git restore src/g4shape/features/g_run_graph.py`.
3. The full suite still shows 3 known failures.

**Commits:** `Add golden-file generator`, `Add golden files and tests (frozen)`.
**Tag:** `git tag -a v0.1-golden -m "Goldens frozen"` (and `git push --tags`).

## Phase 2: dead options

**Branch:** `purge/dead-options`. **May touch:** `lightning/`, `config.py`, `train.py`, `models/utils.py`, and the
tests that use them.

| Remove | Where |
|---|---|
| The classification task | Delete `lightning/classification_module.py`. Remove `TASKS` and the `task` switch in `lightning/build.py`, `TaskConfig.task` in `config.py`, and the `task: regression` line in both configs |
| The auxiliary G4 loss | Remove `aux_g4_weight`, `aux_g4_pos_weight` from `config.py`, `train.py`, `lightning/build.py` and `lightning/regression_module.py` |
| `models/utils.py` | Delete the file (one helper with no caller) |
| `predict_g4` | Delete it from `lightning/base.py`. **Keep `characterize`**: it exposes attention and readout weights, which the interpretability analyses need |

**Tests to edit:**
- `test_lightning_base.py`: remove `test_classifier_tunes_probability_threshold_on_val`, and the classification half
  of the "both tasks" tests.
- `test_regression_module.py`: remove `test_classification_binarises_score` and `test_aux_g4_loss_is_opt_in`.
- `test_weighted_subset.py`: remove `test_weighted_classification_loss`.
- The other test files that set `task` (`test_diffusion_dadgnn.py`, `test_train.py`, `test_window_dataset.py`):
  remove the setting.

**Prompt:**
> Do phase 2 of docs/CLEANING_PLAN.md. One commit for the classification task, one for the auxiliary loss, one for
> models/utils.py and predict_g4.

**Checks:**
1. Tests pass apart from the 3 known failures. Golden tests pass. `python -m g4shape.train --config
   configs/fast_pds_posonly.yaml --print-config` still works.
2. `grep -rniE "classif|aux_g4|predict_g4|count_parameters" src scripts tests` finds only the `binary_mlp` head and
   its class name `BinaryMLPClassifier`. Renaming that class to `MLPHead` is allowed in this phase, as a separate
   commit, if you want it.

## Phase 3: unused branches and options

**Branch:** `purge/branches`. **May touch:** `models/`, `features/`, `data/featurize.py`, and the matching tests.

| Remove | Where |
|---|---|
| The `seq_cnn` and `mlp` global encoders | Delete both files and their registry lines. Delete `SEQ_CHANNELS`, `_onehot_window` and the `sequence_branch` path in `data/featurize.py`; delete `sequence_branch` and `seq_window` from `features/config.py` |
| The anchor node | Delete `graph_encoders/anchor_gat.py` and its registry line. Remove `anchor_node` from `features/config.py` and `features/graph_extras.py` (`is_anchor`, `to_anchor`, `from_anchor`, `anchor_index`) and the anchor branch in `featurize` |
| One-hot run context | Remove `onehot_context` from `features/config.py` and `features/graph_extras.py` |
| GATv2 | Remove `ATTENTION_TYPES`, `attention` and `attention_dim` from `diffusion_conv.py` and `diffusion_gat.py`. Keep only the default (GATv1) scoring path. Delete `tests/test_gatv2_attention.py` |

**Keep** `gap_encoding` (loop-length buckets) and `loop_end_bases`: they are the loop-class features for the tolerance
experiment.

**Tests to edit:**
- `test_graph_extras.py`: remove `test_anchor_structure`, `test_anchor_gat_forward`,
  `test_anchor_gat_rejects_graphs_without_anchor`, and the context half of `test_onehot_context_and_loop_ends` (keep
  the loop-ends half).
- `test_stages.py`: drop the `seq_cnn` and `mlp` cases.
- `test_factory.py`: drop the cases that use removed names.

**Prompt:**
> Do phase 3 of docs/CLEANING_PLAN.md. Commit in this order: seq_cnn and mlp; anchor node; onehot_context; GATv2.
> Run the full tests after each commit.

**Checks:**
1. **Zero test failures from here on.** The 3 known failures disappear with GATv2 and seq_cnn. Golden tests pass.
2. `grep -rniE "seq_cnn|sequence_branch|seq_window|anchor|onehot_context|gatv2|attention_dim" src scripts tests configs`
   prints nothing.

## Phase 4: topology branch and fusion stage

The largest phase; plan two Claude Code sessions.

**Branch:** `purge/topology`. **May touch:** everything under `src/` and `tests/`, plus `scripts/evaluate.py`.

**Sub-steps, one commit each:**

1. **New model.** Add `models/graph_model.py` with `GraphModel(graph_encoder, head)`:
   `forward = head(LayerNorm(graph_encoder(...)))`, and `return_details` gives `x_graph`, `penultimate` and the
   encoder details. Add `tests/test_graph_model.py`: shapes; output finite; trainable parameters exactly **30,442**
   with the posonly feature settings; gradients reach the encoder and the head.
2. **Switch the Lightning module to it.**
   - `lightning/base.py`: `COMPONENTS = ["graph_encoder", "classifier"]`; build a `GraphModel`.
   - `lightning/build.py` and `train.build_model`: only graph-encoder and head arguments.
   - `config.py`: remove `global_encoder`, `fusion` and their `*_kwargs` from `ModelConfig`.
3. **Remove the topology branch from the data path.**
   - Delete `features/topology.py`.
   - Remove `topo_clouds`, `step_size`, `max_step` and `G3_4MERS` from `features/config.py`.
   - `featurize`: no `global_features`, no `use_topology` argument.
   - `feature_dims`: only `node_dim` and `edge_dim`.
   - Remove `use_topology` from `DataConfig`, `window_dataset.py`, `sequence_dataset.py`, `train.py` and
     `scripts/evaluate.py`.
   - Configs: `git mv configs/fast_pds_posonly.yaml configs/pds_r0.yaml` and
     `git mv configs/fast_k.yaml configs/k_r0.yaml` (git then records a rename, not a delete and add). Remove
     `use_topology`, `global_encoder`, `fusion`, their `*_kwargs`, `topo_clouds`, `step_size` and `max_step`. Set
     `run_name` to `pds_r0` / `k_r0`. In `k_r0.yaml` set `position_encoding: true`, so both are the R0 model.
4. **Delete the old stage machinery.**
   - Delete `models/multimodel.py`, `models/fusion/`, `models/global_encoders/`.
   - `models/stages.py`: keep only `GraphEncoder` and `Head`.
   - `models/factory.py`: two registries and `build_components(node_dim, edge_dim, hidden_dim, ...)` returning
     `{graph_encoder, classifier}`.
   - Delete `tests/test_topology_encoder.py` and `tests/test_fusion_head.py`; remove the topology tests in
     `test_features_equivalence.py` and `test_featurize.py`.
5. **Interpretability interface.** `analysis/embeddings.py`: drop `x_global`, `gate`, `fused` and `clouds`; keep
   `x_graph`, `penultimate`, `prediction`.

If a golden test fails after sub-step 2 or 4 only because a parameter name changed, add the old -> new name to
`RENAME` in `tests/test_golden.py`. That is the only golden edit allowed.

**Prompt (session 1):**
> Do phase 4 sub-steps 1 and 2 of docs/CLEANING_PLAN.md.

**Prompt (session 2):**
> Do phase 4 sub-steps 3, 4 and 5.

**Checks:**
1. All tests pass, including the golden tests, with zero failures.
2. `python -c "import sys, g4shape.train; print([m for m in ('KmerTopology','gudhi','ripser') if m in sys.modules])"`
   prints `[]`.
3. `grep -rniE "topolog|topo_|global_enc|global_feat|global_dim|global_layout|fusion|gated|multimodel|use_topology" src scripts tests`
   prints nothing.
4. **CPU dry run:**
   ```bash
   python -m g4shape.train --config configs/pds_r0.yaml \
     --set data.max_train=512 --set data.max_val=256 --set data.max_test=256 --set data.num_workers=0 \
     --set trainer.accelerator=cpu --set trainer.devices=1 --set trainer.run_name=dry_r0
   ```
   It must write `runs/dry_r0/test_metrics.json` and a `best.ckpt` that `g4shape.train.load_trained` can reload.

## Phase 5: configs, dependencies and records

**Branch:** `chore/records`. **May touch:** `configs/`, `environment.yml` (new), `README.md`, `docs/`.

1. **Configs.** Make every value in `pds_r0.yaml` and `k_r0.yaml` explicit: data paths, feature settings with `position_encoding: true`, encoder settings, head, task label
   (35 for PDS, 25 for K+), `y_mean`/`y_std` and the threshold grid, optimiser, trainer, resources. Carry no
   topology or fusion keys.
2. **`environment.yml`:** only the packages the code imports (python 3.11, pytorch, pytorch_geometric,
   pytorch-lightning, numpy, pandas, scipy, scikit-learn, pyyaml, pytest), with no KmerTopology, gudhi or ripser.
   Check: `grep -rhoE "^(import|from) [a-z_]+" src scripts tests | sort -u`.
3. **`README.md`:** what the model is (the one-line diagram above), install
   (`conda env create -f environment.yml && pip install -e .`), test (`python -m pytest -q`) and train
   (`python -m g4shape.train --config configs/pds_r0.yaml`).
4. **Records:**
   - `docs/ROADMAP.md`: the four experiments.
   - `docs/RESULTS.md`: a predictions section, filled in before each run.
   - `docs/DECISIONS.md`: one dated line per cut, with the reason and the commit hash. Take the hashes from
     `git log --oneline v0.1-golden..HEAD`.
5. **Remove `docs/CLEANING_PLAN.md`** in the last commit. It is finished, and git history keeps it.

**Checks:**
1. A fresh environment from `environment.yml`, plus `pip install -e .`, passes all tests.
2. Every file in `src/g4shape` has a one-line reason in `docs/DECISIONS.md` or in `README.md`'s layout section.

**Tag:** `git tag -a v1-core -m "Purged single-branch model" && git push --tags`.

## After cleaning (phase 6, the first experiment)

On the GPU box: `git clone` (or `git pull`) and `pip install -e .`. Train R0 on PDS and K+ with two seeds each.
Bring `runs/` back with rsync. Commit the predictions to `docs/RESULTS.md` before the runs start.

## How to tell the cleaning is finished

| Check | Expected |
|---|---|
| `python -m pytest -q` | all pass, 0 failed |
| Topology packages imported by `g4shape.train` | none |
| R0 trainable parameters | 30,442 |
| `grep` for removed names (phases 2-4) | no matches |
| `git log --oneline v0-import..v1-core` | reads as a list of reasons |
| Python files under `src/g4shape` | 29, down from 37 (9 deleted, `graph_model.py` added) |

## If something goes wrong

| Situation | Do this |
|---|---|
| Uncommitted edits you don't want | `git restore <file>` (one file) or `git restore .` (all) |
| A commit you regret, not yet pushed | `git revert <hash>` (adds an undo commit; history stays honest) |
| Lost track mid-phase | `git stash`, then `git switch main`; the branch is still there |
| Need the pre-cleaning code | `git switch --detach v0-import` to look; `git switch main` to return |
| Golden test fails | Stop. Find which sub-step broke it with `git bisect` or by checking out earlier commits; do not regenerate |
