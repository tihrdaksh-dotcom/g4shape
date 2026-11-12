# g4shape: aggressive cleaning plan, for execution with Claude Code

Checked against `~/g4shape` at commit `58d91a9` (tag `v0-import`) on 2026-11-12. Nothing in your repository was changed;
the checks ran on a scratch clone, since deleted.

## Where the repository stands now

- 64 tracked files: 37 source files, 20 test files, 2 configs, 2 scripts. In the scratch clone, 294 tests pass and 4 fail.
- Three failures are the known ones: two GATv2 config tests, one `seq_cnn` stage test.
- The fourth is new: `tests/test_stages.py` still expects the text `src.models.stages` in two error messages.
  The rename missed `match="..."` strings.
- `CLAUDE.md` is untracked. It points to `docs/FOCUS.md`, which does not exist yet.
- History contains two commits named "Rename package...". `ad799d2` committed a stray copy of the old `src/*` files and
  `58d91a9` removed it. The final tree is correct. Leave the history as it is: rewriting shared history causes more
  trouble than a visible mistake, and the mistake is a useful lesson in reading `git show --stat` before committing.
- The repository has no list of dependencies. `environment.yml` in G4_GNN_fast installs `KmerTopology` (and with it
  gudhi and ripser) only for the branch being cut.

## Target state after cleaning

The model becomes **G-run graph -> diffusion-attention encoder -> weighted-sum readout -> LayerNorm -> MLP head**.
There is no second branch, no fusion stage, no topology vector, and only one task (regression).

| Path | Action | Why |
|---|---|---|
| `features/sequence.py`, `features/g_run_graph.py` | keep | Sequence to tokens and arrangement: idea 1 |
| `features/graph_extras.py` | edit | Keep position, gap buckets and loop-end bases (loop classes are needed for tolerance). Remove the anchor node and one-hot context, which were never used in a result |
| `features/config.py` | edit | Remove topology, sequence-branch, anchor and context fields |
| `features/topology.py` | **delete** | Persistent-homology branch |
| `data/genome_store.py`, `g4seq_index.py`, `sampler.py` | keep | How examples are read and batched |
| `data/window_dataset.py`, `data/featurize.py` | edit | Graph only: no `use_topology`, no `global_features` |
| `data/fasta.py`, `data/sequence_dataset.py` | keep | Score arbitrary sequences, needed later to embed Tm and fold-class oligos |
| `data/prepare.py` | keep | Rebuilds the data from the raw GEO files; this is provenance |
| `models/graph_encoders/diffusion_conv.py`, `diffusion_gat.py` | edit | Remove the GATv2 option |
| `models/graph_encoders/pooling.py`, `models/heads/classifier.py` | keep | Readout and head |
| `models/multimodel.py` | **replace** with `models/graph_model.py` | graph encoder -> LayerNorm -> head |
| `models/stages.py`, `models/factory.py` | edit | Two stages (graph encoder, head) and two registries |
| `models/global_encoders/` (3 files), `models/fusion/gated.py`, `models/graph_encoders/anchor_gat.py`, `models/utils.py` | **delete** | Topology, CNN and MLP branches, gate, anchor encoder, one unused helper |
| `lightning/classification_module.py` | **delete** | Every run is regression |
| `lightning/base.py`, `build.py`, `regression_module.py` | edit | Graph model only; remove the unused auxiliary G4 loss |
| `lightning/metrics.py`, `optim.py`, `resume.py` | keep | Protocol, schedule, pause and resume |
| `resources.py` | keep | GPU planning on the shared box (590 lines, but working and tested) |
| `analysis/embeddings.py` | edit | Graph embedding only; this is the transfer interface |
| `config.py`, `train.py` | edit | Remove global, fusion, task-choice and auxiliary fields |
| `tests/test_topology_encoder.py`, `test_fusion_head.py`, `test_gatv2_attention.py` | **delete** | They test removed code |
| the other 16 test files | edit | Remove the parts that test removed options |
| `configs/` | replace | `pds_r0.yaml`, `k_r0.yaml` with every value explicit |
| `docs/` | add | `FOCUS.md`, `ROADMAP.md`, `RESULTS.md`, `DECISIONS.md` |
| `environment.yml` | add | Without KmerTopology, gudhi or ripser |
| `tests/golden/` | add | Frozen reference behaviour (phase 2) |

About 510 lines of source code are deleted outright, 263 lines of tests go, and roughly a dozen files are edited.

## The guard that makes aggressive cutting safe: golden tests

Before deleting anything, record what the single-branch model does now, and require every later phase to reproduce it.

1. **Feature golden:** SHA-256 of `x`, `edge_index`, `edge_attr` and `y` for 64 fixed chr1 windows and for the synthetic
   genome in `conftest.py`. The synthetic case works on any machine.
2. **Model golden:** save a `state_dict` and the outputs for one fixed batch (`tests/golden/r0_state.pt`,
   `r0_out.pt`), and check outputs to within 1e-6.

The trap: when the fusion stage disappears, parameter names change (for example `model.fusion.norm_graph.weight`
becomes `model.norm.weight`), and the order in which parameters are created may change too, so a fixed seed can produce
different initial weights. **The golden test must load the saved weights by name, using an explicit rename table, and
must not rely on the seed.** A test that relies on the seed would fail for a harmless reason, or would pass after
someone regenerated it.

## Phases for Claude Code

Every phase uses the same rhythm: a branch, one purpose, tests before and after, review the diff, commit, merge.
Start each session with: *"Read CLAUDE.md and docs/FOCUS.md. We are doing phase N of the cleaning plan. Explain your
plan for this phase before changing anything."*

**Phase 0: housekeeping** (branch `chore/housekeeping`)
- Prompt: *"Fix the two `src.models.stages` strings in tests/test_stages.py. Add docs/FOCUS.md with the two core ideas
  (I will paste them). Commit CLAUDE.md."*
- Check: 3 failures left, all known.
- Commits: `Fix leftover src. names in stage-contract test`, `Add CLAUDE.md and docs/FOCUS.md`.

**Phase 1: add the single-branch model** (branch `feat/single-branch`)
- Prompt: *"Port the `none` global encoder, the `graph_only` fusion, their registry entries, configs/tok_pds_r0.yaml
  and tests/test_r0_equivalence.py from ~/G4_GNN_tok commit 932df71 (`git -C ~/G4_GNN_tok show 932df71`). Rename
  the configs to pds_r0.yaml and k_r0.yaml."*
- Check: the R0 model has 30,442 trainable parameters, and its output does not change when `global_features` is
  randomised.
- This is temporary scaffolding; phase 5 removes the stage concept altogether.

**Phase 2: golden tests** (branch `test/golden`)
- Prompt: *"Write tests/test_golden.py and tests/golden/ as in the cleaning plan, for the pds_r0 model. Load weights by
  name and keep a rename table that later phases may extend. Show me the hashes."*
- Check: the golden tests pass. Then deliberately change one feature by hand, confirm the test fails, and undo the
  change. A test that has never failed is not known to work.
- Tag: `v0.1-golden`.

**Phase 3: dead options with no structural role** (branch `purge/dead-options`)
- Prompt: *"Remove models/utils.py, the classification task (lightning/classification_module.py, `task` choice in
  config/build) and the aux_g4 loss. Update the tests that mention them."*
- Check: all tests and the golden tests pass; `grep -rniE "classif|aux_g4" src` prints nothing.

**Phase 4: unused branches and encoder options** (branch `purge/branches`)
- Prompt: *"Remove the seq_cnn and mlp global encoders, the anchor node and anchor_gat, the onehot_context feature and
  the GATv2 attention option. Delete tests/test_gatv2_attention.py and update the other tests."*
- Check: tests pass with **zero** failures from here on; the golden tests pass.

**Phase 5: remove the topology branch and the fusion stage** (branch `purge/topology`; the big one; possibly two
sessions)
- Prompt: *"Replace MultiModalModel with GraphModel (graph encoder -> LayerNorm -> head). Remove features/topology.py,
  topo_curve, gated fusion, the none/graph_only scaffolding, global_features in featurize and the datasets,
  use_topology, global_dim and global_layout, and the topology fields in FeatureConfig. Extend the golden rename table.
  Do it in small commits, running the tests after each."*
- Checks:
  1. All tests and the golden tests pass; parameter count is still 30,442.
  2. `python -c "import sys, g4shape.train; print([m for m in ('KmerTopology','gudhi','ripser') if m in sys.modules])"`
     prints `[]`.
  3. `grep -rniE "topolog|global_|fusion|gated" src` prints nothing.
  4. A 512-row CPU dry run trains, tests and reloads from its checkpoint.

**Phase 6: configs, dependencies, records** (branch `chore/records`)
- Prompt: *"Write environment.yml with only the packages the code imports. Make pds_r0.yaml and k_r0.yaml list every
  value explicitly. Add docs/ROADMAP.md (the four experiments), docs/RESULTS.md (empty results, predictions section)
  and docs/DECISIONS.md (one line per cut, with the reason and the commit). Update README.md with install, test and
  train commands."*
- Check: a fresh environment from `environment.yml` plus `pip install -e .` passes the tests.
- Tag: `v1-core`. This is the purged model.

**Phase 7: first runs** (not cleaning; listed so the order is clear)
R0 on PDS and K+, two seeds each, on the GPU box via `git clone` / `git pull`. Outputs come back with rsync. The
predictions go into docs/RESULTS.md and are committed before the runs start.

## How to tell the cleaning is finished

- `python -m pytest -q` shows only passes, including the golden tests.
- The core import loads no KmerTopology, gudhi or ripser.
- The R0 model has 30,442 parameters and reproduces the golden outputs.
- Every file in `src/g4shape` can be justified in one line against docs/FOCUS.md.
- `git log --oneline v0-import..v1-core` reads like a list of reasons.

## Things to watch while Claude Code works

- **Unexpected changes.** Read `git diff --stat` before every commit. A phase that touches files you did not expect
  is the signal to stop and ask.
- **Edited goldens.** If Claude Code proposes regenerating a golden file, ask why. A behaviour change in a cleaning
  phase is a bug unless you decided it.
- **Old repos.** Keep G4_GNN_fast and G4_GNN_tok read-only from now on; they are the archive.
