# g4shape

Arrangement-of-tokens models for G-quadruplex prediction. The aims are in
[docs/FOCUS.md](docs/FOCUS.md).

## The model (R0)

One model, regression only:

```
G-run graph  ->  diffusion-attention encoder  ->  weighted-sum readout  ->  LayerNorm  ->  MLP head
```

Each window (100 + 15 + 100 nt around a G4-seq bin) becomes a graph whose nodes are G-runs and
whose edges link runs up to 50 nt apart, carrying loop length and signed position relative to
the scored bin. A DADGNN-style diffusion-attention encoder (2 layers, 4 heads, 3 hops) reads it
out with a weighted sum; a LayerNorm and a two-layer MLP predict the standardised mismatch score.
G4 calls threshold that score (threshold tuned on validation). 30,442 trainable parameters.

## Install

```bash
conda env create -f environment.yml && conda activate g4gnn && pip install -e .
```

## Test

```bash
python -m pytest -q
```

`tests/test_golden.py` checks frozen fingerprints of the features, the encoder and the head
(`tests/golden/`, made once by `scripts/make_golden.py`). Never regenerate them: if one fails,
the code changed.

## Train

```bash
python -m g4shape.train --config configs/pds_r0.yaml      # K+ + PDS
python -m g4shape.train --config configs/k_r0.yaml        # K+
```

Override any setting with `--set section.key=value`; `--print-config` shows the resolved
config. A run writes `runs/<run_name>/` (checkpoints, `test_metrics.json`, the config).
`python -m scripts.evaluate <best.ckpt> --config <yaml>` re-scores a checkpoint without training.

## Layout

Every file in `src/g4shape`, with the reason it is kept:

| File | Why it is here |
|---|---|
| `features/sequence.py` | Nucleotide codes: the one sequence representation of the data path |
| `features/config.py` | `FeatureConfig`: the graph's settings |
| `features/g_run_graph.py` | The G-run graph: tokens (G-runs) and their arrangement (idea 1) |
| `features/graph_extras.py` | Signed position, loop-length classes and loop-end bases (the loop-class features of the tolerance experiment) |
| `data/fasta.py` | FASTA reader for the genome store and for scoring sequence files |
| `data/genome_store.py` | hg19 as memory-mapped code arrays |
| `data/g4seq_index.py` | G4-seq scores as a compact row index, split by chromosome |
| `data/prepare.py` | Raw GEO data + hg19 FASTA -> genome store and index |
| `data/featurize.py` | The single place a sequence becomes a model input |
| `data/window_dataset.py` | G4-seq windows, featurized on demand (training data) |
| `data/sequence_dataset.py` | Arbitrary sequences, for scoring other G4 sets (reuse, idea 2) |
| `data/sampler.py` | Rank-independent sample order, so runs can stop and resume |
| `models/stages.py` | The contracts of the two stages (encoder, head) |
| `models/graph_encoders/diffusion_conv.py` | One diffusion-attention head (GATv1 scoring, DADGNN) |
| `models/graph_encoders/diffusion_gat.py` | The encoder; exposes attention and readout weights (readability, idea 2) |
| `models/graph_encoders/pooling.py` | The weighted-sum readout |
| `models/heads/classifier.py` | The MLP head (`binary_mlp`) |
| `models/graph_model.py` | `GraphModel`: encoder -> LayerNorm -> head |
| `models/factory.py` | Builds the stages by name, rejecting unknown settings |
| `lightning/base.py` | Training/evaluation loop, G4-call protocol, `characterize` for analysis |
| `lightning/regression_module.py` | The objective: (weighted) MSE on the standardised score |
| `lightning/build.py` | Wraps the network in the objective |
| `lightning/metrics.py` | Pearson, R2, precision / recall / Jaccard / AUPRC, threshold tuning |
| `lightning/optim.py` | Optimiser and learning-rate schedule (TransG4 recipe) |
| `lightning/resume.py` | Stop and resume a run, possibly on other GPUs |
| `analysis/embeddings.py` | Per-sample embeddings (`x_graph`, `penultimate`) for readability and reuse checks |
| `config.py` | One YAML file per experiment; unknown keys are errors |
| `resources.py` | Uses only free GPUs and idle CPUs of a shared machine |
| `train.py` | Entry point: train, test and reload a run |

Other folders: `configs/` (one YAML per run), `scripts/` (`make_subset.py`: the weighted fast
subsets; `evaluate.py`: re-score a checkpoint; `make_golden.py`: the golden files), `tests/`,
`docs/` (aims, roadmap, results, decisions).
