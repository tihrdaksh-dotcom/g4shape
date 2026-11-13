"""featurize (one sequence -> Data) and SequenceDataset (scoring arbitrary sequences)."""

import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader

from g4shape.data.featurize import feature_dims, featurize, featurize_sequence
from g4shape.data.sequence_dataset import SequenceDataset
from g4shape.features.config import FeatureConfig
from g4shape.features.sequence import encode, reverse_complement
from g4shape.lightning.build import build_lit_model

CFG = FeatureConfig()
G4 = "ACGT" * 20 + "GGGTTAGGGTTAGGGTTAGGG" + "TGCA" * 28 + "ACGTACG"  # 215 nt
SEQS = [G4, "A" * 215, "acgu" * 50 + "N" * 15]


def test_shapes_match_feature_dims():
    dims = feature_dims(CFG)
    d = featurize_sequence(G4, CFG)
    assert d.had_graph and d.x.shape[1] == dims["node_dim"] and d.edge_attr.shape[1] == dims["edge_dim"]
    assert set(dims) == {"node_dim", "edge_dim"}


def test_empty_graph_gets_one_zero_node():
    d = featurize_sequence("A" * 215, CFG)
    assert not d.had_graph and d.x.shape == (1, feature_dims(CFG)["node_dim"]) and not d.x.any()
    assert d.edge_index.shape == (2, 0)


def test_sequence_dataset_from_fasta_and_batching(tmp_path):
    fa = tmp_path / "s.fa"
    fa.write_text("".join(f">seq{i} some description\n{s[:100]}\n{s[100:]}\n" for i, s in enumerate(SEQS)))
    ds = SequenceDataset.from_fasta(fa, y=[50.0, 1.0, 2.0])
    assert len(ds) == 3 and ds[0].name == "seq0" and ds[2].y.item() == 2.0
    assert torch.equal(ds[0].x, featurize_sequence(G4).x)
    b = next(iter(DataLoader(ds, batch_size=3)))
    assert b.seq_id.tolist() == [0, 1, 2]  # metadata survives batching (no "index" offsetting)
    assert b.name == ["seq0", "seq1", "seq2"] and b.num_graphs == 3


def test_reverse_complement_option():
    ds = SequenceDataset([G4], reverse_complement=True)
    ref = featurize(reverse_complement(encode(G4)))
    assert torch.equal(ds[0].x, ref.x) and torch.equal(ds[0].edge_attr, ref.edge_attr)


def test_score_arbitrary_sequences_with_a_model():
    ds = SequenceDataset(SEQS, feature_config=CFG)
    model = build_lit_model(ds.dims["node_dim"], ds.dims["edge_dim"], hidden_dim=16,
                            y_mean=10.0, y_std=5.0,
                            graph_encoder_kwargs={"num_layers": 1, "num_heads": 2})
    tr = pl.Trainer(accelerator="cpu", logger=False, enable_progress_bar=False, enable_model_summary=False)
    out = tr.predict(model, DataLoader(ds, batch_size=2))
    pred = torch.cat([o["prediction"] for o in out])
    assert pred.shape == (3,) and torch.isfinite(pred).all()
