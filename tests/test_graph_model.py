"""GraphModel: graph encoder -> LayerNorm -> head (src/models/graph_model.py)."""

import pytest
import torch
from torch_geometric.data import Batch

from g4shape.data.featurize import feature_dims, featurize_sequence
from g4shape.features.config import FeatureConfig
from g4shape.models.factory import CLASSIFIER_REGISTRY, GRAPH_ENCODER_REGISTRY
from g4shape.models.graph_model import GraphModel

POSONLY = FeatureConfig(min_run_len=2, edge_radius=50, context_bases=2, flank_window=20,
                        position_encoding=True)
SEQS = ("TTAGGGTTAGGGTTAGGGTTAGGGTTA", "AAGGGAGGGAGGGAGGGAA", "ACGTACGTACGTACGT",
        "GGTTGGTTGGTTGG")


def _model(hidden_dim=64, dropout=0.2):
    dims = feature_dims(POSONLY)
    enc = GRAPH_ENCODER_REGISTRY["diffusion_gat"](node_dim=dims["node_dim"], edge_dim=dims["edge_dim"],
                                                  hidden_dim=hidden_dim, dropout=dropout)
    head = CLASSIFIER_REGISTRY["binary_mlp"](in_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout)
    return GraphModel(enc, head)


def _batch():
    return Batch.from_data_list([featurize_sequence(s, POSONLY) for s in SEQS])


def test_shapes_and_finite_output():
    net, b = _model().eval(), _batch()
    out, det = net(b, return_details=True)
    assert out.shape == (len(SEQS), 1) and torch.isfinite(out).all()
    assert det["x_graph"].shape == det["penultimate"].shape == (len(SEQS), 64)
    assert {"attention", "diffusion_coeffs", "node_embeddings", "pool_weights"} <= set(det["graph"])
    assert torch.equal(out, net(b))  # the details path is the plain path


def test_trainable_parameters_r0():
    n = sum(p.numel() for p in _model().parameters() if p.requires_grad)
    assert n == 30_442


def test_gradients_reach_encoder_and_head():
    net = _model().train()
    net(_batch()).sum().backward()
    for name in ("graph_encoder", "norm", "classifier"):
        grads = [p.grad for p in getattr(net, name).parameters()]
        assert all(g is not None for g in grads), name
        assert any(g.abs().sum() > 0 for g in grads), name


def test_rejects_mismatched_widths():
    dims = feature_dims(POSONLY)
    enc = GRAPH_ENCODER_REGISTRY["diffusion_gat"](node_dim=dims["node_dim"], edge_dim=dims["edge_dim"],
                                                  hidden_dim=32)
    head = CLASSIFIER_REGISTRY["binary_mlp"](in_dim=64)
    with pytest.raises(ValueError, match="widths"):
        GraphModel(enc, head)
