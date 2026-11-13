"""Stage contracts (src/models/stages.py): one call signature per stage, declared widths,
checked when the network is assembled."""

import inspect

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from g4shape.models.factory import (CLASSIFIER_REGISTRY, GRAPH_ENCODER_REGISTRY, STAGE_BASES,
                                    build_components)
from g4shape.models.graph_model import GraphModel
from g4shape.models.stages import Head

DIMS = dict(node_dim=6, edge_dim=4, hidden_dim=16)
REGISTRIES = {"graph_encoder": GRAPH_ENCODER_REGISTRY, "classifier": CLASSIFIER_REGISTRY}


def _batch(n=3):
    g = torch.Generator().manual_seed(0)
    out = []
    for _ in range(n):
        k = 4
        out.append(Data(x=torch.randn(k, 6, generator=g), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
                        edge_attr=torch.randn(3, 4, generator=g), y=torch.rand(1, generator=g)))
    return Batch.from_data_list(out)


def test_every_registered_class_implements_its_stage():
    for stage, reg in REGISTRIES.items():
        for name, cls in reg.items():
            assert issubclass(cls, STAGE_BASES[stage]), (stage, name)
            assert "return_details" in inspect.signature(cls.forward).parameters, (stage, name)


def test_every_stage_reports_details_and_plain_path_matches():
    torch.manual_seed(0)
    net = GraphModel(**build_components(**DIMS)).eval()
    b = _batch()
    out, det = net(b, return_details=True)
    assert torch.allclose(out, net(b)) and out.shape == (3, 1)
    for k in ("graph", "head"):
        assert isinstance(det[k], dict), k
    assert det["x_graph"].shape == det["penultimate"].shape == (3, 16)


def test_forward_does_not_inspect_signatures(monkeypatch):
    net = GraphModel(**build_components(**DIMS)).eval()
    b = _batch()

    def boom(*a, **k):
        raise AssertionError("inspect.signature called during forward")
    monkeypatch.setattr(inspect, "signature", boom)
    net(b)
    net(b, return_details=True)


def test_rejects_module_without_contract():
    c = build_components(**DIMS)
    c["classifier"] = nn.Linear(16, 1)  # right shapes, no contract
    with pytest.raises(TypeError, match="classifier must be a g4shape.models.stages.Head"):
        GraphModel(**c)


def test_width_mismatch_is_reported_by_name():
    c = build_components(**DIMS)
    c["classifier"] = CLASSIFIER_REGISTRY["binary_mlp"](in_dim=32)
    with pytest.raises(ValueError, match=r"graph_encoder.out_dim=16 != classifier.in_dim=32"):
        GraphModel(**c)


def test_factory_rejects_registered_class_without_contract(monkeypatch):
    class Plain(nn.Module):
        def __init__(self, in_dim, hidden_dim, dropout=0.2):
            super().__init__()

    monkeypatch.setitem(CLASSIFIER_REGISTRY, "plain", Plain)
    with pytest.raises(TypeError, match="does not subclass g4shape.models.stages.Head"):
        build_components(**DIMS, classifier="plain")


def test_new_stage_needs_only_the_contract(monkeypatch):
    class LinearHead(Head):
        def __init__(self, in_dim, hidden_dim, dropout=0.2):
            super().__init__(in_dim=in_dim)
            self.lin = nn.Linear(in_dim, 1)

        def forward(self, x, return_details=False):
            out = self.lin(x)
            return (out, {"norm": x.norm(dim=1)}) if return_details else out

    monkeypatch.setitem(CLASSIFIER_REGISTRY, "linear", LinearHead)
    net = GraphModel(**build_components(**DIMS, classifier="linear"))
    _, det = net(_batch(), return_details=True)
    assert det["head"]["norm"].shape == (3,) and det["penultimate"] is None
