"""Stage contracts (src/models/stages.py): one call signature per stage, declared widths,
checked when the network is assembled."""

import inspect

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from g4shape.models.factory import (CLASSIFIER_REGISTRY, FUSION_REGISTRY, GLOBAL_ENCODER_REGISTRY,
                                GRAPH_ENCODER_REGISTRY, STAGE_BASES, build_components)
from g4shape.models.fusion.gated import GatedFusion
from g4shape.models.multimodel import MultiModalModel
from g4shape.models.stages import GlobalEncoder

DIMS = dict(node_dim=6, edge_dim=4, global_dim=8, hidden_dim=16)
LAYOUT = {"num_clouds": 2, "num_radii": 2}
REGISTRIES = {"graph_encoder": GRAPH_ENCODER_REGISTRY, "global_encoder": GLOBAL_ENCODER_REGISTRY,
              "fusion": FUSION_REGISTRY, "classifier": CLASSIFIER_REGISTRY}


def _batch(n=3):
    g = torch.Generator().manual_seed(0)
    out = []
    for _ in range(n):
        k = 4
        d = Data(x=torch.randn(k, 6, generator=g), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
                 edge_attr=torch.randn(3, 4, generator=g), y=torch.rand(1, generator=g))
        d.global_features = torch.randn(1, 8, generator=g)
        out.append(d)
    return Batch.from_data_list(out)


def test_every_registered_class_implements_its_stage():
    for stage, reg in REGISTRIES.items():
        for name, cls in reg.items():
            assert issubclass(cls, STAGE_BASES[stage]), (stage, name)
            assert "return_details" in inspect.signature(cls.forward).parameters, (stage, name)


@pytest.mark.parametrize("global_encoder", sorted(GLOBAL_ENCODER_REGISTRY))
def test_every_stage_reports_details_and_plain_path_matches(global_encoder):
    torch.manual_seed(0)
    net = MultiModalModel(**build_components(**DIMS, global_encoder=global_encoder,
                                             global_layout=LAYOUT)).eval()
    b = _batch()
    out, det = net(b, return_details=True)
    assert torch.allclose(out, net(b)) and out.shape == (3, 1)
    for k in ("graph", "global", "fusion", "head"):
        assert isinstance(det[k], dict), k
    assert det["gate"].shape == det["x_graph"].shape == det["x_global"].shape == (3, 16)
    assert det["penultimate"].shape == (3, 16)


def test_forward_does_not_inspect_signatures(monkeypatch):
    net = MultiModalModel(**build_components(**DIMS, global_layout=LAYOUT)).eval()
    b = _batch()

    def boom(*a, **k):
        raise AssertionError("inspect.signature called during forward")
    monkeypatch.setattr(inspect, "signature", boom)
    net(b)
    net(b, return_details=True)


def test_rejects_module_without_contract():
    c = build_components(**DIMS, global_layout=LAYOUT)
    c["global_encoder"] = nn.Linear(8, 16)  # right shapes, no contract
    with pytest.raises(TypeError, match="global_encoder must be a src.models.stages.GlobalEncoder"):
        MultiModalModel(**c)


def test_width_mismatch_is_reported_by_name():
    c = build_components(**DIMS, global_layout=LAYOUT)
    c["fusion"] = GatedFusion(32)
    with pytest.raises(ValueError, match=r"graph_encoder.out_dim=16 != fusion.graph_dim=32"):
        MultiModalModel(**c)


def test_factory_rejects_registered_class_without_contract(monkeypatch):
    class Plain(nn.Module):
        def __init__(self, global_dim, hidden_dim, dropout=0.2):
            super().__init__()

    monkeypatch.setitem(GLOBAL_ENCODER_REGISTRY, "plain", Plain)
    with pytest.raises(TypeError, match="does not subclass src.models.stages.GlobalEncoder"):
        build_components(**DIMS, global_encoder="plain")


def test_new_stage_needs_only_the_contract(monkeypatch):
    class MeanEncoder(GlobalEncoder):
        def __init__(self, global_dim, hidden_dim, dropout=0.2):
            super().__init__(out_dim=hidden_dim)
            self.lin = nn.Linear(1, hidden_dim)

        def forward(self, x, return_details=False):
            out = self.lin(x.mean(1, keepdim=True))
            return (out, {"mean": x.mean(1)}) if return_details else out

    monkeypatch.setitem(GLOBAL_ENCODER_REGISTRY, "mean", MeanEncoder)
    net = MultiModalModel(**build_components(**DIMS, global_encoder="mean"))
    _, det = net(_batch(), return_details=True)
    assert det["global"]["mean"].shape == (3,)
