"""Factory: one source of defaults, strict per-stage kwargs, data layout passed only where used."""

import inspect

import pytest
import torch

from g4shape.lightning.build import build_lit_model
from g4shape.models.factory import (CLASSIFIER_REGISTRY, DEFAULTS, FUSION_REGISTRY,
                                GLOBAL_ENCODER_REGISTRY, GRAPH_ENCODER_REGISTRY,
                                build_components, build_global_encoder, build_graph_encoder)
from g4shape.models.global_encoders.mlp import MLPEncoder
from g4shape.models.global_encoders.topo_curve import TopologyCurveEncoder

DIMS = dict(node_dim=6, edge_dim=4, global_dim=8, hidden_dim=16)
LAYOUT = {"num_clouds": 2, "num_radii": 2}


def test_defaults_defined_once():
    for fn in (build_components, build_lit_model):
        sig = inspect.signature(fn).parameters
        assert {k: sig[k].default for k in DEFAULTS} == DEFAULTS, fn.__name__
    regs = {"graph_encoder": GRAPH_ENCODER_REGISTRY, "global_encoder": GLOBAL_ENCODER_REGISTRY,
            "fusion": FUSION_REGISTRY, "classifier": CLASSIFIER_REGISTRY}
    assert all(DEFAULTS[k] in regs[k] for k in DEFAULTS)
    # one config name per class: no aliases that make two configs look different
    for reg in regs.values():
        assert len(set(reg.values())) == len(reg)


def test_kwarg_typo_raises_with_suggestion():
    with pytest.raises(TypeError, match=r"'num_head'.*did you mean 'num_heads'"):
        build_components(**DIMS, global_layout=LAYOUT, graph_encoder_kwargs={"num_head": 8})
    with pytest.raises(TypeError, match=r"'layer_droput'.*did you mean 'layer_dropout'"):
        build_lit_model(**DIMS, global_layout=LAYOUT, graph_encoder_kwargs={"layer_droput": 0.3})
    with pytest.raises(TypeError, match="norm_input"):
        build_components(**DIMS, global_layout=LAYOUT, fusion_kwargs={"norm_input": False})


def test_wired_dimension_cannot_be_overridden_in_kwargs():
    with pytest.raises(ValueError, match=r"hidden_dim.*set by the builder"):
        build_components(**DIMS, global_layout=LAYOUT, graph_encoder_kwargs={"hidden_dim": 128})


def test_unknown_name_raises_with_suggestion():
    with pytest.raises(ValueError, match=r"unknown graph_encoder 'diffusion_gta'.*did you mean 'diffusion_gat'"):
        build_graph_encoder("diffusion_gta", node_dim=6, edge_dim=4, hidden_dim=16)
    with pytest.raises(ValueError, match="unknown global_encoder"):
        build_components(**DIMS, global_encoder="residual_mlp", global_layout=LAYOUT)
    with pytest.raises(ValueError, match="unknown graph_encoder 'gat'"):  # removed
        build_graph_encoder("gat", node_dim=6, edge_dim=4, hidden_dim=16)


def test_valid_kwargs_reach_the_modules():
    c = build_components(**DIMS, global_layout=LAYOUT,
                         graph_encoder_kwargs={"num_heads": 2, "layer_dropout": 0.3},
                         fusion_kwargs={"modality_dropout": 0.1})
    assert len(c["graph_encoder"].convs[0].heads) == 2 and c["graph_encoder"].layer_dropout == 0.3
    assert c["fusion"].modality_dropout == 0.1


def test_layout_goes_only_to_encoders_that_use_it():
    c = build_components(**DIMS, global_layout=LAYOUT)
    assert isinstance(c["global_encoder"], TopologyCurveEncoder)
    assert (c["global_encoder"].num_clouds, c["global_encoder"].num_radii) == (2, 2)
    c = build_components(**DIMS, global_encoder="mlp", global_layout=LAYOUT)  # no error
    assert isinstance(c["global_encoder"], MLPEncoder)
    # explicit kwargs win over the layout (here: a mismatching layout is then rejected)
    with pytest.raises(ValueError):
        build_components(**DIMS, global_layout=LAYOUT, global_encoder_kwargs={"num_radii": 4})
    # the layout is data, so it is not checked as a hyperparameter but kwargs still are
    with pytest.raises(TypeError):
        build_global_encoder("mlp", global_dim=8, hidden_dim=16, num_radii=2)


def test_dropout_and_fixed_args_wired():
    c = build_components(**DIMS, dropout=0.35, global_layout=LAYOUT)
    assert c["graph_encoder"].dropout == 0.35 and c["classifier"].dropout.p == 0.35
    assert c["classifier"].hidden[0].in_features == 16


def test_lit_model_records_the_applied_architecture():
    m = build_lit_model(**DIMS, global_layout=LAYOUT, graph_encoder_kwargs={"num_heads": 2})
    arch = m.hparams["architecture"]
    assert "branch" not in arch and arch["global_layout"] == LAYOUT
    assert arch["graph_encoder_kwargs"] == {"num_heads": 2}
    assert arch["graph_encoder"] == DEFAULTS["graph_encoder"]
    assert torch.isfinite(next(m.parameters())).all()
