"""Factory: one source of defaults, strict per-stage kwargs."""

import inspect

import pytest
import torch

from g4shape.lightning.build import build_lit_model
from g4shape.models.factory import (CLASSIFIER_REGISTRY, DEFAULTS, GRAPH_ENCODER_REGISTRY,
                                    build_components, build_graph_encoder)

DIMS = dict(node_dim=6, edge_dim=4, hidden_dim=16)


def test_defaults_defined_once():
    for fn in (build_components, build_lit_model):
        sig = inspect.signature(fn).parameters
        assert {k: sig[k].default for k in DEFAULTS} == DEFAULTS, fn.__name__
    regs = {"graph_encoder": GRAPH_ENCODER_REGISTRY, "classifier": CLASSIFIER_REGISTRY}
    assert set(DEFAULTS) == set(regs) and all(DEFAULTS[k] in regs[k] for k in DEFAULTS)
    # one config name per class: no aliases that make two configs look different
    for reg in regs.values():
        assert len(set(reg.values())) == len(reg)


def test_kwarg_typo_raises_with_suggestion():
    with pytest.raises(TypeError, match=r"'num_head'.*did you mean 'num_heads'"):
        build_components(**DIMS, graph_encoder_kwargs={"num_head": 8})
    with pytest.raises(TypeError, match=r"'layer_droput'.*did you mean 'layer_dropout'"):
        build_lit_model(**DIMS, graph_encoder_kwargs={"layer_droput": 0.3})
    with pytest.raises(TypeError, match="hiden_dim"):
        build_components(**DIMS, classifier_kwargs={"hiden_dim": 8})


def test_wired_dimension_cannot_be_overridden_in_kwargs():
    with pytest.raises(ValueError, match=r"hidden_dim.*set by the builder"):
        build_components(**DIMS, graph_encoder_kwargs={"hidden_dim": 128})
    with pytest.raises(ValueError, match=r"hidden_dim.*set by the builder"):
        build_lit_model(**DIMS, graph_encoder_kwargs={"hidden_dim": 128})


def test_unknown_name_raises_with_suggestion():
    with pytest.raises(ValueError, match=r"unknown graph_encoder 'diffusion_gta'.*did you mean 'diffusion_gat'"):
        build_graph_encoder("diffusion_gta", node_dim=6, edge_dim=4, hidden_dim=16)
    with pytest.raises(ValueError, match="unknown classifier"):
        build_components(**DIMS, classifier="residual_mlp")
    with pytest.raises(ValueError, match="unknown graph_encoder 'gat'"):  # removed
        build_graph_encoder("gat", node_dim=6, edge_dim=4, hidden_dim=16)


def test_valid_kwargs_reach_the_modules():
    c = build_components(**DIMS, graph_encoder_kwargs={"num_heads": 2, "layer_dropout": 0.3})
    assert len(c["graph_encoder"].convs[0].heads) == 2 and c["graph_encoder"].layer_dropout == 0.3


def test_dropout_and_fixed_args_wired():
    c = build_components(**DIMS, dropout=0.35)
    assert set(c) == {"graph_encoder", "classifier"}
    assert c["graph_encoder"].dropout == 0.35 and c["classifier"].dropout.p == 0.35
    assert c["classifier"].hidden[0].in_features == 16


def test_lit_model_records_the_applied_architecture():
    m = build_lit_model(**DIMS, graph_encoder_kwargs={"num_heads": 2})
    arch = m.hparams["architecture"]
    assert set(arch) == {"graph_encoder", "classifier", "graph_encoder_kwargs", "classifier_kwargs"}
    assert arch["graph_encoder_kwargs"] == {"num_heads": 2}
    assert arch["graph_encoder"] == DEFAULTS["graph_encoder"]
    assert torch.isfinite(next(m.parameters())).all()
