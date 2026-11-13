"""Build the four stages of the network by name (graph encoder, global encoder, fusion, head).

One registry per stage maps a config name to its class; ``DEFAULTS`` names the default of
each stage and is the only place defaults are defined (``build_components`` and
``src.lightning.build.build_lit_model`` both use it).

Every stage is built the same way (``_build``):

* ``fixed`` arguments are the dimensions the builder wires between stages (``node_dim``,
  ``hidden_dim``, ...) plus the shared ``dropout`` (only passed to classes that take it);
* ``kwargs`` are the user's per-stage hyperparameters and are **checked**: a key the class
  does not accept raises (with the closest valid name), as does a key that overrides a
  wired dimension. A config typo therefore fails at build time instead of silently
  training with the default value;
* ``build_components(global_layout=...)`` is different: it describes the *input data*
  (topology ``{num_clouds, num_radii}``), so each key is passed only to a global encoder
  that takes it (``topo_curve`` does).

A new stage class must subclass its contract in ``src.models.stages`` (``STAGE_BASES``)
and be imported and added to its registry here by hand.
"""

from __future__ import annotations

import difflib
import inspect
from typing import Any, Mapping, Optional

from g4shape.models.fusion.gated import GatedFusion
from g4shape.models.global_encoders.topo_curve import TopologyCurveEncoder
from g4shape.models.graph_encoders.diffusion_gat import DiffusionGATEncoder
from g4shape.models.heads.classifier import BinaryMLPClassifier
from g4shape.models.stages import Fusion, GlobalEncoder, GraphEncoder, Head

__all__ = ["GRAPH_ENCODER_REGISTRY", "GLOBAL_ENCODER_REGISTRY", "FUSION_REGISTRY",
           "CLASSIFIER_REGISTRY", "DEFAULTS", "STAGE_BASES", "build_graph_encoder", "build_global_encoder",
           "build_fusion", "build_classifier", "build_components"]

GRAPH_ENCODER_REGISTRY = {
    "diffusion_gat": DiffusionGATEncoder,
}

GLOBAL_ENCODER_REGISTRY = {
    "topo_curve": TopologyCurveEncoder,
}

FUSION_REGISTRY = {
    "gated": GatedFusion,
}

CLASSIFIER_REGISTRY = {
    "binary_mlp": BinaryMLPClassifier,
}

DEFAULTS = {
    "graph_encoder": "diffusion_gat",
    "global_encoder": "topo_curve",
    "fusion": "gated",
    "classifier": "binary_mlp",
}

STAGE_BASES = {  # the contract a registered class must implement (src/models/stages.py)
    "graph_encoder": GraphEncoder,
    "global_encoder": GlobalEncoder,
    "fusion": Fusion,
    "classifier": Head,
}

_OPTIONAL_SHARED = {"dropout"}  # wired by the builder, but only into classes that take it


def _accepted(cls) -> tuple[set[str], bool]:
    """(named parameters of ``cls.__init__``, whether it takes ``**kwargs``)."""
    params = inspect.signature(cls).parameters.values()
    named = {p.name for p in params
             if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    return named, any(p.kind == p.VAR_KEYWORD for p in params)


def _suggest(key: str, options) -> str:
    close = difflib.get_close_matches(key, sorted(options), n=1)
    return f" (did you mean {close[0]!r}?)" if close else ""


def _build(stage: str, registry: Mapping[str, type], name: Optional[str],
           fixed: Mapping[str, Any], kwargs: Optional[Mapping[str, Any]]):
    """Instantiate ``registry[name]`` with the wired ``fixed`` args + checked ``kwargs``."""
    key = DEFAULTS[stage] if name is None else str(name).lower()
    if key not in registry:
        raise ValueError(f"unknown {stage} {key!r}{_suggest(key, registry)}; "
                         f"available: {sorted(registry)}")
    cls = registry[key]
    if not issubclass(cls, STAGE_BASES[stage]):
        raise TypeError(f"{cls.__name__} is registered as a {stage} but does not subclass "
                        f"g4shape.models.stages.{STAGE_BASES[stage].__name__}")
    named, var_kw = _accepted(cls)
    kwargs = dict(kwargs or {})

    clash = sorted(set(kwargs) & set(fixed))
    if clash:
        raise ValueError(f"{stage} {key!r}: {clash} is set by the builder (pass it to "
                         f"build_components / build_lit_model), not in {stage}_kwargs")
    unknown = sorted(k for k in kwargs if k not in named) if not var_kw else []
    if unknown:
        hints = "".join(f"\n  {k!r}{_suggest(k, named - set(fixed))}" for k in unknown)
        raise TypeError(f"{stage} {key!r} ({cls.__name__}) does not accept:{hints}\n"
                        f"accepted: {sorted(named - set(fixed))}")

    args = {k: v for k, v in fixed.items()
            if k not in _OPTIONAL_SHARED or k in named or var_kw}
    missing = sorted(set(args) - named) if not var_kw else []
    if missing:  # a registered class that breaks the stage's constructor contract
        raise TypeError(f"{cls.__name__} must accept {missing} to be a {stage}")
    return cls(**args, **kwargs)


# Single-stage builders (one stage on its own, e.g. in tests); build_components builds all four.
def build_graph_encoder(name=None, node_dim=None, edge_dim=None, hidden_dim=64, dropout=0.2,
                        **kwargs):
    return _build("graph_encoder", GRAPH_ENCODER_REGISTRY, name,
                  dict(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim,
                       dropout=dropout), kwargs)


def build_global_encoder(name=None, global_dim=None, hidden_dim=64, dropout=0.2, **kwargs):
    return _build("global_encoder", GLOBAL_ENCODER_REGISTRY, name,
                  dict(global_dim=global_dim, hidden_dim=hidden_dim, dropout=dropout), kwargs)


def build_fusion(name=None, hidden_dim=64, **kwargs):
    return _build("fusion", FUSION_REGISTRY, name, dict(hidden_dim=hidden_dim), kwargs)


def build_classifier(name=None, in_dim=64, hidden_dim=64, dropout=0.2, **kwargs):
    return _build("classifier", CLASSIFIER_REGISTRY, name,
                  dict(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout), kwargs)


def build_components(
    node_dim,
    edge_dim,
    global_dim,
    hidden_dim,
    dropout=0.2,
    graph_encoder=DEFAULTS["graph_encoder"],
    global_encoder=DEFAULTS["global_encoder"],
    fusion=DEFAULTS["fusion"],
    classifier=DEFAULTS["classifier"],
    graph_encoder_kwargs=None,
    global_encoder_kwargs=None,
    fusion_kwargs=None,
    classifier_kwargs=None,
    global_layout=None,
):
    """The four stages of the network, by name: ``{graph_encoder, global_encoder, fusion,
    classifier}``, all sharing ``hidden_dim`` (the head maps ``hidden_dim`` -> 1).

    ``global_layout``: shape of the global feature vector, from the data
    (``load_window_bundle(...)["global_layout"]`` = topology ``{num_clouds, num_radii}``);
    each key goes to the global encoder only if it takes it. Explicit
    ``global_encoder_kwargs`` win.

    Lightning-free: ``MultiModalModel(**build_components(...))`` is the bare network;
    ``src.lightning.build.build_lit_model`` wraps the same components for training.
    """
    global_kwargs = dict(global_encoder_kwargs or {})
    key = DEFAULTS["global_encoder"] if global_encoder is None else str(global_encoder).lower()
    if global_layout and key in GLOBAL_ENCODER_REGISTRY:  # unknown names fail in _build
        named, var_kw = _accepted(GLOBAL_ENCODER_REGISTRY[key])
        global_kwargs = {**{k: v for k, v in global_layout.items() if var_kw or k in named},
                         **global_kwargs}
    # _build gets each kwargs dict as a dict (not **-unpacked) so that a wired key inside it,
    # e.g. graph_encoder_kwargs={"hidden_dim": 128}, reaches the clash check
    return {
        "graph_encoder": _build("graph_encoder", GRAPH_ENCODER_REGISTRY, graph_encoder,
                                dict(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim,
                                     dropout=dropout), graph_encoder_kwargs),
        "global_encoder": _build("global_encoder", GLOBAL_ENCODER_REGISTRY, global_encoder,
                                 dict(global_dim=global_dim, hidden_dim=hidden_dim,
                                      dropout=dropout), global_kwargs),
        "fusion": _build("fusion", FUSION_REGISTRY, fusion, dict(hidden_dim=hidden_dim),
                         fusion_kwargs),
        "classifier": _build("classifier", CLASSIFIER_REGISTRY, classifier,
                             dict(in_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout),
                             classifier_kwargs),
    }
