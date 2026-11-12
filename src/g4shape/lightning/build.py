"""Builds a trainable Lightning module from component names.

Layering: ``src.models`` holds the plain ``nn.Module``s and builds them by name
(``src.models.factory.build_components``) without knowing about Lightning; this module,
in the Lightning layer, picks the training objective and wraps those components.
Dependencies only point downwards (``lightning`` -> ``models``), never the reverse.
"""

from g4shape.lightning.classification_module import LitClassificationModel
from g4shape.lightning.metrics import G4_LABEL_THRESHOLD, resolve_threshold_grid
from g4shape.lightning.regression_module import LitRegressionModel
from g4shape.models.factory import DEFAULTS, build_components

__all__ = ["build_lit_model"]

TASKS = ("regression", "classification")


def build_lit_model(
    node_dim: int,
    edge_dim: int,
    global_dim: int,
    hidden_dim: int,
    optim=None,
    dropout: float = 0.2,
    graph_encoder: str = DEFAULTS["graph_encoder"],
    global_encoder: str = DEFAULTS["global_encoder"],
    fusion: str = DEFAULTS["fusion"],
    classifier: str = DEFAULTS["classifier"],
    graph_encoder_kwargs=None,
    global_encoder_kwargs=None,
    fusion_kwargs=None,
    classifier_kwargs=None,
    task: str = "regression",
    y_mean: float = 0.0,
    y_std: float = 1.0,
    global_layout=None,
    label_threshold=G4_LABEL_THRESHOLD,
    threshold_grid=None,
    aux_g4_weight=0.0,
    aux_g4_pos_weight=1.0,
):
    """``task``: "regression" (default; TransG4: MSE on the mismatch score, G4 calls by
    thresholding the prediction) or "classification" (BCE on ``y > label_threshold``).
    ``global_layout``: ``load_window_bundle(...)["global_layout"]`` (topology
    ``{num_clouds, num_radii}``), given to the global encoder if it takes it (see
    ``build_components``). Component kwargs are checked: an unknown key raises.
    ``optim``: optimizer / lr-schedule settings over TransG4's defaults
    (``src.lightning.optim.TRANSG4_OPTIM``), e.g. ``{"scheduler": "plateau"}``.
    ``aux_g4_weight`` / ``aux_g4_pos_weight``: optional auxiliary G4 loss (regression only,
    see ``LitRegressionModel``; 0 = off)."""
    threshold_grid = resolve_threshold_grid(threshold_grid)  # list or {start, stop, step}
    extra = {} if threshold_grid is None else {"threshold_grid": threshold_grid}
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    if aux_g4_weight and task != "regression":
        raise ValueError("aux_g4_weight is only defined for task='regression'")
    if aux_g4_weight:
        extra.update(aux_g4_weight=aux_g4_weight, aux_g4_pos_weight=aux_g4_pos_weight)

    graph_encoder_kwargs = dict(graph_encoder_kwargs or {})
    global_encoder_kwargs = dict(global_encoder_kwargs or {})
    global_layout = dict(global_layout or {})
    fusion_kwargs = dict(fusion_kwargs or {})
    classifier_kwargs = dict(classifier_kwargs or {})

    components = build_components(
        node_dim=node_dim,
        edge_dim=edge_dim,
        global_dim=global_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        graph_encoder=graph_encoder,
        global_encoder=global_encoder,
        fusion=fusion,
        classifier=classifier,
        graph_encoder_kwargs=graph_encoder_kwargs,
        global_encoder_kwargs=global_encoder_kwargs,
        fusion_kwargs=fusion_kwargs,
        classifier_kwargs=classifier_kwargs,
        global_layout=global_layout,
    )

    if task == "regression":  # the 1-output classifier head is the regressor head
        model = LitRegressionModel(**components, optim=optim,
                                   y_mean=y_mean, y_std=y_std, label_threshold=label_threshold,
                                   **extra)
    else:
        model = LitClassificationModel(**components, optim=optim,
                                       label_threshold=label_threshold, **extra)

    model.hparams.update(
        {
            "architecture": {
                "graph_encoder": graph_encoder,
                "global_encoder": global_encoder,
                "fusion": fusion,
                "classifier": classifier,
                "graph_encoder_kwargs": graph_encoder_kwargs,
                "global_encoder_kwargs": global_encoder_kwargs,
                "fusion_kwargs": fusion_kwargs,
                "classifier_kwargs": classifier_kwargs,
                "global_layout": global_layout,
            }
        }
    )

    return model
