"""Builds a trainable Lightning module from component names.

Layering: ``src.models`` holds the plain ``nn.Module``s and builds them by name
(``src.models.factory``) without knowing about Lightning; this module, in the Lightning
layer, wraps those components in the training objective. Dependencies only point
downwards (``lightning`` -> ``models``), never the reverse.
"""

from g4shape.lightning.metrics import G4_LABEL_THRESHOLD, resolve_threshold_grid
from g4shape.lightning.regression_module import LitRegressionModel
from g4shape.models.factory import DEFAULTS, build_classifier, build_graph_encoder

__all__ = ["build_lit_model"]


def build_lit_model(
    node_dim: int,
    edge_dim: int,
    hidden_dim: int,
    optim=None,
    dropout: float = 0.2,
    graph_encoder: str = DEFAULTS["graph_encoder"],
    classifier: str = DEFAULTS["classifier"],
    graph_encoder_kwargs=None,
    classifier_kwargs=None,
    y_mean: float = 0.0,
    y_std: float = 1.0,
    label_threshold=G4_LABEL_THRESHOLD,
    threshold_grid=None,
):
    """Score regression, as TransG4: MSE on the mismatch score, G4 calls by thresholding the
    prediction. The network is ``GraphModel(graph_encoder, classifier)``; component kwargs
    are checked: an unknown key raises. ``optim``: optimizer / lr-schedule settings over
    TransG4's defaults (``src.lightning.optim.TRANSG4_OPTIM``), e.g. ``{"scheduler": "plateau"}``."""
    threshold_grid = resolve_threshold_grid(threshold_grid)  # list or {start, stop, step}
    extra = {} if threshold_grid is None else {"threshold_grid": threshold_grid}
    graph_encoder_kwargs = dict(graph_encoder_kwargs or {})
    classifier_kwargs = dict(classifier_kwargs or {})

    model = LitRegressionModel(
        graph_encoder=build_graph_encoder(graph_encoder, node_dim=node_dim, edge_dim=edge_dim,
                                          hidden_dim=hidden_dim, dropout=dropout,
                                          **graph_encoder_kwargs),
        classifier=build_classifier(classifier, in_dim=hidden_dim, hidden_dim=hidden_dim,
                                    dropout=dropout, **classifier_kwargs),
        optim=optim, y_mean=y_mean, y_std=y_std, label_threshold=label_threshold, **extra)

    model.hparams.update(
        {
            "architecture": {
                "graph_encoder": graph_encoder,
                "classifier": classifier,
                "graph_encoder_kwargs": graph_encoder_kwargs,
                "classifier_kwargs": classifier_kwargs,
            }
        }
    )
    return model
