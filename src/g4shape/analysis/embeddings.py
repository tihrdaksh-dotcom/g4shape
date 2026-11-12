"""Collect per-sample embeddings over a dataset and project them to 2-D/3-D for plotting.

Typical use::

    emb = collect_embeddings(lit_model, test_loader, max_batches=200)
    Y, info = project(emb["fused"], method="pca")          # or "tsne" / "umap"
    plot_projection(Y, emb["y"], title=f"fused, PCA {info['explained_variance_ratio']}")
    save_embeddings("runs/x/test_embeddings.npz", emb)

``collect_embeddings`` returns numpy arrays aligned on the sample axis:

* model stages: ``x_graph``, ``x_global`` [N, H] (branch embeddings), ``gate`` [N, H]
  (topology weight per fused dimension), ``gate_mean`` [N] (0 = graph only, 1 = topology
  only), ``fused`` [N, H], ``penultimate`` [N, H_head], ``clouds`` [N, C, d] (topo_curve
  per-cloud vectors, when available);
* outputs: ``output`` [N] (raw head output), ``prediction`` [N] (``model.postprocess``:
  score in original units for a regression model, probability for a classifier; equal
  to ``output`` for a bare ``MultiModalModel``, which does not know its task),
  ``is_g4`` [N] (Lightning models only: prediction > ``detection_threshold``), ``y`` [N];
* metadata when present on the batch: ``chrom``, ``start``, ``strand``, ``split``,
  ``had_graph``, and ``num_nodes`` [N] (graph size).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch

__all__ = ["collect_embeddings", "project", "plot_projection", "save_embeddings", "load_embeddings"]

_STAGES = ("x_graph", "x_global", "gate", "fused", "penultimate")
_META = ("chrom", "start", "strand", "split", "had_graph")


def _np(v) -> np.ndarray:
    if torch.is_tensor(v):
        return v.detach().cpu().numpy()
    return np.asarray(v)


@torch.no_grad()
def collect_embeddings(model, loader: Iterable, max_batches: Optional[int] = None,
                       device: Optional[torch.device | str] = None) -> Dict[str, np.ndarray]:
    """Eval-mode pass over ``loader`` (PyG batches) with ``return_details=True``.

    ``model``: a ``LitG4Model`` (``LitRegressionModel`` / ``LitClassificationModel``) or a
    bare ``MultiModalModel``.
    """
    was_training = model.training
    model.eval()
    if device is not None:
        model.to(device)
    dev = next(model.parameters()).device
    postprocess = getattr(model, "postprocess", None)  # the model knows its own output units
    threshold = getattr(model, "detection_threshold", None)
    chunks: Dict[str, list] = {}

    def add(key, value):
        if value is not None:
            chunks.setdefault(key, []).append(_np(value))

    try:
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            batch = batch.to(dev)
            out, det = model(batch, return_details=True)
            out = out.view(-1)
            for k in _STAGES:
                add(k, det.get(k))
            if det.get("gate") is not None:
                add("gate_mean", det["gate"].mean(dim=1))
            if isinstance(det.get("global"), dict) and "clouds" in det["global"]:
                add("clouds", det["global"]["clouds"])
            add("output", out)
            pred = postprocess(out) if postprocess is not None else out
            add("prediction", pred)
            if threshold is not None:
                add("is_g4", pred > threshold)
            add("y", batch.y.view(-1).float())
            add("num_nodes", torch.bincount(batch.batch, minlength=batch.num_graphs))
            for k in _META:
                v = getattr(batch, k, None)
                if v is not None:
                    add(k, v)
    finally:
        model.train(was_training)
    return {k: np.concatenate(v) for k, v in chunks.items()}


def project(X: np.ndarray, method: str = "pca", n_components: int = 2, standardize: bool = True,
            seed: int = 0, **kwargs: Any):
    """Low-dimensional projection of ``X`` [N, D] (3-D inputs are flattened to [N, -1]).

    method: ``"pca"`` (numpy SVD; ``info`` holds ``explained_variance_ratio`` and
    ``components``), ``"tsne"`` (scikit-learn) or ``"umap"`` (needs ``umap-learn``).
    Returns ``(Y [N, n_components], info dict)``.
    """
    X = np.asarray(X, dtype=np.float64).reshape(len(X), -1)
    mean, std = X.mean(0), X.std(0)
    if standardize:
        X = (X - mean) / np.where(std > 0, std, 1.0)
    else:
        X = X - mean
    method = method.lower()
    if method == "pca":
        _, s, vt = np.linalg.svd(X, full_matrices=False)
        var = s ** 2
        info = {"explained_variance_ratio": var[:n_components] / var.sum(),
                "components": vt[:n_components]}
        return X @ vt[:n_components].T, info
    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=n_components, random_state=seed, **kwargs).fit_transform(X), {}
    if method == "umap":
        try:
            import umap
        except ImportError as e:  # pragma: no cover
            raise ImportError("method='umap' needs `pip install umap-learn` in env g4gnn") from e
        return umap.UMAP(n_components=n_components, random_state=seed, **kwargs).fit_transform(X), {}
    raise ValueError(f"unknown method {method!r} (pca | tsne | umap)")


def plot_projection(Y: np.ndarray, color: Optional[np.ndarray] = None, ax=None, title: str = "",
                    cmap: str = "viridis", s: float = 4.0, label: str = ""):
    """2-D scatter of a projection, coloured by ``color`` (e.g. ``emb["y"]``,
    ``emb["gate_mean"]``, ``emb["prediction"]``). Returns the matplotlib Axes."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))
    sc = ax.scatter(Y[:, 0], Y[:, 1], c=color, cmap=cmap if color is not None else None, s=s,
                    linewidths=0, alpha=0.8)
    if color is not None:
        plt.colorbar(sc, ax=ax, label=label)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def save_embeddings(path, emb: Dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **emb)


def load_embeddings(path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}
