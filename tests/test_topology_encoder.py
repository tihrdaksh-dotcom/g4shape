import numpy as np
import pytest
import torch

from src.features.config import FeatureConfig
from src.features.sequence import encode
from src.features.topology import extract_topology, topology_dim, topology_feature_names, topology_layout
from src.models.factory import build_global_encoder
from src.models.utils import count_parameters
from src.models.global_encoders.mlp import MLPEncoder
from src.models.global_encoders.topo_curve import TopologyCurveEncoder, preprocess_topology

CFG = FeatureConfig()
SEQS = ["GGGAGGGTGGG" + "ACGT" * 51, "A" * 215, "N" * 215, "GGGTTAGGGTTAGGGTTAGGG" * 10 + "ACGTA" * 3]


def _batch():
    return torch.from_numpy(np.stack([extract_topology(encode(s[:215]), CFG) for s in SEQS]))


def test_preprocess_matches_definition():
    x = _batch()
    curves, log_count = preprocess_topology(x, CFG.max_step)
    C = len(CFG.topo_clouds)
    assert curves.shape == (len(SEQS), C, 2, CFG.max_step) and log_count.shape == (len(SEQS), C, 1)
    v = x.view(len(SEQS), 2, C, CFG.max_step)
    n = v[:, 0, :, 0]
    torch.testing.assert_close(log_count[..., 0], torch.log1p(n))
    torch.testing.assert_close(curves[:, :, 0], v[:, 0] / n.clamp(min=1)[..., None])
    torch.testing.assert_close(curves[:, :, 1], torch.log1p(v[:, 1]))
    assert curves[:, :, 0].min() >= 0 and curves[:, :, 0].max() <= 1  # survival in [0, 1]
    assert torch.all(curves[2] == 0) and torch.all(log_count[2] == 0)  # all-N window: empty clouds


def test_encoder_shapes_details_and_gradients():
    enc = TopologyCurveEncoder(topology_dim(CFG), hidden_dim=64, num_radii=CFG.max_step)
    assert enc.num_clouds == len(CFG.topo_clouds)
    x = _batch()
    out, det = enc(x, return_details=True)
    assert out.shape == (len(SEQS), 64) and torch.isfinite(out).all()
    assert det["clouds"].shape == (len(SEQS), enc.num_clouds, 32)
    out.sum().backward()
    assert all(p.grad is not None for p in enc.parameters())


def test_cloud_order_matters():
    """Clouds keep their identity: swapping two clouds' curves changes the embedding."""
    torch.manual_seed(0)
    enc = TopologyCurveEncoder(topology_dim(CFG), num_radii=CFG.max_step).eval()
    x = _batch()[:1]
    v = x.view(1, 2, -1, CFG.max_step).clone()
    v[:, :, [2, 0]] = v[:, :, [0, 2]]  # swap the A and G clouds
    assert not torch.allclose(enc(x), enc(v.reshape(1, -1)))


def test_factory_default_and_param_budget():
    gd = topology_dim(CFG)
    enc = build_global_encoder("topo_curve", global_dim=gd, hidden_dim=64, **topology_layout(CFG))
    assert isinstance(enc, TopologyCurveEncoder)
    assert count_parameters(enc) < count_parameters(MLPEncoder(gd, 64)) / 3


def test_rejects_wrong_layout():
    with pytest.raises(ValueError):
        TopologyCurveEncoder(281, num_radii=20)
    with pytest.raises(ValueError, match="num_radii"):  # no silent default any more
        TopologyCurveEncoder(topology_dim(CFG))
    # max_step=40, 7 clouds -> 560 = 2 * 14 * 20: divisible, so only num_clouds catches it
    cfg = FeatureConfig(max_step=40)
    with pytest.raises(ValueError):
        TopologyCurveEncoder(topology_dim(cfg), num_radii=20, num_clouds=len(cfg.topo_clouds))
    enc = TopologyCurveEncoder(topology_dim(cfg), **topology_layout(cfg))
    assert (enc.num_clouds, enc.num_radii) == (7, 40)


def test_layout_matches_feature_names():
    names = np.array(topology_feature_names(CFG)).reshape(2, *topology_layout(CFG).values())
    assert names[0, 2, 0].startswith("betti_G_") and names[1, 0, -1].startswith("eigmin_A_")
