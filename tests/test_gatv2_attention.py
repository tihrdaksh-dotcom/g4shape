"""Opt-in GATv2 attention of the diffusion encoder (``attention="gatv2"``): the default stays the
DADGNN/GATv1 scorer, GATv2 matches its definition, and its neighbour ranking depends on the
destination and edge features (which GATv1's additive score cannot express)."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from g4shape.config import ExperimentConfig
from g4shape.data.featurize import featurize
from g4shape.features.sequence import encode
from g4shape.models.graph_encoders.diffusion_conv import DiffusionAttentionConv
from g4shape.models.graph_encoders.diffusion_gat import DiffusionGATEncoder
from g4shape.train import build_model


def rand_graph(n=7, e=20, d=8, de=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    ei = torch.randint(0, n, (2, e), generator=g)
    ei = torch.cat([ei, torch.arange(n).repeat(2, 1)], dim=1)  # self-loops: every node has an in-edge
    ea = torch.randn(ei.size(1), de, generator=g)
    return x, ei, ea


def test_default_is_gat_and_unchanged():
    """attention="gat" (default) keeps the original parameters and RNG use, so run1 / baseline
    checkpoints load and a seeded model is initialised exactly as before."""
    torch.manual_seed(0)
    a = DiffusionAttentionConv(8, 8, edge_dim=5)
    torch.manual_seed(0)
    b = DiffusionAttentionConv(8, 8, edge_dim=5, attention="gat")
    assert a.attention == "gat"
    assert set(a.state_dict()) == {"edge_lin.weight", "edge_lin.bias", "attn_nn.weight",
                                   "attn_nn.bias", "temp"}
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k])


def test_gatv2_matches_definition():
    torch.manual_seed(1)
    x, ei, ea = rand_graph()
    conv = DiffusionAttentionConv(8, 8, edge_dim=5, k=2, attention="gatv2", attention_dim=6,
                                  negative_slope=0.2).eval()
    assert conv.attn_nn is None and conv.attn_src.out_features == 6
    with torch.no_grad():
        conv.attn_dst.bias.normal_()
        out, attn = conv(x, ei, ea, return_attention=True)
        src, dst = ei
        emb = conv.edge_lin(ea)
        h, ref = x, conv.temp[0] * x
        for t in range(conv.k):
            z = conv.attn_src.weight @ h[src].T + conv.attn_dst.weight @ h[dst].T \
                + conv.attn_dst.bias[:, None] + conv.attn_edge.weight @ emb.T
            e = (conv.attn_vec.weight @ F.leaky_relu(z, 0.2)).flatten()
            w = torch.zeros_like(e)
            for n in range(x.size(0)):  # softmax over the in-edges of n
                m = dst == n
                w[m] = torch.softmax(e[m], dim=0)
            assert torch.allclose(attn[t], w, atol=1e-5)
            h = torch.zeros_like(h).index_add_(0, dst, (h[src] + emb) * w[:, None])
            ref = ref + conv.temp[t + 1] * h
    assert torch.allclose(out, ref, atol=1e-5)


def _ranking_depends_on_destination(conv, x, ei, ea):
    """Two destinations with the same in-neighbours and edge features: does their softmax
    ranking of those neighbours differ?"""
    with torch.no_grad():
        _, attn = conv(x, ei, ea, return_attention=True)
    a0, a1 = attn[0][ei[1] == 0], attn[0][ei[1] == 1]
    return not torch.equal(torch.argsort(a0), torch.argsort(a1))


@pytest.mark.parametrize("attention,expected", [("gat", False), ("gatv2", True)])
def test_only_gatv2_ranks_neighbours_per_destination(attention, expected):
    """GATv1 gives every destination the same neighbour order (dst term cancels in the softmax);
    GATv2 can rank them differently. Edge feats are identical per source, so they can't differ."""
    torch.manual_seed(3)
    n_src, d = 6, 8
    x = torch.randn(n_src + 2, d) * 3
    src = torch.arange(2, 2 + n_src).repeat(2)
    dst = torch.cat([torch.zeros(n_src), torch.ones(n_src)]).long()
    ei = torch.stack([src, dst])
    ea = torch.randn(n_src, 5).repeat(2, 1)
    differs = False
    for seed in range(20):  # GATv2 differs for almost any init; GATv1 never does
        torch.manual_seed(seed)
        conv = DiffusionAttentionConv(d, d, edge_dim=5, k=1, attention=attention,
                                      negative_slope=0.2).eval()
        differs |= _ranking_depends_on_destination(conv, x, ei, ea)
    assert differs is expected


def test_gatv2_attention_uses_edge_attr():
    torch.manual_seed(4)
    x, ei, ea = rand_graph()
    conv = DiffusionAttentionConv(8, 8, edge_dim=5, k=1, attention="gatv2").eval()
    with torch.no_grad():
        _, a = conv(x, ei, ea, return_attention=True)
        _, b = conv(x, ei, ea + torch.randn_like(ea), return_attention=True)
    assert not torch.allclose(a, b)


def test_gatv2_without_edge_features():
    x, ei, _ = rand_graph()
    conv = DiffusionAttentionConv(8, 8, edge_dim=None, attention="gatv2")
    assert conv.attn_edge is None
    assert conv(x, ei).shape == (7, 8)


def test_unknown_attention_raises():
    with pytest.raises(ValueError, match="attention"):
        DiffusionAttentionConv(8, 8, attention="gatv3")


def test_encoder_passes_attention_through():
    enc = DiffusionGATEncoder(node_dim=9, edge_dim=5, hidden_dim=16, attention="gatv2", attention_dim=4)
    heads = [h for layer in enc.convs for h in layer.heads]
    assert heads and all(h.attention == "gatv2" and h.attn_src.out_features == 4 for h in heads)


@pytest.mark.parametrize("cond", ["k", "pds"])
def test_experiment_config_builds_and_runs(cond):
    """configs/fast_<c>_gatv2_pos.yaml = baseline config + GATv2 + position features only."""
    base = ExperimentConfig.from_yaml(f"configs/fast_{cond}.yaml")
    exp = ExperimentConfig.from_yaml(f"configs/fast_{cond}_gatv2_pos.yaml")
    assert exp.features.position_encoding and not base.features.position_encoding
    assert exp.model.graph_encoder_kwargs == {"attention": "gatv2", "attention_dim": 16,
                                              "negative_slope": 0.2}
    b, e = base.to_dict(), exp.to_dict()
    for d in (b, e):
        d["features"].pop("position_encoding"), d["model"].pop("graph_encoder_kwargs")
        d["trainer"].pop("run_name")
    assert b == e  # nothing else differs from the baseline

    model = build_model(exp).eval()
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list("ACGT"), 100)) + "GGGTTAGGGTTAGGGTTAGGG" + "".join(rng.choice(list("ACGT"), 94))
    batch = Batch.from_data_list([featurize(encode(seq), exp.features, True)])
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (1, 1) and torch.isfinite(out).all()
    heads = [h for layer in model.model.graph_encoder.convs for h in layer.heads]
    assert all(h.attention == "gatv2" for h in heads)
