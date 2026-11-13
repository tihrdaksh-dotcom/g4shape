"""Checks the PyG diffusion encoder against a literal port of DADGNN's SingleHeadGATLayer/GATLayer
(``attention_diffusion.py`` of https://github.com/KEAML-JLU/DADGNN, commit b0360ff), written with
explicit per-destination loops (no DGL)."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from g4shape.models.graph_encoders.diffusion_conv import DiffusionAttentionConv
from g4shape.models.graph_encoders.diffusion_gat import DiffusionGATEncoder, DiffusionGATLayer
from g4shape.models.graph_encoders.pooling import WeightAndSumPool
from g4shape.lightning.build import build_lit_model


def ref_single_head(conv, feats, edge_index):
    """DADGNN SingleHeadGATLayer.forward (returns features_two), eval mode, using conv's weights."""
    src, dst = edge_index
    N = feats.size(0)
    two = feats * conv.temp[0]
    z = feats
    for k in range(conv.k):
        e = F.leaky_relu(conv.attn_nn(torch.cat([z[src], z[dst]], dim=1)))  # DGL default slope 0.01
        w = torch.zeros_like(e)
        for n in range(N):  # edge_softmax over in-edges of n
            m = dst == n
            if m.any():
                w[m] = torch.softmax(e[m], dim=0)
        h = torch.zeros_like(z)
        for ei in range(src.numel()):
            h[dst[ei]] += z[src[ei]] * w[ei]
        z = h
        two = two + conv.temp[k + 1] * z
    return two


def rand_graph(n=7, e=20, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    ei = torch.randint(0, n, (2, e), generator=g)
    ei = torch.cat([ei, torch.arange(n).repeat(2, 1)], dim=1)  # self-loops: every node has an in-edge
    return x, ei


@pytest.mark.parametrize("k,alpha", [(1, 0.1), (3, 0.1), (4, 0.3)])
def test_conv_matches_dadgnn_single_head(k, alpha):
    torch.manual_seed(1)
    x, ei = rand_graph()
    conv = DiffusionAttentionConv(8, 8, k=k, alpha=alpha).eval()
    with torch.no_grad():
        conv.attn_nn.bias.normal_()  # exercise the bias too
        out, attn = conv(x, ei, return_attention=True)
        ref = ref_single_head(conv, x, ei)
    assert torch.allclose(out, ref, atol=1e-5)
    assert attn.shape == (k, ei.size(1))
    sums = torch.zeros(x.size(0)).index_add_(0, ei[1], attn[0])
    assert torch.allclose(sums, torch.ones(x.size(0)), atol=1e-5)  # softmax over in-edges


def test_temp_init_matches_dadgnn():
    k, alpha = 3, 0.1
    conv = DiffusionAttentionConv(4, 4, k=k, alpha=alpha)
    t = alpha * (1 - alpha) ** np.arange(k + 1)
    t[-1] = (1 - alpha) ** k
    assert np.allclose(conv.temp.detach().numpy(), t)


def test_layer_mean_and_cat_merge_match_dadgnn():
    torch.manual_seed(2)
    x, ei = rand_graph()
    for merge in ("mean", "cat"):
        layer = DiffusionGATLayer(8, 8, num_heads=3, merge=merge).eval()
        with torch.no_grad():
            heads = [ref_single_head(h, x, ei) for h in layer.heads]
            ref = layer.merge_proj(torch.cat(heads, 1)) if merge == "cat" else torch.stack(heads).mean(0)
            out, attn = layer(x, ei, return_attention=True)
        assert torch.allclose(out, ref, atol=1e-5)
        assert attn.shape == (3, 3, ei.size(1))


def test_weight_and_sum_matches_dgl_definition():
    torch.manual_seed(3)
    pool = WeightAndSumPool(5)
    x = torch.randn(6, 5)
    batch = torch.tensor([0, 0, 0, 1, 1, 2])
    out = pool(x, batch)
    w = torch.sigmoid(pool.gate_nn(x))
    ref = torch.stack([(w[batch == b] * x[batch == b]).sum(0) for b in range(3)])
    assert torch.allclose(out, ref, atol=1e-6)


def make_batch(with_empty=False):
    graphs = []
    for i, n in enumerate([4, 6, 0 if with_empty else 3]):
        e = 0 if n == 0 else 2 * n
        graphs.append(Data(
            x=torch.randn(n, 5),
            edge_index=torch.randint(0, max(n, 1), (2, e)),
            edge_attr=torch.randn(e, 3),
            global_features=torch.randn(1, 6),
            y=torch.tensor([i % 2]),
        ))
    return Batch.from_data_list(graphs)


def test_encoder_defaults_are_dadgnn_and_aligned_with_graphless_records():
    enc = DiffusionGATEncoder(node_dim=5, edge_dim=3, hidden_dim=16).eval()
    assert enc.input_lin1 is not None and isinstance(enc.pool, WeightAndSumPool)
    assert enc.use_edge_attr and enc.add_self_loops
    b = make_batch(with_empty=True)
    out = enc(b.x, b.edge_index, b.edge_attr, b.batch, num_graphs=b.num_graphs)
    assert out.shape == (3, 16)
    assert torch.all(out[2] == 0)  # graph-less record -> zero embedding row, aligned


def test_exact_dadgnn_mode_ignores_edge_attr():
    enc = DiffusionGATEncoder(node_dim=5, edge_dim=3, hidden_dim=16, use_edge_attr=False).eval()
    b = make_batch()
    a = enc(b.x, b.edge_index, b.edge_attr, b.batch)
    c = enc(b.x, b.edge_index, torch.randn_like(b.edge_attr), b.batch)
    assert torch.allclose(a, c)


def test_edge_attr_changes_output_by_default():
    enc = DiffusionGATEncoder(node_dim=5, edge_dim=3, hidden_dim=16).eval()
    b = make_batch()
    a = enc(b.x, b.edge_index, b.edge_attr, b.batch)
    c = enc(b.x, b.edge_index, torch.randn_like(b.edge_attr), b.batch)
    assert not torch.allclose(a, c)


def test_details_and_characterize():
    lit = build_lit_model(node_dim=5, edge_dim=3, hidden_dim=16,
                          graph_encoder_kwargs={"num_layers": 2, "num_heads": 2, "k": 3})
    b = make_batch()
    res = lit.characterize(b)
    g = res["details"]["graph"]
    E = g["edge_index"].size(1)
    assert res["output"].shape == (3, 1) and res["prediction"].shape == res["is_g4"].shape == (3,)
    torch.testing.assert_close(res["prediction"], res["output"].view(-1))  # y_mean 0, y_std 1
    assert len(g["attention"]) == 2 and g["attention"][0].shape == (2, 3, E)
    assert g["diffusion_coeffs"][0].shape == (2, 4)
    assert g["pool_weights"].shape == (b.x.size(0), 1)
    assert len(g["node_embeddings"]) == 3
    assert lit.training  # characterize restores the mode
    # details path == plain path
    lit.eval()
    assert torch.allclose(lit(b), lit(b, return_details=True)[0])


def test_gradients_flow_and_other_pools():
    for pool in ("mean", "add", "weight_and_sum"):
        enc = DiffusionGATEncoder(node_dim=5, edge_dim=3, hidden_dim=16, pool=pool, batch_norm=True,
                                  residual=True, activation="elu")
        b = make_batch()
        enc(b.x, b.edge_index, b.edge_attr, b.batch, num_graphs=3).sum().backward()
        assert all(p.grad is not None for p in enc.convs[0].heads[0].parameters())
