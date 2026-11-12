"""Opt-in graph extras (src/features/graph_extras.py) and the anchor_gat encoder."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from g4shape.data.featurize import feature_dims, featurize
from g4shape.features.config import FeatureConfig
from g4shape.features.g_run_graph import extract_graph
from g4shape.features.graph_extras import (extract_graph_full, graph_edge_feature_names,
                                       graph_node_feature_names, target_bin)
from g4shape.features.sequence import encode
from g4shape.models.factory import build_components
from g4shape.models.multimodel import MultiModalModel

ALL = FeatureConfig(position_encoding=True, gap_encoding="bucket", anchor_node=True,
                    onehot_context=6, loop_end_bases=2)


def _window(seed=0, n=215, motif="GGGTTAGGGTTAGGGTTAGGG", at=80):
    rng = np.random.default_rng(seed)
    seq = "".join(rng.choice(list("ACGT"), n))
    return encode(seq[:at] + motif + seq[at + len(motif):])


def test_defaults_are_the_base_graph():
    codes = _window()
    cfg = FeatureConfig()
    a, b = extract_graph(codes, cfg), extract_graph_full(codes, cfg)
    assert a.keys() == b.keys()
    for k in a:
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k]))
    d = featurize(codes, cfg, use_topology=False)
    assert torch.equal(d.x, torch.from_numpy(a["node_features"]))
    assert torch.equal(d.edge_attr, torch.from_numpy(a["edge_features"]))


@pytest.mark.parametrize("cfg", [
    FeatureConfig(position_encoding=True), FeatureConfig(gap_encoding="bucket"),
    FeatureConfig(anchor_node=True), FeatureConfig(onehot_context=4),
    FeatureConfig(loop_end_bases=3), ALL])
def test_dims_match_names(cfg):
    for codes in (_window(), encode("A" * 215), encode("GG" + "A" * 213)):
        g = extract_graph_full(codes, cfg)
        assert g["node_features"].shape[1] == len(graph_node_feature_names(cfg))
        assert g["edge_features"].shape[1] == len(graph_edge_feature_names(cfg))
        assert g["edge_index"].shape[1] == g["edge_features"].shape[0]
        assert np.isfinite(g["node_features"]).all() and np.isfinite(g["edge_features"]).all()
        d = featurize(codes, cfg, use_topology=False)
        dims = feature_dims(cfg)
        assert d.x.shape[1] == dims["node_dim"] and d.edge_attr.shape[1] == dims["edge_dim"]


def test_signed_offsets():
    cfg = FeatureConfig(position_encoding=True)
    lo, hi, centre = target_bin(215, cfg)
    assert (lo, hi, centre) == (100, 115, 107.0)
    codes = encode("A" * 50 + "GGG" + "A" * 104 + "GGG" + "A" * 55)  # runs 5' and 3' of the bin
    g = extract_graph_full(codes, cfg)
    names = graph_node_feature_names(cfg)
    off = g["node_features"][:, names.index("node_offset_scaled")] * 107.5
    assert np.allclose(off, [51 - 107, 158 - 107])
    assert (g["node_features"][:, names.index("in_bin")] == 0).all()
    rbf = g["node_features"][:, names.index("node_rbf_0"):names.index("in_bin")]
    assert rbf[0].argmax() < rbf.shape[1] // 2 < rbf[1].argmax()


def test_gap_buckets_and_scaled_gap():
    cfg = FeatureConfig(gap_encoding="bucket")
    codes = encode("A" * 20 + "GGG" + "T" * 4 + "GGG" + "A" * 185)
    g = extract_graph_full(codes, cfg)
    names = graph_edge_feature_names(cfg)
    e = g["edge_features"]
    assert np.allclose(e[:, names.index("gap_scaled")], 4 / 50)
    assert (e[:, names.index("gap_bucket_4_5")] == 1).all()
    assert e[:, [n.startswith("gap_bucket") for n in names]].sum(1).tolist() == [1, 1]


def test_anchor_structure():
    cfg = FeatureConfig(anchor_node=True, position_encoding=True)
    codes = _window()
    g = extract_graph_full(codes, cfg)
    base = extract_graph(codes, cfg)
    n_runs, a = g["n_runs"], g["anchor_index"]
    assert a == n_runs == base["n_nodes"] and g["n_nodes"] == n_runs + 1
    x, ei = g["node_features"], g["edge_index"]
    assert x[:, -1].tolist() == [0.0] * n_runs + [1.0]
    to_a = ei[:, base["n_edges"]:]
    assert (to_a[1, :n_runs] == a).all() and (to_a[0, n_runs:] == a).all()
    # no G-runs: the anchor alone, had_graph False
    d = featurize(encode("A" * 215), cfg, use_topology=False)
    assert d.x.shape[0] == 1 and d.x[0, -1] == 1 and not d.had_graph


def test_onehot_context_and_loop_ends():
    cfg = FeatureConfig(onehot_context=2, loop_end_bases=1)
    codes = encode("AC" + "GGG" + "TA" + "GGG" + "C" * 206)
    g = extract_graph_full(codes, cfg)
    nn_ = graph_node_feature_names(cfg)
    x = g["node_features"]
    assert x[0, nn_.index("ctx_up2_A")] == 1 and x[0, nn_.index("ctx_up1_C")] == 1
    assert x[0, nn_.index("ctx_down1_T")] == 1 and x[0, nn_.index("ctx_down2_A")] == 1
    en = graph_edge_feature_names(cfg)
    e = g["edge_features"]
    assert (e[:, en.index("loop_first1_T")] == 1).all() and (e[:, en.index("loop_last1_A")] == 1).all()
    # run at the sequence start: upstream context outside the sequence = 0.25
    g2 = extract_graph_full(encode("GGG" + "A" * 212), cfg)
    assert np.allclose(g2["node_features"][0, nn_.index("ctx_up1_A"):nn_.index("ctx_up1_A") + 4], .25)


@pytest.mark.parametrize("readout", ["both", "anchor", "pool"])
def test_anchor_gat_forward(readout):
    cfg = ALL
    datas = [featurize(c, cfg, use_topology=False)
             for c in (_window(0), _window(1), encode("A" * 215))]
    for d in datas:
        d.global_features = torch.zeros(1, 2 * 7 * 20)
    batch = Batch.from_data_list(datas)
    dims = feature_dims(cfg)
    net = MultiModalModel(**build_components(
        dims["node_dim"], dims["edge_dim"], dims["global_dim"], 16, graph_encoder="anchor_gat",
        graph_encoder_kwargs={"readout": readout}, global_layout=dims["global_layout"]))
    out, det = net(batch, return_details=True)
    assert out.shape == (3, 1) and torch.isfinite(out).all()
    g = det["graph"]
    assert g["anchor_mask"].sum() == 3
    assert (g["pool_weights"][g["anchor_mask"]] == 0).all()
    out.sum().backward()


def test_anchor_gat_rejects_graphs_without_anchor():
    datas = [featurize(_window(0), FeatureConfig(), use_topology=False)]
    for d in datas:
        d.global_features = torch.zeros(1, 2 * 7 * 20)
    batch = Batch.from_data_list(datas)
    dims = feature_dims(FeatureConfig())
    net = MultiModalModel(**build_components(
        dims["node_dim"], dims["edge_dim"], dims["global_dim"], 16, graph_encoder="anchor_gat",
        global_layout=dims["global_layout"]))
    with pytest.raises(ValueError, match="anchor"):
        net(batch)
