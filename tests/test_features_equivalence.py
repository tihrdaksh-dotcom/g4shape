"""Code-array feature extractors: graph == brute-force reference; topology == the K-mer\ntopology paper's reference code (Hozumi & Wei 2024) on the same radius grid."""

import contextlib
import io

import numpy as np
import pytest
from KmerTopology._compute_topology import compute_filtration_topology
from KmerTopology.kmer_topology import KmerTopology

from g4shape.features.config import G3_4MERS, FeatureConfig
from g4shape.features.g_run_graph import edge_feature_names, extract_graph, node_feature_names
from g4shape.features.sequence import decode, encode
from g4shape.features.topology import extract_topology, topology_dim, topology_feature_names


def normalize_sequence(seq: str) -> str:
    """Reference alphabet normalisation (string version): drop whitespace, upper-case,
    U -> T, any other non-ACGT character -> N."""
    s = "".join(seq.split()).upper().replace("U", "T")
    return "".join(c if c in "ACGT" else "N" for c in s)


def _sequences():
    rng = np.random.default_rng(1)
    seqs = [
        "".join(rng.choice(list("ACGT"), 215, p=[0.29, 0.21, 0.21, 0.29])) for _ in range(25)
    ]
    seqs += [
        "G" * 215, "C" * 215, "A" * 215, "N" * 215, "GGG", "G", "",
        "GGGTTAGGGTTAGGGTTAGGGTTA" * 8, "ACGTN" * 43, "cccgggUUU" * 20,
        "".join(rng.choice(list("ACGTN"), 215)),
        "GGGAGGGAGGGAGGGA" + "ACGT" * 50 + "CCCTCCCTCCCTCCC",
        "G" * 3 + "N" * 30 + "G" * 4,
    ]
    return seqs


def _brute_graph(seq, cfg):
    """Slow reference of the G-run graph definition (regex runs, python loops)."""
    import re
    s = normalize_sequence(seq)
    runs = [(m.start(), m.end()) for m in re.finditer("G+", s) if m.end() - m.start() >= cfg.min_run_len]
    classes = list(range(cfg.min_run_len, 5))

    def onehot(L):
        return [float(L == k) for k in classes] + [float(L >= 5)]

    edges, feats = [], []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            gap = runs[j][0] - runs[i][1]
            if gap > cfg.edge_radius:
                continue
            loop = s[runs[i][1]:runs[j][0]]
            comp = [loop.count(b) / gap for b in "ACGT"]
            li, lj = runs[i][1] - runs[i][0], runs[j][1] - runs[j][0]
            for (u, v, d, lu, lv) in ((i, j, 1.0, li, lj), (j, i, -1.0, lj, li)):
                edges.append((u, v))
                feats.append([d, float(gap), float(j - i == 1)] + comp + onehot(lu) + onehot(lv))
    return runs, edges, feats


@pytest.mark.parametrize("seq", _sequences())
def test_graph_matches_reference(seq):
    cfg = FeatureConfig()
    runs, edges, feats = _brute_graph(seq, cfg)
    new = extract_graph(encode(seq), cfg)
    assert new["n_nodes"] == len(runs) and new["n_edges"] == len(edges)
    assert new["node_features"].shape[1] == len(node_feature_names(cfg)) == 33
    assert new["edge_features"].shape[1] == len(edge_feature_names(cfg)) == 15
    if runs:
        np.testing.assert_array_equal(new["run_starts"], [r[0] for r in runs])
        np.testing.assert_array_equal(new["run_ends"], [r[1] for r in runs])
    if edges:
        np.testing.assert_array_equal(new["edge_index"], np.array(edges).T)
        np.testing.assert_allclose(new["edge_features"], np.array(feats, np.float32), atol=1e-6)


# ---- K-mer topology: src.features.topology wraps the authors' KmerTopology package


def _brute_positions(s, cloud):
    """1-based starts of every k-mer of the cloud, by plain string scanning."""
    pos = set()
    for km in cloud.split("|"):
        pos |= {i + 1 for i in range(len(s) - len(km) + 1) if s[i:i + len(km)] == km}
    return np.array(sorted(pos), dtype=float)


_TOPO_CFGS = [FeatureConfig(),
              FeatureConfig(topo_clouds=("A", "C", "G", "T"), max_step=20),
              FeatureConfig(topo_clouds=("GG", "GAGG|GGAG|GGGA|AGGG", "G|GG"), step_size=3.0, max_step=12)]


@pytest.mark.parametrize("cfg", _TOPO_CFGS, ids=["default", "1mers", "unions_step3"])
@pytest.mark.parametrize("seq", _sequences())
def test_topology_wraps_package(seq, cfg, capsys):
    new = extract_topology(encode(seq), cfg)
    assert capsys.readouterr().out == ""  # the package's "not valid dna" prints are silenced
    assert new.dtype == np.float32
    assert new.shape == (topology_dim(cfg),) == (len(topology_feature_names(cfg)),)
    s = normalize_sequence(seq)
    curves = [compute_filtration_topology(_brute_positions(s, c), cfg.step_size, cfg.max_step)
              for c in cfg.topo_clouds]
    ref = np.concatenate([b for b, _ in curves] + [e for _, e in curves])
    np.testing.assert_array_equal(new, ref.astype(np.float32))
    if cfg.topo_clouds == ("A", "C", "G", "T"):  # all 1-mers == the package's KmerTopology
        with contextlib.redirect_stdout(io.StringIO()):
            pkg = np.concatenate(KmerTopology(s, 1, cfg.step_size, cfg.max_step))
        np.testing.assert_array_equal(new, pkg.astype(np.float32))


def test_decode_roundtrip():
    for seq in _sequences():
        assert decode(encode(seq)) == normalize_sequence(seq)


def test_topology_default_layout():
    cfg = FeatureConfig()
    assert topology_dim(cfg) == 7 * 2 * 20 == 280
    assert cfg.topo_clouds[6] == G3_4MERS and len(G3_4MERS.split("|")) == 12
    assert all(km.count("G") == 3 and len(km) == 4 for km in G3_4MERS.split("|"))
    names = topology_feature_names(cfg)
    assert names[0] == "betti_A_r0.00" and names[140] == "eigmin_A_r0.00"
    v = extract_topology(encode("GGGAGGGTGGG"), cfg).reshape(2, 7, 20)  # [betti|eigmin] x 7 clouds x radius
    # G at 1-3, 5-7, 9-11 (1-based) -> 9 points; consecutive gaps 1,1,2,1,1,2,1,1:
    # r=0: 9 isolated G's; r=1.05: the 3 G-runs; r>=2.1: one cluster (1-nt loops bridged)
    np.testing.assert_array_equal(v[0, 2, :4], [9, 3, 1, 1])
    assert v[1, 2, 0] == 0 and v[1, 2, 1] > 0
    np.testing.assert_array_equal(v[0, 0, :2], [1, 1])  # a single A -> Betti 1 at every radius
    np.testing.assert_array_equal(v[:, 1], 0)  # no C -> all-zero C curves
    # GG starts 1,2,5,6,9,10: r=0 six points; r=1.05 the 3 runs; GG gap across a 1-nt loop is 3
    np.testing.assert_array_equal(v[0, 4, :4], [6, 3, 3, 1])
    # every 4-mer start 1..8 has exactly 3 G's (GGGA GGAG GAGG AGGG GGGT GGTG GTGG TGGG):
    # 8 consecutive starts -> r=0: 8 points, r>=1.05: one cluster
    np.testing.assert_array_equal(v[0, 6, :3], [8, 1, 1])
