"""Code-array feature extractor: graph == brute-force reference."""

import numpy as np
import pytest

from g4shape.features.config import FeatureConfig
from g4shape.features.g_run_graph import edge_feature_names, extract_graph, node_feature_names
from g4shape.features.sequence import decode, encode


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


def test_decode_roundtrip():
    for seq in _sequences():
        assert decode(encode(seq)) == normalize_sequence(seq)
