"""Synthetic genome -> store -> index -> dataset -> DataLoader, no external data."""

import numpy as np
import pytest
from torch_geometric.loader import DataLoader

from g4shape.data.genome_store import GenomeStore
from g4shape.data.g4seq_index import G4SeqIndex
from g4shape.data.window_dataset import G4WindowDataset, load_window_bundle
from g4shape.features.config import FeatureConfig
from g4shape.features.g_run_graph import extract_graph
from g4shape.features.sequence import decode, encode, reverse_complement

COMP = str.maketrans("ACGTN", "TGCAN")


def _revcomp(s):
    return s.translate(COMP)[::-1]


def test_split_counts(data):
    _, _, counts = data
    assert counts == {"train": 10, "val": 10, "test": 10}  # 5 rows x 2 strands per chromosome


def test_window_content_padding_and_strand(data):
    d, seqs, _ = data
    g = GenomeStore(d / "genome")
    s = seqs["chr1"].upper()
    w = g.window(g.chrom_id("chr1"), 300, 315, flank=100)
    assert decode(w) == s[200:415]
    w = g.window(g.chrom_id("chr1"), 5, 20, flank=100)           # left edge
    assert decode(w) == "N" * 95 + s[0:120]
    w = g.window(g.chrom_id("chr1"), 1480, 1495, flank=100)      # right edge
    assert decode(w) == s[1380:1500] + "N" * 95
    w = g.window(g.chrom_id("chr1"), 300, 315, flank=100, reverse=True)
    assert decode(w) == _revcomp(s[200:415])
    # lower case and N in the source FASTA
    assert decode(g.window(g.chrom_id("chr3"), 0, 4)) == "ACGN"


def test_dataset_sample_matches_direct_extraction(data):
    d, seqs, _ = data
    ds = G4WindowDataset(d / "genome", d / "index", "val")  # chr2
    assert len(ds) == 10
    cfg = FeatureConfig()
    for i in range(len(ds)):
        item = ds[i]
        chrom, start, strand, score = ds.index.row(i)
        s = seqs["chr2"].upper()
        win = "N" * max(100 - start, 0) + s[max(start - 100, 0):start + 115]
        win = win + "N" * (215 - len(win))
        if strand == 1:
            win = _revcomp(win)
        ref = extract_graph(encode(win), cfg)
        np.testing.assert_array_equal(item.x.numpy(), ref["node_features"])
        np.testing.assert_array_equal(item.edge_index.numpy(), ref["edge_index"])
        assert item.y.item() == pytest.approx(score)
        assert item.chrom == "chr2" and item.strand == ("+" if strand else "-")


def test_no_g_window_gets_placeholder_node(data):
    d, _, _ = data
    ds = G4WindowDataset(d / "genome", d / "index", "val", flank=0,
                         feature_config=FeatureConfig(min_run_len=15))  # no 15-G run anywhere
    item = ds[0]
    assert item.x.shape == (1, ds.node_feature_dim) and not item.had_graph
    assert item.edge_index.shape == (2, 0) and float(item.x.abs().sum()) == 0.0


def test_dataloader_with_workers_and_subsample(data):
    d, _, _ = data
    bundle = load_window_bundle(d / "genome", d / "index", max_train=6, max_val=4, max_test=3, seed=1)
    assert len(bundle["train_dataset"]) == 6 and len(bundle["val_dataset"]) == 4
    assert len(bundle["test_dataset"]) == 3
    loader = DataLoader(bundle["train_dataset"], batch_size=4, num_workers=2, shuffle=True)
    batches = list(loader)
    assert sum(b.num_graphs for b in batches) == 6
    b = batches[0]
    assert b.x.shape[1] == bundle["node_dim"] and b.edge_attr.shape[1] == bundle["edge_dim"]
    assert b.y.dtype.is_floating_point and b.batch.max() == b.num_graphs - 1


def test_embeddings_end_to_end(data, tmp_path):
    """Real windows -> regression model -> per-sample embeddings + projections."""
    import torch
    from g4shape.analysis.embeddings import collect_embeddings, load_embeddings, project, save_embeddings
    from g4shape.lightning.build import build_lit_model

    d, _, _ = data
    bundle = load_window_bundle(d / "genome", d / "index")
    torch.manual_seed(0)
    model = build_lit_model(bundle["node_dim"], bundle["edge_dim"], hidden_dim=16,
                            y_mean=50.0, y_std=40.0,
                            graph_encoder_kwargs={"num_layers": 1, "num_heads": 2})
    loader = DataLoader(bundle["test_dataset"], batch_size=4)
    emb = collect_embeddings(model, loader)
    n = len(bundle["test_dataset"])
    for k in ("x_graph", "penultimate"):
        assert emb[k].shape == (n, 16) and np.isfinite(emb[k]).all(), k
    assert emb["y"].shape == emb["prediction"].shape == (n,)
    np.testing.assert_allclose(emb["prediction"], emb["output"] * 40.0 + 50.0, rtol=1e-5)
    assert set(emb["chrom"]) == {"chr1"} and set(emb["strand"]) <= {"+", "-"}
    assert emb["num_nodes"].shape == (n,) and model.training  # mode restored

    Y, info = project(emb["x_graph"], "pca")
    assert Y.shape == (n, 2) and 0 < info["explained_variance_ratio"].sum() <= 1 + 1e-9
    Y, _ = project(emb["penultimate"], "tsne", perplexity=3)
    assert Y.shape == (n, 2)
    save_embeddings(tmp_path / "e.npz", emb)
    back = load_embeddings(tmp_path / "e.npz")
    np.testing.assert_array_equal(back["x_graph"], emb["x_graph"])


def test_dataset_item_is_featurize_of_its_window(data):
    """Both datasets share one featurization: an index row == featurize(its window codes)."""
    import torch
    from g4shape.data.featurize import featurize

    d, _, _ = data
    ds = G4WindowDataset(d / "genome", d / "index", "train")
    g = GenomeStore(d / "genome")
    for i in range(len(ds)):
        item = ds[i]
        chrom, start, strand, _ = ds.index.row(i)
        ref = featurize(g.window(chrom, start, start + 15, 100, reverse=strand == 1), ds.cfg)
        for k in ("x", "edge_index", "edge_attr"):
            assert torch.equal(item[k], ref[k]), (i, k)
        assert item.had_graph == ref.had_graph
