import torch

from src.models.fusion.gated import GatedFusion
from src.models.heads.classifier import BinaryMLPClassifier


def test_gated_fusion_details_and_scale_invariance():
    torch.manual_seed(0)
    f = GatedFusion(8).eval()
    g, t = torch.randn(5, 8), torch.randn(5, 8)
    fused, det = f(g, t, return_details=True)
    assert fused.shape == det["gate"].shape == (5, 8)
    assert torch.allclose(fused, f(g, t))
    # graph embeddings are node sums; with norm_inputs their scale no longer matters
    assert torch.allclose(f(10 * g, t), fused, atol=1e-4)
    assert not torch.allclose(GatedFusion(8, norm_inputs=False).eval()(10 * g, t),
                              GatedFusion(8, norm_inputs=False).eval()(g, t))


def test_modality_dropout_train_only():
    torch.manual_seed(0)
    f = GatedFusion(8, modality_dropout=1.0)
    g, t = torch.randn(64, 8), torch.randn(64, 8)
    _, det = f.train()(g, t, return_details=True)
    zero_g = (det["x_graph_n"] == 0).all(1)
    zero_t = (det["x_global_n"] == 0).all(1)
    assert torch.all(zero_g ^ zero_t)  # exactly one branch dropped per sample
    _, det = f.eval()(g, t, return_details=True)
    assert not (det["x_graph_n"] == 0).all(1).any()


def test_head_penultimate():
    h = BinaryMLPClassifier(8, hidden_dim=6).eval()
    x = torch.randn(4, 8)
    y, det = h(x, return_details=True)
    assert y.shape == (4, 1) and det["penultimate"].shape == (4, 6) and (det["penultimate"] >= 0).all()
    assert torch.allclose(y, h(x))
