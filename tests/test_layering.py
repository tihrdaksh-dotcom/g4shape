"""Package layering: ``src.models`` must stay Lightning-free (``lightning`` -> ``models`` only)."""

import subprocess
import sys
from pathlib import Path

import torch

from src.models.factory import build_components
from src.models.multimodel import MultiModalModel

ROOT = Path(__file__).resolve().parents[1]


def test_models_do_not_import_lightning():
    modules = sorted(".".join(p.relative_to(ROOT / "src").with_suffix("").parts)
                     for p in (ROOT / "src" / "g4shape" / "models").rglob("*.py"))
    code = ("import sys\n" + "".join(f"import {m}\n" for m in modules)
            + "bad = sorted(m for m in sys.modules if m.split('.')[0] == 'pytorch_lightning'"
              " or m.startswith('src.lightning'))\n"
            + "assert not bad, bad\n")
    res = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_bare_network_matches_lit_model():
    from src.lightning.build import build_lit_model

    kw = dict(node_dim=5, edge_dim=3, global_dim=8, hidden_dim=16,
              global_encoder_kwargs={"num_radii": 2})
    torch.manual_seed(0)
    net = MultiModalModel(**build_components(**kw))
    torch.manual_seed(0)
    lit = build_lit_model(**kw)
    a, b = net.state_dict(), lit.model.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# Allowed src-internal imports per layer (a layer may import only the ones listed).
# features: numpy feature maths | data: genome/index/datasets | models: nn.Modules
# lightning: training objectives | analysis: post-hoc tools | resources: GPU/CPU planning
# config/train: entry points
ALLOWED = {
    "features": {"features"},
    "data": {"data", "features"},
    "models": {"models"},
    "lightning": {"lightning", "models"},
    "analysis": {"analysis"},
    "resources": set(),
    "config": {"features", "lightning", "models", "resources"},
    "train": {"config", "data", "lightning", "resources"},
}


def _src_imports(path):
    import ast

    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                 else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
        for n in names:
            yield n


def test_layer_imports_follow_the_allowed_map():
    bad = []
    for path in sorted((ROOT / "src" / "g4shape").rglob("*.py")):
        layer = path.relative_to(ROOT / "src" / "g4shape").parts[0].removesuffix(".py")
        assert layer in ALLOWED, f"new top-level module {layer!r}: add it to ALLOWED"
        for name in _src_imports(path):
            parts = name.split(".")
            if parts[0] == "g4shape" and len(parts) > 1 and parts[1] not in ALLOWED[layer]:
                bad.append(f"{path.relative_to(ROOT)} imports {name}")
    assert not bad, "\n".join(bad)


def test_src_and_tests_are_self_contained():
    """No import or sys.path hack pointing at reference/ or rework/ (or any path outside)."""
    bad = []
    for path in sorted([*(ROOT / "src" / "g4shape").rglob("*.py"), *(ROOT / "tests").rglob("*.py")]):
        if path.name == "test_layering.py":
            continue
        text = path.read_text()
        for needle in ("sys.path", "reference/", "rework/"):  # module imports: AST check below
            if needle in text:
                bad.append(f"{path.relative_to(ROOT)} mentions {needle!r}")
        for name in _src_imports(path):
            if name.split(".")[0] in ("reference", "rework", "processor"):
                bad.append(f"{path.relative_to(ROOT)} imports {name}")
    assert not bad, "\n".join(bad)
