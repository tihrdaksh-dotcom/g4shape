"""Experiment configuration: everything one training run needs, in one YAML file.

    data:      where the genome store / G4-seq index live, window flank, row caps, loaders
    features:  ``FeatureConfig`` (G-run graph + topology)
    model:     stage names + per-stage kwargs (``src.models.factory``), width, dropout
    task:      target standardisation, G4 label, threshold grid
    optim:     optimizer / lr schedule overrides (``src.lightning.optim.TRANSG4_OPTIM``)
    trainer:   epochs, seed, device override, output directory
    resources: how much of a shared machine to use (free GPUs only, idle CPU share, niceness)

Defaults reproduce TransG4's setup (215-nt windows, chr1 test / chr2 val via the index,
regression, Adam eps=1e-7 + exponential decay 0.99 / 10k steps, 1 epoch, seed 42, batch 256).
Unknown sections or keys raise (with the closest valid name), like the model factory, so
a typo in a YAML file cannot silently fall back to a default.

``ExperimentConfig.to_dict()`` is stored in every checkpoint (``hparams["experiment"]``),
so a trained model always carries the feature settings it was trained with.
"""

from __future__ import annotations

import difflib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import yaml

from g4shape.features.config import FeatureConfig
from g4shape.lightning.optim import resolve_optim
from g4shape.models.factory import DEFAULTS
from g4shape.resources import ResourceConfig

__all__ = ["DataConfig", "ModelConfig", "TaskConfig", "TrainerConfig", "ExperimentConfig"]


@dataclass
class DataConfig:
    genome_dir: str = ""                    # build_genome_store output (required to train)
    index_dir: str = ""                     # build_g4seq_index output (required to train)
    flank: int = 100                        # TransG4: 100 + 15 + 100 = 215 nt
    reverse_strand: Optional[str] = "+"     # TransG4 reverse-complements '+' rows
    use_topology: bool = True
    max_train: Optional[int] = None         # seeded random row caps (genome scale)
    max_val: Optional[int] = None           # validation runs every epoch: cap it on hg19
    max_test: Optional[int] = None          # None = whole test chromosome (paper numbers)
    subsample_seed: int = 0
    batch_size: int = 256                   # TransG4 train batch; GLOBAL (split over GPUs)
    eval_batch_size: int = 512              # TransG4 test batch; global
    num_workers: Any = "auto"               # per rank; "auto": from the idle CPU budget


@dataclass
class ModelConfig:
    hidden_dim: int = 64
    dropout: float = 0.2
    graph_encoder: str = DEFAULTS["graph_encoder"]
    global_encoder: str = DEFAULTS["global_encoder"]
    fusion: str = DEFAULTS["fusion"]
    classifier: str = DEFAULTS["classifier"]
    graph_encoder_kwargs: dict = field(default_factory=dict)
    global_encoder_kwargs: dict = field(default_factory=dict)
    fusion_kwargs: dict = field(default_factory=dict)
    classifier_kwargs: dict = field(default_factory=dict)


@dataclass
class TaskConfig:
    label_threshold: Optional[float] = 35.0  # G4 = mismatch score > 35
    y_mean: Optional[float] = None          # None: computed from the train split
    y_std: Optional[float] = None
    threshold_grid: Any = None              # None: task default (TransG4 30.0..34.9 / 0.01..0.99);
                                            # a list, or {start, stop, step} (metrics.resolve_threshold_grid)
    aux_g4_weight: float = 0.0              # regression: + w * BCE(G4) on the same output (0 = off)
    aux_g4_pos_weight: float = 1.0


@dataclass
class TrainerConfig:
    max_epochs: int = 1                     # TransG4 trains one epoch
    seed: int = 42                          # TransG4's random seed
    accelerator: str = "auto"               # "auto": free GPUs (see resources), else CPU; "cpu" forces CPU
    devices: Any = "auto"                   # "auto" | max number of GPUs | list of GPU indices
    strategy: str = "auto"                  # "auto": ddp when > 1 device
    precision: Any = "32-true"
    gradient_clip_val: Optional[float] = None
    log_every_n_steps: int = 50
    checkpoint_every_minutes: float = 30.0  # resume.ckpt this often (<= 0: only when stopped)
    out_dir: str = "runs"
    run_name: Optional[str] = None          # None: <timestamp>_<graph>-<global>


_SECTIONS = {"data": DataConfig, "model": ModelConfig, "task": TaskConfig, "trainer": TrainerConfig,
             "resources": ResourceConfig}


def _suggest(key, options) -> str:
    close = difflib.get_close_matches(key, sorted(options), n=1)
    return f" (did you mean {close[0]!r}?)" if close else ""


def _section(cls, data: Optional[dict], where: str):
    data = dict(data or {})
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"unknown key(s) in {where}: "
                         + ", ".join(f"{k!r}{_suggest(k, known)}" for k in unknown))
    return cls(**data)


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    optim: dict = field(default_factory=dict)  # overrides of TRANSG4_OPTIM
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)

    def __post_init__(self):
        if self.data.reverse_strand not in ("+", "-", None):
            raise ValueError("data.reverse_strand must be '+', '-' or null")
        self.optim = {k: v for k, v in resolve_optim(self.optim).items()}  # validates keys
        self.optim["betas"] = list(self.optim["betas"])  # YAML-friendly

    # ---- (de)serialisation ------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "ExperimentConfig":
        data = dict(data or {})
        known = {"data", "features", "model", "task", "optim", "trainer", "resources"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError("unknown config section(s): "
                             + ", ".join(f"{k!r}{_suggest(k, known)}" for k in unknown))
        return cls(
            data=_section(DataConfig, data.get("data"), "data"),
            features=FeatureConfig.from_dict(dict(data.get("features") or {})),
            model=_section(ModelConfig, data.get("model"), "model"),
            task=_section(TaskConfig, data.get("task"), "task"),
            optim=dict(data.get("optim") or {}),
            trainer=_section(TrainerConfig, data.get("trainer"), "trainer"),
            resources=_section(ResourceConfig, data.get("resources"), "resources"),
        )

    def to_dict(self) -> dict:
        out = {name: asdict(getattr(self, name)) for name in _SECTIONS}
        features = self.features.to_dict()
        features["topo_clouds"] = list(features["topo_clouds"])
        out.update(features=features, optim=dict(self.optim))
        return out

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()) or {})

    def to_yaml(self, path: Optional[str | Path] = None) -> str:
        text = yaml.safe_dump(self.to_dict(), sort_keys=False)
        if path is not None:
            Path(path).write_text(text)
        return text

    def with_overrides(self, overrides) -> "ExperimentConfig":
        """``["model.hidden_dim=32", "optim.scheduler=plateau", ...]`` (values parsed as YAML)."""
        data = self.to_dict()
        for item in overrides or []:
            key, sep, raw = str(item).partition("=")
            if not sep or "." not in key:
                raise ValueError(f"override must look like 'section.key=value', got {item!r}")
            section, name = key.split(".", 1)
            if section not in data:
                raise ValueError(f"unknown config section {section!r}{_suggest(section, data)}")
            data[section][name] = yaml.safe_load(raw)
        return ExperimentConfig.from_dict(data)
