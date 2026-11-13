"""Feature-extraction settings of the G-run graph (see g_run_graph.py and graph_extras.py)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class FeatureConfig:
    """Tunables of the G-run graph.

    Graph:    min_run_len, edge_radius, context_bases, flank_window
    Graph extras (graph_extras.py; all off by default = the original graph, bit for bit):
              position_encoding (signed offset of runs / loops from the target bin, RBF),
              gap_encoding ("raw" | "bucket": loop-length classes + gap / edge_radius),
              loop_end_bases (one-hot of the first / last N loop bases); target_len (bin length, centred in the
              sequence), pos_rbf / pos_sigma (number / width in nt of the RBF centres).
    """

    min_run_len: int = 2
    edge_radius: int = 50
    context_bases: int = 2
    flank_window: int = 20
    position_encoding: bool = False
    gap_encoding: str = "raw"
    loop_end_bases: int = 0
    target_len: int = 15
    pos_rbf: int = 16
    pos_sigma: float = 10.0

    def __post_init__(self) -> None:
        checks = {
            "min_run_len must be >= 1": self.min_run_len >= 1,
            "edge_radius must be >= 1": self.edge_radius >= 1,
            "context_bases must be between 0 and 10": 0 <= self.context_bases <= 10,
            "flank_window must be >= 1": self.flank_window >= 1,
            "gap_encoding must be 'raw' or 'bucket'": self.gap_encoding in ("raw", "bucket"),
            "loop_end_bases must be between 0 and 10": 0 <= self.loop_end_bases <= 10,
            "target_len must be >= 1": self.target_len >= 1,
            "pos_rbf must be >= 2": self.pos_rbf >= 2,
            "pos_sigma must be positive": self.pos_sigma > 0,
        }
        failed = [msg for msg, ok in checks.items() if not ok]
        if failed:
            raise ValueError("invalid FeatureConfig: " + "; ".join(failed))

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown config key(s): {', '.join(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)
