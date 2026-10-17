"""Configuration for the fedviz package."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class VizConfig:
    """Paths and settings for the visualization suite."""

    data_dir: Path = REPO_ROOT / "core" / "data"
    catalog_file: Path = REPO_ROOT / "core" / "data" / "catalog.yaml"
    results_dirs: list[Path] = field(default_factory=lambda: [
        REPO_ROOT / "dml" / "results",
        REPO_ROOT / "encoder" / "results",
        REPO_ROOT / "cde" / "results",
    ])
    cache_dir: Path = REPO_ROOT / "visualization_package" / ".cache"
    default_theme: str = "dark"
    max_rows_display: int = 10_000
    default_lookback_days: int = 252 * 5

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
