"""Dataset registry, read from data/catalog.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from fedci.config import CATALOG_FILE, DATA

LAYERS = ("raw", "interim", "processed")
GRAINS = ("daily", "monthly", "event")


@dataclass(frozen=True)
class Dataset:
    name: str
    path: str
    layer: str
    source: str
    grain: str
    date_column: str
    key: tuple[str, ...]
    questions: tuple[str, ...]
    description: str
    na_values: tuple[str, ...] = field(default=())

    @property
    def file(self) -> Path:
        return DATA / self.path

    @property
    def exists(self) -> bool:
        return self.file.exists()


@lru_cache(maxsize=1)
def catalog() -> dict[str, Dataset]:
    """All registered datasets, keyed by name."""
    spec = yaml.safe_load(CATALOG_FILE.read_text(encoding="utf-8"))["datasets"]
    out = {}
    for name, d in spec.items():
        ds = Dataset(
            name=name,
            path=d["path"],
            layer=d["layer"],
            source=d["source"],
            grain=d["grain"],
            date_column=d["date_column"],
            key=tuple(d["key"]),
            questions=tuple(d.get("questions", [])),
            description=d.get("description", ""),
            na_values=tuple(d.get("na_values", [])),
        )
        if ds.layer not in LAYERS:
            raise ValueError(f"{name}: layer must be one of {LAYERS}, got {ds.layer!r}")
        if ds.grain not in GRAINS:
            raise ValueError(f"{name}: grain must be one of {GRAINS}, got {ds.grain!r}")
        out[name] = ds
    return out


def get(name: str) -> Dataset:
    cat = catalog()
    if name not in cat:
        raise KeyError(f"Unknown dataset {name!r}. Registered: {', '.join(sorted(cat))}")
    return cat[name]


def for_question(qid: str) -> list[Dataset]:
    return [d for d in catalog().values() if qid in d.questions]
