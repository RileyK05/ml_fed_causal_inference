"""Check registry for the cheap tests. Each check is ``fn(data, cfg) -> Result``."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

# kind -> needs the user's go-ahead before running (docs/encoder/cheap-test-menu.md)
KINDS = {
    "descriptive": False,   # A: data checks, nothing fitted
    "gradient": False,      # B: forward/backward passes, no optimizer step
    "fitted": True,         # C, R: linear fits on the real panel
    "synthetic": True,      # S: linear fits on simulated data
}


@dataclass
class Result:
    metrics: dict = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    notes: str = ""
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Check:
    id: str
    title: str
    kind: str
    fn: Callable
    uses_real: bool = True   # False: the check builds its own synthetic data

    @property
    def needs_approval(self) -> bool:
        return KINDS[self.kind]


REGISTRY: dict[str, Check] = {}


def check(id: str, title: str, kind: str, uses_real: bool = True):
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}")

    def wrap(fn):
        if id in REGISTRY:
            raise ValueError(f"duplicate check id {id}")
        REGISTRY[id] = Check(id, title, kind, fn, uses_real)
        return fn

    return wrap
