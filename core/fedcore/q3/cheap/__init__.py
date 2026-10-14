"""Cheap tests for Q3 pre-training decisions (docs/encoder/cheap-test-menu.md).

A1-A7 describe the data, C1-C5 and R1-R11 are linear fits on the real panel inside the
development period (R1 also reads the outer design), S1-S5 are synthetic stress tests.
The gradient checks B1-B4 need the transformer and live in ``fedenc.cheap_gradients``.
Run everything through ``python -m fedenc.cheap``; nothing here runs on import.
"""
from fedcore.q3.cheap import audits, fitted, premise, synth  # noqa: F401  (register the checks)
from fedcore.q3.cheap.base import KINDS, REGISTRY, Check, Result, check
from fedcore.q3.cheap.data import CheapData, Config, load_real, synthetic_data, window_ols

__all__ = [
    "KINDS", "REGISTRY", "Check", "Result", "check", "CheapData", "Config", "load_real",
    "synthetic_data", "window_ols",
]
