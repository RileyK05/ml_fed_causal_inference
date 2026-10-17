"""fedviz — Generalized visualization suite for FOMC causal inference models.

A plugin-based visualization framework that any market model can plug into.
Provides interactive Streamlit pages for market overview, event studies,
cross-sectional analysis, model diagnostics, results browsing, and comparison.
"""

from fedviz.config import VizConfig
from fedviz.registry import VizRegistry, get_registry
from fedviz.data import VizData
from fedviz.themes import Theme

__version__ = "0.1.0"

__all__ = [
    "VizConfig",
    "VizRegistry",
    "get_registry",
    "VizData",
    "Theme",
]
