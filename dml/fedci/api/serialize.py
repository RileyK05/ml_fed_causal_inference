"""DataFrame/figure to JSON-safe values for the API.

Pandas Timestamps and NaNs are not JSON-serializable; both helpers below go through
pandas' own JSON encoder (ISO dates) and back to plain Python objects.
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go


def records(df: pd.DataFrame) -> list[dict]:
    """JSON-safe records: ISO dates, NaN -> None, numbers stay numbers."""
    return json.loads(df.to_json(orient="records", date_format="iso"))


def figure_json(fig: go.Figure) -> dict:
    """Plotly figure as plain JSON-safe dict."""
    return json.loads(fig.to_json())


__all__ = ["records", "figure_json"]
