"""Stage 1 (frozen): Q3's saved out-of-sample predictions as a per-meeting rate-sensitivity ranking.

Q3's b-only models predict mu = mu_base + delta_b x s, so delta_b = (mu - mu_base) / s is the
model's extra sensitivity to the surprise beyond beta. It uses only pre-meeting data. The trade
direction needs the surprise's sign, which is known at the day-0 close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fedtrade.config import Q3_RUNS


def load_q3(prefix: str, arm: str, runs=Q3_RUNS) -> tuple[pd.DataFrame, str]:
    """Member-averaged predictions of the newest run named ``<prefix>-<arm>``, indexed (meeting, firm_id)."""
    hits = sorted(runs.glob(f"*_{prefix}-{arm.lower()}"))
    if not hits:
        raise FileNotFoundError(f"no Q3 run *_{prefix}-{arm.lower()} under {runs}")
    p = pd.read_csv(hits[-1] / "tables" / "predictions.csv", parse_dates=["meeting"])
    g = p.groupby(["meeting", "firm_id"])
    return g[["mu", "mu_base"]].mean().join(g["target"].first()), hits[-1].name


def rate_sensitivity(pred: pd.DataFrame, shock: pd.Series) -> pd.Series:
    """delta_b per (meeting, firm); NaN where the meeting's surprise is zero or missing."""
    s = shock.reindex(pred.index.get_level_values(0)).to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = (pred["mu"] - pred["mu_base"]).to_numpy(dtype=float) / s
    return pd.Series(np.where(np.isfinite(db), db, np.nan), index=pred.index, name="delta_b")
