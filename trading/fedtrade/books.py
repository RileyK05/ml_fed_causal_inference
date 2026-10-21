"""Stage 2 v1: rule books on the frozen ranking (no fitting). One trade per meeting.

Within a meeting, the *sensitive* leg is the q of firms with the lowest delta_b: by Q3's model
they fall most after hawkish surprises and gain most after dovish ones (on development data,
mostly distressed, levered firms). The *insensitive* leg is the q with the highest delta_b
(mostly cash-rich tech). ``direction = -sign(s)`` is +1 when the sensitive leg should gain.

Entered at the day-0 close, held over the drift window; ``car`` is each firm's post-meeting
abnormal return (``fedcore.q3.drift.car_table``), in % of one leg.

- ``decile``: direction x (sensitive - insensitive), the trade in docs/encoder/q3-v6-drift.md s5.
- ``sensitive``: direction x (sensitive - average firm); the half that carried the edge (s9).
- ``sensitive_sector``: the same against each firm's own GICS sector average (no sector bet).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DECILE = 0.1
STRATEGIES = ("decile", "sensitive", "sensitive_sector")
LEGS = {"decile": 2, "sensitive": 1, "sensitive_sector": 1}   # stock legs traded per trade (for costs)


def legs(delta_b: pd.Series, q: float = DECILE) -> tuple[pd.Index, pd.Index]:
    """(sensitive, insensitive) firm ids of one meeting: lowest and highest q of delta_b."""
    d = delta_b.dropna()
    if len(d) < 10 or np.ptp(d.to_numpy()) < 1e-12:
        return pd.Index([]), pd.Index([])
    k = max(1, int(round(q * len(d))))
    order = d.sort_values(kind="stable").index
    return order[:k], order[-k:]


def trade_returns(delta_b: pd.Series, car: pd.Series, s: float, sector: pd.Series | None = None,
                  q: float = DECILE) -> dict:
    """Each strategy's return for one meeting (NaN when it cannot trade). Series indexed by firm."""
    ok = car.notna() & delta_b.notna()
    car, delta_b = car[ok], delta_b[ok]
    sens, insens = legs(delta_b, q)
    out = dict.fromkeys(STRATEGIES, np.nan)
    if not len(sens) or not np.isfinite(s) or s == 0:
        return out
    direction = -np.sign(s)
    out["decile"] = direction * (car[sens].mean() - car[insens].mean())
    out["sensitive"] = direction * (car[sens].mean() - car.mean())
    if sector is not None:
        sec = sector.reindex(car.index)
        rel = car - car.groupby(sec).transform("mean")
        out["sensitive_sector"] = direction * rel[sens].mean()
    return out
