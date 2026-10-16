"""Bounded Q3 panels and their variants: sector inputs, market-adjusted target, surprise measure.

Nothing here fits a model, and nothing uses the 2026 holdout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GICS_SECTORS = ("10", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60")


def with_sector(panel, rows):
    """Append GICS sector one-hot columns to the fundamentals (observed when the sector is known).

    One-hot columns have median 0 and IQR 0, so the trainer's scaler leaves them as 0/1."""
    import dataclasses

    code = rows["gsector"].astype("string").str[:2].fillna("").to_numpy(dtype=object)
    onehot = np.stack([code == s for s in GICS_SECTORS], axis=1).astype(np.float32)
    known = np.repeat(np.isin(code, GICS_SECTORS)[:, None], len(GICS_SECTORS), axis=1)
    onehot[~known] = np.nan  # the panel contract: unobserved entries are NaN
    out = dataclasses.replace(panel, fundamentals=np.concatenate([panel.fundamentals, onehot], axis=1),
                              fundamentals_mask=np.concatenate([panel.fundamentals_mask, known], axis=1))
    out.validate()
    return out


def market_adjusted(panel, market=None):
    """Replace the target with the event-study abnormal return: return - beta x market return that day.

    Beta is the firm's last observed 63-day beta before the meeting (1.0 when it has none), so it is
    known in advance; the market return (Fama-French ``mktrf + rf``, percent) is the meeting day's.
    High beta alone then earns nothing: a and b must explain what is left after beta x market.
    """
    import dataclasses

    from fedcore.q3.contracts import BETA63
    from fedcore.q3.real import BETA_FALLBACK

    if market is None:
        from fedcore.data import load

        ff = load("ff_factors_daily", columns=["mktrf", "rf"])
        market = pd.Series(100.0 * (ff["mktrf"] + ff["rf"]).to_numpy(dtype=np.float64),
                           index=pd.to_datetime(ff["date"]))
    day = market.reindex(pd.DatetimeIndex(panel.meeting)).to_numpy(dtype=np.float64)
    if np.isnan(day).any():
        raise ValueError("some meetings have no market return")
    b = panel.history[:, :, BETA63].astype(np.float64)
    ok = np.isfinite(b)
    last = b.shape[1] - 1 - np.argmax(ok[:, ::-1], axis=1)
    beta = np.where(ok.any(axis=1), b[np.arange(len(b)), last], BETA_FALLBACK)
    out = dataclasses.replace(panel, target=(panel.target - beta * day).astype(np.float32))
    out.validate()
    return out


SHOCKS = ("STMT", "ME")


def with_shock(panel, measure, surprises=None):
    """Replace the shock with ``10 x`` another USMPD surprise measure.

    ``STMT`` is the statement window (the panel's default). ``ME`` is the whole monetary event,
    statement plus press conference; from 2011, when press conferences began, it also carries
    the news released at the press conference.
    """
    import dataclasses

    if measure not in SHOCKS:
        raise ValueError(f"shock must be one of {SHOCKS}")
    if surprises is None:
        from fedcore.data import load

        mps = load("mps_surprises")
        surprises = pd.Series(mps[measure].to_numpy(dtype=np.float64), index=pd.to_datetime(mps["Date"]))
    s = 10.0 * surprises.groupby(level=0).first().reindex(pd.DatetimeIndex(panel.meeting)).to_numpy()
    if np.isnan(s).any():
        raise ValueError(f"some meetings have no {measure} surprise")
    out = dataclasses.replace(panel, shock=s.astype(np.float32))
    out.validate()
    return out


TARGETS = ("raw", "market_adjusted")


def bounded_panel(through, universe="sp500", with_rows=False, sector=False, target="raw", shock="STMT"):
    """The real panel through an inclusive historical date; never the 2026 holdout.

    ``with_rows`` also returns the matching row metadata (meeting, permno, in_sp500, ...).
    ``sector`` appends GICS sector one-hot columns to the fundamentals (see ``with_sector``).
    ``target="market_adjusted"`` swaps in the abnormal return (see ``market_adjusted``).
    ``shock="ME"`` uses the whole monetary event surprise (see ``with_shock``)."""
    from fedcore.q3.real import load_real_panel

    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    through = pd.Timestamp(through)
    if pd.isna(through) or through >= pd.Timestamp("2026-01-01"):
        raise ValueError("through must be a historical date before the separate 2026 holdout")
    panel, rows, _ = load_real_panel(universe=universe)
    if sector:
        panel = with_sector(panel, rows)
    keep = np.flatnonzero(panel.meeting <= through.to_datetime64())
    panel = panel.subset(keep)
    if shock != "STMT":
        panel = with_shock(panel, shock)
    if target == "market_adjusted":
        panel = market_adjusted(panel)
    if with_rows:
        return panel, through, rows.iloc[keep].reset_index(drop=True)
    return panel, through
