"""Post-meeting drift: do a model's beyond-beta calls keep paying off in the days after the meeting?

The day-0 spread (``fedcore.q3.beyond_beta``) asks whether the model ranks firms' announcement-day
moves beyond beta. This module asks whether the same ranking predicts what happens next:

- ``CAR_k``: the sum of daily abnormal returns (return minus beta63-as-of-the-prior-day times the
  market, ``channels.abnormal_returns``) on trading days 1..k after the meeting. NaN when under
  80% of those days are observed, or when the window reaches past ``end``. This keeps the final
  test's returns out of development work.
- Drift spread: ``beyond_beta`` with ``target = CAR_k`` and ``mu_base = 0``. Firms are ranked by
  the model's predicted extra sensitivity, and the result is in bp of k-day drift per 10bp surprise.
- Sign-flip placebo: deciles are ranked by delta_b, which does not depend on the surprise's sign.
  A drift that follows the Fed's direction should beat random sign flips of s (|s| kept). A
  static characteristic drift (value, size, January) would not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fedcore.q3.beyond_beta import meeting_table

MIN_OBSERVED = 0.8


def car_table(rows: pd.DataFrame, abn: np.ndarray, calendar: pd.DatetimeIndex, permnos: np.ndarray,
              k: int, end) -> pd.Series:
    """CAR over days 1..k after each row's meeting, indexed (meeting, firm_id).

    ``rows``: meeting, permno. ``abn``: (firm, day) abnormal returns on ``calendar``."""
    pos = calendar.get_indexer(pd.DatetimeIndex(rows["meeting"]))
    firm = np.searchsorted(permnos, rows["permno"].to_numpy(dtype=np.int64))
    last = calendar.searchsorted(pd.Timestamp(end), side="right") - 1
    out = np.full(len(rows), np.nan)
    ok = (pos >= 0) & (pos + k <= last)
    if ok.any():
        win = np.stack([abn[firm[ok], pos[ok] + d] for d in range(1, k + 1)], axis=1)
        v = np.nansum(win, axis=1)
        v[np.isfinite(win).sum(1) < max(1, int(MIN_OBSERVED * k))] = np.nan
        out[ok] = v
    idx = pd.MultiIndex.from_arrays([pd.DatetimeIndex(rows["meeting"]), rows["permno"].astype(np.int64)],
                                    names=["meeting", "firm_id"])
    return pd.Series(out, index=idx)


def drift_frame(pred: pd.DataFrame, car: pd.Series) -> pd.DataFrame:
    """``beyond_beta`` input whose target is the drift: dev = mu - mu_base, miss = CAR."""
    return pd.DataFrame(dict(target=car.reindex(pred.index), mu_base=0.0, mu=pred["mu"] - pred["mu_base"])).dropna()


def _spread(s: np.ndarray, gap: np.ndarray) -> float:
    return 100.0 * float((s * gap).sum() / (s ** 2).sum())


def sign_flip_p(table: pd.DataFrame, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """(share of sign-flipped spreads >= the real one, the real spread). ``table`` from ``meeting_table``."""
    s, gap = table["s"].to_numpy(dtype=float), table["gap"].to_numpy(dtype=float)
    real = _spread(s, gap)
    rng = np.random.default_rng(seed)
    flips = np.array([_spread(np.abs(s) * rng.choice([-1.0, 1.0], len(s)), gap) for _ in range(n)])
    return float(np.mean(flips >= real)), real


def drift_table(pred: pd.DataFrame, car: pd.Series, shock: pd.Series, min_abs_s: float = 0.25, exclude=()) -> pd.DataFrame:
    t = meeting_table(drift_frame(pred, car), shock)
    return t[(t["s"].abs() >= min_abs_s) & ~t.index.isin(pd.DatetimeIndex(list(exclude)))]
