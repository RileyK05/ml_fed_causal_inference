"""Headline Q3 metric: does the model rank and size firms' departures from beta? (q3-v6-plan.md)

Daily firm returns carry ~2pt of news no Fed model can see, so MSE barely moves and is kept as a
do-no-harm guardrail only. These metrics look at the departure from the base model instead:

- ``dev`` = model prediction minus base prediction, demeaned within meeting (what the model adds);
- ``miss`` = realized return minus base prediction, demeaned within meeting (what the base got wrong).

Metrics, all on a chosen set of meetings and in basis points per 10bp surprise where they have units:

- ``spread``: per meeting, firms ranked by predicted extra sensitivity (dev / s); realized ``miss``
  of the top decile minus the bottom decile. Pooled over meetings as the through-origin slope of
  that gap on s, so the unit is "bp more return per 10bp surprise for the firms the model flags".
  A meeting where the model makes no ranking (dev all zero) counts as a zero gap.
- ``predicted_spread``: the same with ``dev`` in place of ``miss`` (what the model claimed).
- ``rank_ic``: mean within-meeting Spearman correlation of dev and miss (meetings with a ranking).
- ``calibration``: pooled within-meeting slope of miss on dev (1 = sizes right, 0 = noise).

Intervals resample whole meetings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DECILE = 0.1
BOOT = 5000


def departures(pred: pd.DataFrame) -> pd.DataFrame:
    """``pred`` indexed by (meeting, firm) with columns target, mu, mu_base -> adds dev and miss."""
    g = lambda c: pred[c] - pred.groupby(level=0)[c].transform("mean")
    out = pred.copy()
    out["dev"] = g("mu") - g("mu_base")
    out["miss"] = g("target") - g("mu_base")
    return out


def _meeting_parts(d: pd.DataFrame, s: float, q: float) -> dict:
    dev, miss = d["dev"].to_numpy(float), d["miss"].to_numpy(float)
    ok = np.isfinite(dev) & np.isfinite(miss)
    dev, miss = dev[ok], miss[ok]
    out = dict(s=s, gap=0.0, pred_gap=0.0, ic=np.nan, xy=float(dev @ miss), xx=float(dev @ dev))
    if len(dev) < 10 or np.ptp(dev) < 1e-12:
        return out
    k = max(1, int(round(q * len(dev))))
    order = np.argsort(dev / s, kind="stable")
    lo, hi = order[:k], order[-k:]
    out["gap"] = float(miss[hi].mean() - miss[lo].mean())
    out["pred_gap"] = float(dev[hi].mean() - dev[lo].mean())
    r_dev, r_miss = pd.Series(dev).rank().to_numpy(), pd.Series(miss).rank().to_numpy()
    out["ic"] = float(np.corrcoef(r_dev, r_miss)[0, 1])
    return out


def meeting_table(pred: pd.DataFrame, shock: pd.Series, q: float = DECILE) -> pd.DataFrame:
    """One row per meeting: s, top-minus-bottom realized gap, predicted gap, rank IC, and the
    calibration sums. ``shock`` is indexed by meeting (10bp units); zero-surprise meetings are dropped."""
    d = departures(pred)
    rows = []
    for m, sub in d.groupby(level=0):
        s = float(shock.get(m, np.nan))
        if not np.isfinite(s) or s == 0:
            continue
        rows.append(dict(meeting=m, **_meeting_parts(sub, s, q)))
    return pd.DataFrame(rows).set_index("meeting")


def _stats(t: pd.DataFrame) -> dict:
    ss = float((t["s"] ** 2).sum())
    ic = t["ic"].dropna()
    return dict(spread=100 * float((t["s"] * t["gap"]).sum()) / ss,
                predicted_spread=100 * float((t["s"] * t["pred_gap"]).sum()) / ss,
                rank_ic=float(ic.mean()) if len(ic) else np.nan,
                calibration=float(t["xy"].sum() / t["xx"].sum()) if t["xx"].sum() > 0 else np.nan)


def summarize(table: pd.DataFrame, n_boot: int = BOOT, seed: int = 0) -> dict:
    """Point estimates and meeting-bootstrap 95% intervals for every metric of a meeting table."""
    point = _stats(table)
    rng = np.random.default_rng(seed)
    draws = [_stats(table.iloc[rng.integers(0, len(table), len(table))]) for _ in range(n_boot)]
    out = dict(meetings=len(table))
    for k, v in point.items():
        x = np.array([d[k] for d in draws], dtype=float)
        x = x[np.isfinite(x)]
        lo, hi = (np.quantile(x, 0.025), np.quantile(x, 0.975)) if len(x) else (np.nan, np.nan)
        out[k], out[k + "_lo"], out[k + "_hi"] = v, float(lo), float(hi)
    return out


def _slice(t: pd.DataFrame, min_abs_s: float, exclude) -> pd.DataFrame:
    return t[(t["s"].abs() >= min_abs_s) & ~t.index.isin(pd.DatetimeIndex(list(exclude)))]


def score(pred: pd.DataFrame, shock: pd.Series, *, min_abs_s: float = 0.25, exclude=(), q: float = DECILE,
          n_boot: int = BOOT, seed: int = 0) -> dict:
    """``summarize`` over meetings with |s| >= ``min_abs_s`` (10bp units), minus ``exclude``."""
    return summarize(_slice(meeting_table(pred, shock, q), min_abs_s, exclude), n_boot=n_boot, seed=seed)


def paired(pred_a: pd.DataFrame, pred_b: pd.DataFrame, shock: pd.Series, *, min_abs_s: float = 0.25, exclude=(),
           q: float = DECILE, n_boot: int = BOOT, seed: int = 0) -> dict:
    """Spread and rank IC of arm a minus arm b, the same meetings resampled for both."""
    ta = _slice(meeting_table(pred_a, shock, q), min_abs_s, exclude)
    tb = _slice(meeting_table(pred_b, shock, q), min_abs_s, exclude)
    common = ta.index.intersection(tb.index)
    ta, tb = ta.loc[common], tb.loc[common]

    def stats(i):
        a, b = _stats(ta.iloc[i]), _stats(tb.iloc[i])
        return dict(spread=a["spread"] - b["spread"], rank_ic=a["rank_ic"] - b["rank_ic"])

    point = stats(np.arange(len(common)))
    rng = np.random.default_rng(seed)
    draws = [stats(rng.integers(0, len(common), len(common))) for _ in range(n_boot)]
    out = dict(meetings=len(common))
    for k, v in point.items():
        x = np.array([d[k] for d in draws], dtype=float)
        x = x[np.isfinite(x)]
        out[k], out[k + "_lo"], out[k + "_hi"] = v, float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))
    return out
