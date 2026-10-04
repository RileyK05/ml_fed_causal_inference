"""Hand-built history summaries used by the tabular baselines.

Per channel and window (21, 63, 252 days): trailing mean, population volatility,
and peak-to-trough decline of the cumulative sum. Plus the last observed beta63,
the compounded price max-drawdown over the whole window, and 63-day momentum
(compounded return, in percent).

Missing days are skipped. They do not enter means, and a price path holds flat
across them. A window with no observed days is NaN; the baseline imputes that
from training rows.
"""
from __future__ import annotations

import numpy as np

from fedcore.q3.contracts import BETA63, CHANNELS, Q3Panel, RET

WINDOWS = (21, 63, 252)


def summary_feature_names(n_channels: int = len(CHANNELS)) -> list[str]:
    names = [CHANNELS[c] if c < len(CHANNELS) else f"ch{c}" for c in range(n_channels)]
    out: list[str] = []
    for name in names:
        for window in WINDOWS:
            out.append(f"{name}_mean_{window}")
            out.append(f"{name}_vol_{window}")
            out.append(f"{name}_drawdown_{window}")
    out.extend(["last_beta63", "max_drawdown", "momentum_63"])
    return out


def summary_features(panel_or_history, mask: np.ndarray | None = None) -> np.ndarray:
    """Return (N, n_features) float64. Accepts a Q3Panel or a history array plus mask."""
    history, mask = _history_and_mask(panel_or_history, mask)
    n, length, n_channels = history.shape
    if n_channels <= BETA63:
        raise ValueError(f"history needs a beta63 channel at index {BETA63}")
    columns: list[np.ndarray] = []
    for channel in range(n_channels):
        series = history[:, :, channel]
        for window in WINDOWS:
            width = min(window, length)
            sl = series[:, -width:]
            sm = mask[:, -width:]
            mean, vol = masked_mean_std(sl, sm)
            columns.append(mean)
            columns.append(vol)
            columns.append(cumsum_drawdown(sl, sm))
    columns.append(last_observed(history[:, :, BETA63], mask))
    columns.append(price_max_drawdown(history[:, :, RET], mask))
    width = min(63, length)
    columns.append(compounded_percent(history[:, -width:, RET], mask[:, -width:]))
    return np.column_stack(columns).astype(np.float64, copy=False)


def channel_window_vol(history: np.ndarray, mask: np.ndarray, channel: int, window: int) -> np.ndarray:
    """Population std of one channel over the last `window` days. NaN if fewer than 2 observed days."""
    width = min(window, history.shape[1])
    _, vol = masked_mean_std(history[:, -width:, channel], mask[:, -width:])
    return vol


def masked_mean_std(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Population mean and std. `values` is (N, W). Mean is NaN with no observations; std is NaN with fewer than 2."""
    x = np.asarray(values, dtype=np.float64)
    m = np.asarray(mask, dtype=bool) & np.isfinite(x)
    count = m.sum(axis=1)
    clean = np.where(m, x, 0.0)
    total = clean.sum(axis=1)
    mean = np.divide(total, count, out=np.full(x.shape[0], np.nan), where=count > 0)
    square = np.where(m, x * x, 0.0).sum(axis=1)
    var = np.divide(square, count, out=np.full(x.shape[0], np.nan), where=count > 1) - mean * mean
    var = np.where(count > 1, np.clip(var, 0.0, None), np.nan)
    return mean, np.sqrt(var)


def cumsum_drawdown(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Largest decline of the cumulative sum from its running peak, in the channel's own units."""
    x = np.asarray(values, dtype=np.float64)
    m = np.asarray(mask, dtype=bool) & np.isfinite(x)
    path = np.cumsum(np.where(m, x, 0.0), axis=1)
    peak = np.maximum.accumulate(path, axis=1)
    drawdown = (peak - path).max(axis=1)
    drawdown = np.where(m.any(axis=1), drawdown, np.nan)
    return drawdown.astype(np.float64, copy=False)


def compound_price(ret_pct: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Price starting at 1. `ret_pct` is in percent. Missing days hold the price flat. Shape (N, W+1)."""
    ret = np.asarray(ret_pct, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(ret)
    n, width = ret.shape
    price = np.empty((n, width + 1), dtype=np.float64)
    price[:, 0] = 1.0
    for t in range(width):
        step = np.where(valid[:, t], 1.0 + ret[:, t] / 100.0, 1.0)
        price[:, t + 1] = price[:, t] * np.maximum(step, 1e-6)
    return price


def price_max_drawdown(ret_pct: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compounded peak-to-trough drawdown as a fraction (0.15 means 15 percent). NaN if nothing is observed."""
    valid = np.asarray(mask, dtype=bool) & np.isfinite(ret_pct)
    price = compound_price(ret_pct, mask)
    peak = np.maximum.accumulate(price, axis=1)
    drawdown = (peak - price) / np.maximum(peak, 1e-12)
    out = drawdown.max(axis=1)
    return np.where(valid.any(axis=1), out, np.nan)


def compounded_percent(ret_pct: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compounded return in percent. NaN if the window has no observed day."""
    valid = np.asarray(mask, dtype=bool) & np.isfinite(ret_pct)
    price = compound_price(ret_pct, mask)
    out = (price[:, -1] - 1.0) * 100.0
    return np.where(valid.any(axis=1), out, np.nan)


def last_observed(series: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Last finite observed value along axis 1. NaN when the row is entirely masked."""
    x = np.asarray(series, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(x)
    reversed_valid = valid[:, ::-1]
    has = reversed_valid.any(axis=1)
    from_end = np.argmax(reversed_valid, axis=1)
    index = x.shape[1] - 1 - from_end
    out = np.full(x.shape[0], np.nan, dtype=np.float64)
    rows = np.flatnonzero(has)
    out[rows] = x[rows, index[rows]]
    return out


def _history_and_mask(panel_or_history, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(panel_or_history, Q3Panel):
        if mask is not None:
            raise ValueError("pass either a Q3Panel or history and mask, not both")
        return panel_or_history.history, panel_or_history.history_mask
    if mask is None:
        raise TypeError("mask is required when summary_features is called with a history array")
    history = np.asarray(panel_or_history)
    mask = np.asarray(mask, dtype=bool)
    if history.ndim != 3 or mask.shape != history.shape[:2]:
        raise ValueError(f"history {history.shape} and mask {mask.shape} do not align")
    return history, mask
