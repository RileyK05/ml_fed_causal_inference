"""Synthetic firm-meeting panel with a known susceptibility.

The history is an ordinary-day factor model. The event-day target is not a slice
of that path. It is generated from the planted response below, so the label
depends on pre-event state and the meeting shock only.

Nothing in this module reads or writes ``core/data``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fedcore.q3.contracts import (
    BETA63,
    CHANNELS,
    DLOGVOL,
    L,
    Q3Panel,
    RET,
    RET_REL,
    RVOL21,
)
from fedcore.q3.features import channel_window_vol, compound_price

FUNDAMENTAL_NAMES = ("leverage", "log_size", "profitability")
CONTEXT_NAMES = ("rate", "vix", "slope", "credit")

# Trading days between meetings: 30 business days is about six weeks.
MEETING_GAP = 30
LIST_SPAN = 150
DROP_RATE = 0.01
SHOCK_SD = 0.3
PATTERN_WINDOW = 60
DRAWDOWN_THRESHOLD = 0.15
RECOVERY_FRACTION = 1.0 / 3.0
RVOL_WINDOW = 63
RVOL_FALLBACK = 1.5
MIN_BETA_DAYS = 10

# b = B_INTERCEPT + B_LEVERAGE * leverage + B_RVOL * rvol63 + B_PATTERN * pattern
B_INTERCEPT = 0.5
B_LEVERAGE = 2.0
B_RVOL = 0.9
B_PATTERN = 1.5

NOISE_SD = {"easy": 0.2, "realistic": 1.5}


def sequence_pattern(history: np.ndarray, history_mask: np.ndarray) -> np.ndarray:
    """1 when the last 60 days contain a compounded drawdown above 15 percent
    followed by a recovery of at least one third of that peak-to-trough loss.

    Price compounds ``ret`` (percent) on observed days only. A missing day holds
    the price flat. The trough has to occur before the last day, so the recovery
    is in the window rather than a drawdown that is still open. Returns float32 (N,).
    """
    width = min(PATTERN_WINDOW, history.shape[1])
    ret = np.asarray(history[:, -width:, RET], dtype=np.float64)
    mask = np.asarray(history_mask[:, -width:], dtype=bool)
    price = compound_price(ret, mask)
    peak = np.maximum.accumulate(price, axis=1)
    drawdown = (peak - price) / np.maximum(peak, 1e-12)
    trough = np.argmax(drawdown, axis=1)
    rows = np.arange(trough.shape[0])
    max_dd = drawdown[rows, trough]
    loss = peak[rows, trough] - price[rows, trough]
    recovered = np.divide(
        price[:, -1] - price[rows, trough], loss, out=np.zeros(rows.shape[0]), where=loss > 1e-8
    )
    flag = (max_dd > DRAWDOWN_THRESHOLD) & (trough < price.shape[1] - 1) & (recovered >= RECOVERY_FRACTION)
    return flag.astype(np.float32)


def synthetic_components(panel: Q3Panel) -> pd.DataFrame:
    """Planted a and b for a panel from :func:`make_synthetic_panel`, one row per panel row.

    Column 0 of ``fundamentals`` is leverage. Context columns are ``CONTEXT_NAMES``.
    ``rvol63`` is the population std of daily percent returns over the last 63 days,
    or 1.5 when fewer than two days are observed. ``pattern`` is :func:`sequence_pattern`.

        b = 0.5 + 2.0 * leverage + 0.9 * rvol63 + 1.5 * pattern
        a = 0.30 * (leverage - 0.45) + 0.04 * (rate - 2) + 0.03 * (vix - 18)
    """
    pattern = sequence_pattern(panel.history, panel.history_mask).astype(np.float64)
    rvol = channel_window_vol(panel.history, panel.history_mask, RET, RVOL_WINDOW).astype(np.float64)
    rvol = np.where(np.isfinite(rvol), rvol, RVOL_FALLBACK)
    leverage = np.asarray(panel.fundamentals[:, 0], dtype=np.float64)
    rate = np.asarray(panel.context[:, 0], dtype=np.float64)
    vix = np.asarray(panel.context[:, 1], dtype=np.float64)
    b_fund = B_LEVERAGE * leverage
    b_level = B_RVOL * rvol
    b_seq = B_PATTERN * pattern
    b = B_INTERCEPT + b_fund + b_level + b_seq
    a = 0.30 * (leverage - 0.45) + 0.04 * (rate - 2.0) + 0.03 * (vix - 18.0)
    return pd.DataFrame(
        {
            "leverage": leverage,
            "rvol63": rvol,
            "pattern": pattern,
            "b_fund": b_fund,
            "b_level": b_level,
            "b_seq": b_seq,
            "a_true": a,
            "b_true": b,
        }
    )


def make_synthetic_panel(
    n_firms: int = 200,
    n_meetings: int = 160,
    noise: str = "realistic",
    seed: int = 0,
) -> Q3Panel:
    """Firm-meeting panel whose event-day return is ``a + b * s + eps``.

    ``s`` is the meeting shock in units of 10bp, drawn N(0, 0.3): a typical
    surprise is about 3bp. ``eps`` has sd 0.2 when ``noise`` is ``"easy"`` and
    sd 1.5 when ``noise`` is ``"realistic"`` (close to a daily equity return).

    The same seed produces the same history, shock, fundamentals and context
    under both noise settings. Only the target noise changes.

    Channels, in order, on each observed day:

    - ``ret``: daily simple return in percent. Market factor, plus a fixed firm
      beta, plus idiosyncratic noise with a two-regime volatility and occasional
      high-vol jumps.
    - ``dlogvol``: change in log volume since the previous observed day. Volume
      rises with absolute return. The first observed day is 0.
    - ``rvol21``: population std of ``ret`` over the last 21 trading days
      (1.5 if fewer than two days are observed).
    - ``ret_rel``: ``ret`` minus the market return that day.
    - ``beta63``: population regression of ``ret`` on the market over the last
      63 trading days. Fewer than 10 observed days, or a flat market, falls
      back to the firm's true beta so the observed day stays finite.

    About 1 percent of listed firm-days are dropped. Firms also draw a listing
    date in the first 150 trading days, so early meetings are left-padded.
    Masked steps are NaN in every channel. History ends at the close before
    the meeting. The target is the planted event return, not the ordinary
    simulated return on the meeting day.

    ``b`` depends on leverage (in the fundamentals), trailing volatility (which
    summary features see) and :func:`sequence_pattern` (a path event that
    summary features only partly capture through drawdown and momentum).
    """
    if n_firms < 1 or n_meetings < 1:
        raise ValueError("n_firms and n_meetings must be positive")
    if noise not in NOISE_SD:
        raise ValueError(f"noise must be one of {tuple(NOISE_SD)}, got {noise!r}")

    rng = np.random.default_rng(seed)
    meeting_pos = L + np.arange(n_meetings, dtype=int) * MEETING_GAP
    n_days = int(meeting_pos[-1]) + 1
    sim = _simulate(n_firms, n_days, rng)

    hist = np.empty((n_meetings, n_firms, L, len(CHANNELS)), dtype=np.float32)
    hmask = np.empty((n_meetings, n_firms, L), dtype=bool)
    for j, day in enumerate(meeting_pos):
        hist[j] = sim["channels"][:, day - L : day, :]
        hmask[j] = sim["observed"][:, day - L : day]
    n = n_meetings * n_firms
    history = hist.reshape(n, L, len(CHANNELS))
    history_mask = hmask.reshape(n, L)

    calendar = pd.bdate_range("1998-01-05", periods=n_days)
    meeting = np.repeat(calendar[meeting_pos].to_numpy(dtype="datetime64[ns]"), n_firms)
    firm_id = np.tile(np.arange(n_firms, dtype=np.int64), n_meetings)

    fund = np.tile(sim["fundamentals"], (n_meetings, 1)).astype(np.float32)
    fund_mask = np.ones(fund.shape, dtype=bool)

    vix = np.array(
        [float(np.std(sim["market"][day - 21 : day], ddof=0) * np.sqrt(252.0)) for day in meeting_pos],
        dtype=np.float64,
    )
    rate = np.clip(2.0 + np.cumsum(rng.normal(0.0, 0.10, size=n_meetings)), 0.0, 8.0)
    slope_eps = rng.normal(0.0, 0.05, size=n_meetings)
    credit_eps = rng.normal(0.0, 0.05, size=n_meetings)
    slope = np.empty(n_meetings, dtype=np.float64)
    credit = np.empty(n_meetings, dtype=np.float64)
    slope[0] = 1.0 + slope_eps[0]
    credit[0] = 1.5 + credit_eps[0]
    for j in range(1, n_meetings):
        slope[j] = 0.85 * slope[j - 1] + 0.15 * 1.0 + slope_eps[j]
        credit[j] = 0.90 * credit[j - 1] + 0.10 * 1.5 + 0.04 * (vix[j] - 18.0) + credit_eps[j]
    context_m = np.column_stack((rate, vix, slope, credit)).astype(np.float32)
    context = np.repeat(context_m, n_firms, axis=0)
    shock = np.repeat(rng.normal(0.0, SHOCK_SD, size=n_meetings), n_firms).astype(np.float32)

    # Placeholder target so the component helper can read history and covariates.
    # The planted values are then written from that same helper.
    draft = Q3Panel(
        meeting=meeting,
        firm_id=firm_id,
        history=history,
        history_mask=history_mask,
        fundamentals=fund,
        fundamentals_mask=fund_mask,
        context=context,
        shock=shock,
        target=np.zeros(n, dtype=np.float32),
    )
    parts = synthetic_components(draft)
    eps = rng.normal(0.0, 1.0, size=n) * NOISE_SD[noise]
    target = (parts["a_true"].to_numpy() + parts["b_true"].to_numpy() * shock.astype(np.float64) + eps).astype(np.float32)
    panel = Q3Panel(
        meeting=meeting,
        firm_id=firm_id,
        history=history,
        history_mask=history_mask,
        fundamentals=fund,
        fundamentals_mask=fund_mask,
        context=context,
        shock=shock,
        target=target,
        a_true=parts["a_true"].to_numpy(dtype=np.float32),
        b_true=parts["b_true"].to_numpy(dtype=np.float32),
    )
    panel.validate()
    return panel


def make_pretrain_windows(
    n_windows: int,
    seed: int = 0,
    *,
    n_firms: int = 48,
    n_days: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Ordinary-day windows from the same generator: ``(history, mask)``.

    No meeting and no target. Windows are sampled with replacement from one
    simulated universe (``n_firms`` paths). Shape is ``(n_windows, 252, C)``
    and ``(n_windows, 252)``.
    """
    if n_windows < 1:
        raise ValueError("n_windows must be positive")
    rng = np.random.default_rng(seed)
    n_days = int(n_days or (L + 600))
    if n_days <= L:
        raise ValueError(f"n_days must exceed {L}")
    sim = _simulate(n_firms, n_days, rng)
    ends = rng.integers(L, n_days, size=n_windows)
    firms = rng.integers(0, n_firms, size=n_windows)
    history = np.empty((n_windows, L, len(CHANNELS)), dtype=np.float32)
    mask = np.empty((n_windows, L), dtype=bool)
    channels = sim["channels"]
    observed = sim["observed"]
    for i, (firm, end) in enumerate(zip(firms, ends)):
        history[i] = channels[firm, end - L : end]
        mask[i] = observed[firm, end - L : end]
    return history, mask


def _simulate(n_firms: int, n_days: int, rng: np.random.Generator) -> dict:
    """One universe. Random draws are in a fixed order so a seed is the whole panel."""
    log_sig = np.zeros(n_days, dtype=np.float64)
    innov = rng.normal(0.0, 0.08, size=n_days)
    for t in range(1, n_days):
        log_sig[t] = 0.97 * log_sig[t - 1] + innov[t]
    market_sig = 0.85 * np.exp(np.clip(log_sig, -1.0, 1.0))
    market = rng.normal(0.0, 1.0, size=n_days) * market_sig

    beta_true = rng.uniform(0.5, 1.5, size=n_firms)
    state = np.zeros(n_firms, dtype=np.int8)
    idio = np.empty((n_firms, n_days), dtype=np.float64)
    stay = np.array([0.985, 0.97])
    regime_sig = np.array([0.85, 2.2])
    for t in range(n_days):
        state = np.where(rng.random(n_firms) > stay[state], 1 - state, state).astype(np.int8)
        shock = rng.normal(0.0, 1.0, size=n_firms) * regime_sig[state]
        jump = (state == 1) & (rng.random(n_firms) < 0.05)
        idio[:, t] = np.where(jump, shock - 7.0, shock)
    ret = beta_true[:, None] * market[None, :] + idio

    log_vol = 11.0 + 0.30 * np.abs(ret) + rng.normal(0.0, 0.12, size=ret.shape)
    list_day = rng.integers(0, LIST_SPAN + 1, size=n_firms)
    day = np.arange(n_days)
    observed = (day[None, :] >= list_day[:, None]) & (rng.random((n_firms, n_days)) >= DROP_RATE)

    rvol = _rolling_std(ret, observed, 21, RVOL_FALLBACK)
    beta = _rolling_beta(ret, market, observed, RVOL_WINDOW, beta_true)
    dlog = _dlog_volume(log_vol, observed)
    rel = ret - market[None, :]

    channels = np.empty((n_firms, n_days, len(CHANNELS)), dtype=np.float64)
    channels[:, :, RET] = ret
    channels[:, :, DLOGVOL] = dlog
    channels[:, :, RVOL21] = rvol
    channels[:, :, RET_REL] = rel
    channels[:, :, BETA63] = beta
    channels[~observed] = np.nan
    channels = channels.astype(np.float32)
    if not np.isfinite(channels[observed]).all():
        raise RuntimeError("an observed day produced a non-finite channel")

    fundamentals = np.column_stack(
        (
            rng.uniform(0.05, 0.85, size=n_firms),
            rng.normal(10.0, 0.8, size=n_firms),
            rng.normal(0.08, 0.03, size=n_firms),
        )
    ).astype(np.float64)
    return {
        "channels": channels,
        "observed": observed,
        "market": market,
        "fundamentals": fundamentals,
        "beta_true": beta_true,
    }


def _window_sum(x: np.ndarray, width: int) -> np.ndarray:
    """Sum over [t-width+1, t] inclusive. The left edge uses the days that exist."""
    c = np.cumsum(x, axis=1)
    c_pad = np.concatenate((np.zeros((x.shape[0], 1), dtype=c.dtype), c), axis=1)
    left_idx = np.clip(np.arange(x.shape[1]) + 1 - width, 0, None)
    return c_pad[:, 1:] - c_pad[:, left_idx]


def _rolling_std(ret: np.ndarray, observed: np.ndarray, width: int, fallback: float) -> np.ndarray:
    obs = observed.astype(np.float64)
    x = np.where(observed, ret, 0.0)
    count = _window_sum(obs, width)
    total = _window_sum(x, width)
    square = _window_sum(x * x, width)
    mean = np.divide(total, count, out=np.zeros_like(count), where=count > 0)
    var = np.divide(square, count, out=np.zeros_like(count), where=count > 1) - mean * mean
    std = np.sqrt(np.clip(var, 0.0, None))
    return np.where(count >= 2, std, fallback)


def _rolling_beta(
    ret: np.ndarray, market: np.ndarray, observed: np.ndarray, width: int, beta_true: np.ndarray
) -> np.ndarray:
    obs = observed.astype(np.float64)
    count = _window_sum(obs, width)
    r = np.where(observed, ret, 0.0)
    m = np.where(observed, market[None, :], 0.0)
    rm = np.where(observed, ret * market[None, :], 0.0)
    m2 = np.where(observed, np.broadcast_to(market[None, :] ** 2, ret.shape), 0.0)
    sum_r = _window_sum(r, width)
    sum_m = _window_sum(m, width)
    sum_rm = _window_sum(rm, width)
    sum_m2 = _window_sum(m2, width)
    mean_r = np.divide(sum_r, count, out=np.zeros_like(count), where=count > 0)
    mean_m = np.divide(sum_m, count, out=np.zeros_like(count), where=count > 0)
    cov = np.divide(sum_rm, count, out=np.zeros_like(count), where=count > 0) - mean_r * mean_m
    var_m = np.divide(sum_m2, count, out=np.zeros_like(count), where=count > 0) - mean_m * mean_m
    ok = (count >= MIN_BETA_DAYS) & (var_m > 1e-8)
    beta = np.divide(cov, var_m, out=np.zeros_like(cov), where=var_m > 1e-8)
    return np.where(ok, beta, beta_true[:, None])


def _dlog_volume(log_vol: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Change in log volume since the previous observed day. The first observed day is 0."""
    held = np.where(observed, log_vol, np.nan)
    prev = np.full_like(held, np.nan)
    prev[:, 1:] = held[:, :-1]
    finite = np.isfinite(prev)
    index = np.where(finite, np.arange(prev.shape[1])[None, :], 0)
    np.maximum.accumulate(index, axis=1, out=index)
    filled = prev[np.arange(prev.shape[0])[:, None], index]
    change = log_vol - filled
    return np.where(np.isfinite(change), change, 0.0)


