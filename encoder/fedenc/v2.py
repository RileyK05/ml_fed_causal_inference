"""Modular Q3 v2 susceptibility experiments: CAPM residual base, macro history, regime FiLM,
two-shock splits (policy/info and target/slope), b-only mode, and shock-weighted training.

Every mechanism can be toggled independently via ``V2Config`` or CLI flags:

* ``base`` (``"capm"`` | ``"pooled"``): anchor on ``CAPMPooled`` (``c + beta*(d + k*s)``) and
  train the networks on the residual ``r - mu_base`` so the encoder cannot merely re-learn beta.
* ``history_macro`` (bool): append daily 1y yield change (``dy1``), ``HML`` and ``SMB`` to the
  252-day sequence tensor, plus trailing 63-day ``rate_beta`` and ``hml_beta`` to fundamentals.
* ``regime`` (bool): append pre-meeting regime state (63d stock-bond correlation, 252d market
  drawdown, risk-free rate level and 63d change) to context and apply FiLM modulation to the b head.
* ``shock_split`` (``"none"`` | ``"policy_info"`` | ``"target_slope"``): split the surprise into
  policy vs. Fed-information shocks or the surprise plus the yield-curve slope change, with a
  separate ``b`` head per shock (``fedcore.q3.shocks``). For ``target_slope`` the slope response
  goes into ``a``, so the returned ``b`` stays the sensitivity to s and ``mu = a + b*s`` holds.
* ``shock_weight`` (float, default 0.0): weight each meeting in training and early-stopping loss
  by ``min(1 + alpha * s^2 / mean_train(s^2), w_max)`` to focus on large-surprise meetings.
* ``b_only`` (bool): set network ``a`` deviations to zero so 100% of network training and early
  stopping is driven by shock sensitivity ``b``.

Every arm is scored against ``capm_two_shock`` (beta-scaled policy and information loadings), the
v2 reference, as well as ``capm_pooled`` and ``pooled``.

CLI usage (training requires user go-ahead in chat):

    python -m fedenc.v2 ablation --through 2021-12-31 --seeds 0 1 2 --device cuda
    python -m fedenc.v2 single --name v2_full --base capm --history-macro --regime \
        --shock-split policy_info --shock-weight 2.0 --b-only --device cuda
"""
from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from fedcore.q3.baselines import BASELINES, CAPMPooled, PooledOLS
from fedcore.q3.contracts import Q3Panel, RET_REL
from fedcore.q3.shocks import info_flags, meeting_table, split_shock
from fedcore.q3.model import _MLP
from fedcore.q3.train import (
    ChannelScaler,
    _meeting_index_groups,
    _meeting_rows,
    resolve_device,
    seed_everything,
    validation_meetings,
)
from fedenc.decouple import LAMBDA_CROSS_FIRM, _chunks, _segment_mean, _segments
from fedenc.encoder_transformer import PatchTransformerEncoder

SHOCK_SPLITS = ("none", "policy_info", "target_slope")
V2_REFERENCES = ("capm_two_shock", "capm_pooled", "pooled")
REFERENCE_MAKERS = {name: BASELINES[name] for name in (*V2_REFERENCES, "capm_target_slope")}
BASES = ("capm", "pooled")
W_MAX_DEFAULT = 5.0


@dataclass(frozen=True)
class V2Config:
    """Modular switches for Q3 v2 experiments."""

    base: str = "capm"
    history_macro: bool = False
    regime: bool = False
    shock_split: str = "none"
    shock_weight: float = 0.0
    w_max: float = W_MAX_DEFAULT
    b_only: bool = False
    lambda_mean: float = LAMBDA_CROSS_FIRM
    members: int = 3
    max_epochs: int = 40
    patience: int = 10

    def __post_init__(self) -> None:
        if self.base not in BASES:
            raise ValueError(f"base must be one of {BASES}, got {self.base!r}")
        if self.shock_split not in SHOCK_SPLITS:
            raise ValueError(f"shock_split must be one of {SHOCK_SPLITS}, got {self.shock_split!r}")
        if self.shock_weight < 0.0:
            raise ValueError("shock_weight must be non-negative")
        if self.w_max < 1.0:
            raise ValueError("w_max must be >= 1.0")


@dataclass(frozen=True)
class MacroTables:
    """Pre-loaded daily macro series and meeting-level shock decompositions.

    Can be passed explicitly in unit tests to avoid touching WRDS/disk data.
    """

    calendar: pd.DatetimeIndex
    dy1: np.ndarray          # (T_cal,) daily 1y yield change
    hml: np.ndarray          # (T_cal,) daily HML return (%)
    smb: np.ndarray          # (T_cal,) daily SMB return (%)
    sb_corr63: np.ndarray    # (T_cal,) 63d rolling corr(mkt, dy1)
    mkt_dd252: np.ndarray    # (T_cal,) 252d market drawdown (%)
    rf_level: np.ndarray     # (T_cal,) annualized risk-free rate (%)
    rf_chg63: np.ndarray     # (T_cal,) 63d change in annualized rf
    meetings: pd.DataFrame   # fedcore.q3.shocks.meeting_table(): sp500, slope by meeting


_CACHED_TABLES: MacroTables | None = None


def load_macro_tables() -> MacroTables:
    """Load and cache daily macro series and USMPD statement columns from the catalog."""
    global _CACHED_TABLES
    if _CACHED_TABLES is not None:
        return _CACHED_TABLES

    from fedcore.data import load

    ff = load("ff_factors_daily").assign(date=lambda d: pd.to_datetime(d["date"])).set_index("date").sort_index()
    y1 = (
        load("treasury_1y")
        .rename(columns={"Date": "date", "SVENY01": "y1"})
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .set_index("date")
        .sort_index()
    )
    cal = pd.DatetimeIndex(ff.index)
    dy1_s = y1["y1"].diff().reindex(cal)
    mkt_ret = 100.0 * (ff["mktrf"] + ff["rf"])
    hml_ret = 100.0 * ff["hml"]
    smb_ret = 100.0 * ff["smb"]
    rf_ann = 252.0 * 100.0 * ff["rf"]

    sb_corr63 = mkt_ret.rolling(63, min_periods=15).corr(dy1_s).fillna(0.0)
    cum_mkt = (1.0 + mkt_ret / 100.0).cumprod()
    mkt_dd252 = ((cum_mkt / cum_mkt.rolling(252, min_periods=21).max() - 1.0) * 100.0).fillna(0.0)
    rf_chg63 = rf_ann.diff(63).fillna(0.0)

    _CACHED_TABLES = MacroTables(
        calendar=cal,
        dy1=dy1_s.fillna(0.0).to_numpy(dtype=np.float32),
        hml=hml_ret.fillna(0.0).to_numpy(dtype=np.float32),
        smb=smb_ret.fillna(0.0).to_numpy(dtype=np.float32),
        sb_corr63=sb_corr63.to_numpy(dtype=np.float32),
        mkt_dd252=mkt_dd252.to_numpy(dtype=np.float32),
        rf_level=rf_ann.fillna(0.0).to_numpy(dtype=np.float32),
        rf_chg63=rf_chg63.to_numpy(dtype=np.float32),
        meetings=meeting_table(),
    )
    return _CACHED_TABLES


def make_synthetic_macro_tables(panel: Q3Panel, seed: int = 0) -> MacroTables:
    """Deterministic macro tables covering ``panel.meetings()`` for unit tests."""
    rng = np.random.default_rng(seed)
    meetings = panel.meetings()
    start = meetings.min() - pd.Timedelta(days=450)
    cal = pd.bdate_range(start, meetings.max() + pd.Timedelta(days=5))
    t = len(cal)
    sp500 = rng.normal(0.0, 0.5, size=len(meetings))
    sp500[0] = np.nan  # one meeting without an S&P window move
    m_df = pd.DataFrame({"sp500": sp500, "slope": rng.normal(0.0, 0.4, size=len(meetings))}, index=meetings)
    return MacroTables(
        calendar=cal,
        dy1=rng.normal(0.0, 0.05, size=t).astype(np.float32),
        hml=rng.normal(0.0, 0.5, size=t).astype(np.float32),
        smb=rng.normal(0.0, 0.5, size=t).astype(np.float32),
        sb_corr63=np.clip(rng.normal(0.0, 0.3, size=t), -1.0, 1.0).astype(np.float32),
        mkt_dd252=np.minimum(rng.normal(-3.0, 4.0, size=t), 0.0).astype(np.float32),
        rf_level=np.abs(rng.normal(2.5, 1.5, size=t)).astype(np.float32),
        rf_chg63=rng.normal(0.0, 0.5, size=t).astype(np.float32),
        meetings=m_df,
    )


def _augment_inputs(panel: Q3Panel, cfg: V2Config, tables: MacroTables) -> dict:
    """Build (history, history_mask, fundamentals, fundamentals_mask, context, shock_2d, is_info).

    All features are computed strictly from observations before each meeting.
    """
    n, length, _ = panel.history.shape
    meetings = pd.DatetimeIndex(panel.meeting)
    unique_m = panel.meetings()

    hist = panel.history
    hist_mask = panel.history_mask
    fund = panel.fundamentals
    fund_mask = panel.fundamentals_mask
    ctx = panel.context

    if cfg.history_macro or cfg.regime:
        cal = tables.calendar
        m_to_prior_pos = {}
        for m in unique_m:
            pos = int(cal.searchsorted(m, side="left"))
            if pos < length:
                raise ValueError(f"meeting {m.date()} has only {pos} prior macro days, need {length}")
            m_to_prior_pos[m] = pos

    if cfg.history_macro:
        macro_seq = np.empty((n, length, 3), dtype=np.float32)
        rate_beta = np.full(n, np.nan, dtype=np.float32)
        hml_beta = np.full(n, np.nan, dtype=np.float32)

        for m in unique_m:
            rows_m = np.flatnonzero(meetings == m)
            pos = m_to_prior_pos[m]
            sl = slice(pos - length, pos)
            dy_w = tables.dy1[sl]
            hml_w = tables.hml[sl]
            smb_w = tables.smb[sl]
            block = np.stack([dy_w, hml_w, smb_w], axis=-1)[None, :, :]
            macro_seq[rows_m] = block

            # Trailing 63d rate_beta and hml_beta from observed days
            rr_63 = panel.history[rows_m, -63:, RET_REL].astype(np.float64)
            obs_63 = hist_mask[rows_m, -63:]
            dy_63 = dy_w[-63:].astype(np.float64)[None, :]
            hml_63 = hml_w[-63:].astype(np.float64)[None, :]

            cnt = np.maximum(obs_63.sum(axis=1, keepdims=True), 1)
            rr_mean = np.where(obs_63, rr_63, 0.0).sum(axis=1, keepdims=True) / cnt
            rr_dev = np.where(obs_63, rr_63 - rr_mean, 0.0)

            dy_mean = np.where(obs_63, dy_63, 0.0).sum(axis=1, keepdims=True) / cnt
            dy_dev = np.where(obs_63, dy_63 - dy_mean, 0.0)
            denom_dy = (dy_dev ** 2).sum(axis=1)
            has_obs = obs_63.sum(axis=1) >= 10

            rb = np.zeros(len(rows_m), dtype=np.float64)
            ok_dy = has_obs & (denom_dy > 1e-8)
            rb[ok_dy] = (rr_dev[ok_dy] * dy_dev[ok_dy]).sum(axis=1) / denom_dy[ok_dy]
            rate_beta[rows_m] = np.where(has_obs, rb, np.nan).astype(np.float32)

            hml_mean = np.where(obs_63, hml_63, 0.0).sum(axis=1, keepdims=True) / cnt
            hml_dev = np.where(obs_63, hml_63 - hml_mean, 0.0)
            denom_hml = (hml_dev ** 2).sum(axis=1)
            hb = np.zeros(len(rows_m), dtype=np.float64)
            ok_hml = has_obs & (denom_hml > 1e-8)
            hb[ok_hml] = (rr_dev[ok_hml] * hml_dev[ok_hml]).sum(axis=1) / denom_hml[ok_hml]
            hml_beta[rows_m] = np.where(has_obs, hb, np.nan).astype(np.float32)

        macro_seq[~hist_mask] = np.nan
        hist = np.concatenate([hist, macro_seq], axis=-1)
        extra_f = np.column_stack([rate_beta, hml_beta])
        extra_f_mask = np.isfinite(extra_f)
        extra_f[~extra_f_mask] = np.nan
        fund = np.concatenate([fund, extra_f], axis=1)
        fund_mask = np.concatenate([fund_mask, extra_f_mask], axis=1)

    if cfg.regime:
        reg_cols = np.empty((n, 4), dtype=np.float32)
        for m in unique_m:
            rows_m = np.flatnonzero(meetings == m)
            p_idx = m_to_prior_pos[m] - 1
            reg_cols[rows_m, 0] = tables.sb_corr63[p_idx]
            reg_cols[rows_m, 1] = tables.mkt_dd252[p_idx]
            reg_cols[rows_m, 2] = tables.rf_level[p_idx]
            reg_cols[rows_m, 3] = tables.rf_chg63[p_idx]
        ctx = np.concatenate([ctx, reg_cols], axis=1)

    s = np.asarray(panel.shock, dtype=np.float32)
    is_info = info_flags(panel, tables.meetings) == 1.0
    if cfg.shock_split == "none":
        shock_mat = s[:, None]
    else:
        shock_mat = split_shock(panel, cfg.shock_split, tables.meetings).astype(np.float32)

    return {
        "history": hist,
        "history_mask": hist_mask,
        "fundamentals": fund,
        "fundamentals_mask": fund_mask,
        "context": ctx,
        "shock_mat": shock_mat,
        "shock_1d": s,
        "is_info": is_info,
    }


@dataclass
class _V2Scaler:
    """Fits ChannelScaler-style median/IQR clipping and scaling on arbitrary channel/feature widths."""

    hist_med: np.ndarray
    hist_iqr: np.ndarray
    fund_med: np.ndarray
    fund_iqr: np.ndarray
    ctx_med: np.ndarray
    ctx_iqr: np.ndarray
    s2_mean: float

    @classmethod
    def fit(cls, aug: dict) -> "_V2Scaler":
        from fedcore.q3.train import _columns

        h_med, h_iqr = _columns(aug["history"], aug["history_mask"])
        f_med, f_iqr = _columns(aug["fundamentals"], aug["fundamentals_mask"])
        ctx = aug["context"]
        c_med, c_iqr = _columns(ctx, np.ones(ctx.shape, dtype=bool))
        s1 = aug["shock_1d"].astype(np.float64)
        s2_mean = max(float(np.mean(s1 ** 2)), 1e-8)
        return cls(h_med, h_iqr, f_med, f_iqr, c_med, c_iqr, s2_mean)

    def transform(self, aug: dict, panel: Q3Panel, target: np.ndarray, device: torch.device) -> dict:
        from fedcore.q3.train import _scale

        h = _scale(aug["history"], self.hist_med, self.hist_iqr)
        h = np.where(aug["history_mask"][..., None], h, np.float32(0.0))
        f = _scale(aug["fundamentals"], self.fund_med, self.fund_iqr)
        f = np.where(aug["fundamentals_mask"], f, np.float32(0.0))
        c = _scale(aug["context"], self.ctx_med, self.ctx_iqr)
        codes = pd.factorize(pd.DatetimeIndex(panel.meeting), sort=False)[0]
        return {
            "history": torch.as_tensor(h, dtype=torch.float32, device=device),
            "history_mask": torch.as_tensor(aug["history_mask"], dtype=torch.bool, device=device),
            "fundamentals": torch.as_tensor(f, dtype=torch.float32, device=device),
            "fundamentals_mask": torch.as_tensor(aug["fundamentals_mask"], dtype=torch.float32, device=device),
            "context": torch.as_tensor(c, dtype=torch.float32, device=device),
            "shock_mat": torch.as_tensor(aug["shock_mat"], dtype=torch.float32, device=device),
            "shock_1d": torch.as_tensor(aug["shock_1d"], dtype=torch.float32, device=device),
            "is_info": torch.as_tensor(aug["is_info"], dtype=torch.bool, device=device),
            "target": torch.as_tensor(target, dtype=torch.float32, device=device),
            "codes": torch.as_tensor(codes, dtype=torch.long, device=device),
        }


class V2StructModel(nn.Module):
    """Centered-a structural model with optional FiLM regime conditioning, multi-shock b head, and b_only."""

    def __init__(
        self,
        encoder: nn.Module,
        n_fund: int,
        n_ctx: int,
        n_shocks: int = 1,
        *,
        regime_film: bool = False,
        b_only: bool = False,
        z_dim: int = 64,
        branch_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.n_shocks = int(n_shocks)
        self.regime_film = bool(regime_film)
        self.b_only = bool(b_only)
        self.fund = _MLP(n_fund * 2, branch_dim, dropout)
        self.context_mlp = _MLP(n_ctx, branch_dim, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(int(encoder.out_dim) + branch_dim, z_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
        )
        if self.regime_film:
            self.film_gamma = nn.Linear(branch_dim, z_dim)
            self.film_beta = nn.Linear(branch_dim, z_dim)
            nn.init.zeros_(self.film_gamma.weight)
            nn.init.zeros_(self.film_gamma.bias)
            nn.init.zeros_(self.film_beta.weight)
            nn.init.zeros_(self.film_beta.bias)
        self.head_a = None if self.b_only else nn.Linear(z_dim, 1)
        self.head_b = nn.Linear(z_dim + branch_dim, self.n_shocks)
        self.a0 = nn.Parameter(torch.zeros(1), requires_grad=not self.b_only)

    def forward(self, history, history_mask, fundamentals, fundamentals_mask, context):
        dtype = self.head_b.weight.dtype
        h = self.encoder(history.to(dtype), history_mask)
        f = self.fund(torch.cat((fundamentals.to(dtype), fundamentals_mask.to(dtype)), dim=-1))
        z = self.fusion(torch.cat((h, f), dim=-1))
        c = self.context_mlp(context.to(dtype))
        if self.head_a is None:
            a_raw = torch.zeros(z.shape[0], dtype=dtype, device=z.device)
        else:
            a_raw = self.head_a(z).squeeze(-1)
        z_b = z * (1.0 + torch.tanh(self.film_gamma(c))) + self.film_beta(c) if self.regime_film else z
        b_mat = self.head_b(torch.cat((z_b, c), dim=-1))
        return a_raw, b_mat


def v2_outputs(model: V2StructModel, batch: dict, rows: torch.Tensor, shock_split: str):
    """Return (mu, a, b_eff, b_mat) for whole-meeting ``rows``."""
    args = [batch[k][rows] for k in ("history", "history_mask", "fundamentals", "fundamentals_mask", "context")]
    a_raw, b_mat = model(*args)
    inverse, width, count = _segments(batch["codes"][rows])
    if model.b_only:
        a = torch.zeros_like(a_raw)
    else:
        a = a_raw - _segment_mean(a_raw, inverse, width, count)[inverse] + model.a0

    shock_mat = batch["shock_mat"][rows].to(b_mat.dtype)
    shock_resp = (b_mat * shock_mat).sum(dim=-1)
    mu = a + shock_resp

    if shock_split == "none":
        b_eff = b_mat.squeeze(-1)
    elif shock_split == "policy_info":
        is_info = batch["is_info"][rows]
        b_eff = torch.where(is_info, b_mat[:, 1], b_mat[:, 0])
    else:
        # First column multiplies s (10 * STMT); second multiplies slope_shock
        b_eff = b_mat[:, 0]
    return mu, a, b_eff, b_mat


def _meeting_weights(batch: dict, rows: torch.Tensor, inverse: torch.Tensor, width: int, count: torch.Tensor,
                     cfg: V2Config, s2_mean: float) -> torch.Tensor:
    if cfg.shock_weight <= 0.0:
        return torch.ones(width, dtype=torch.float64, device=rows.device)
    s_m = _segment_mean(batch["shock_1d"][rows].double(), inverse, width, count)
    w = 1.0 + cfg.shock_weight * (s_m ** 2) / max(s2_mean, 1e-8)
    return torch.clamp(w, min=1.0, max=cfg.w_max)


def evaluate_v2(model: V2StructModel, batch: dict, groups: list[np.ndarray], cfg: V2Config, s2_mean: float):
    """Weighted objective, unweighted mean part, unweighted cross-firm part."""
    model.eval()
    obj_parts, mean_parts, cross_parts = [], [], []
    with torch.no_grad():
        for idx in _chunks(groups):
            rows = torch.as_tensor(idx, device=batch["target"].device)
            mu, _, _, _ = v2_outputs(model, batch, rows, cfg.shock_split)
            codes = batch["codes"][rows]
            inverse, width, count = _segments(codes)
            y = batch["target"][rows].double()
            m_pred = mu.double()
            y_bar = _segment_mean(y, inverse, width, count)
            m_bar = _segment_mean(m_pred, inverse, width, count)
            m_loss = (y_bar - m_bar) ** 2
            c_loss = _segment_mean(((y - y_bar[inverse]) - (m_pred - m_bar[inverse])) ** 2, inverse, width, count)
            w = _meeting_weights(batch, rows, inverse, width, count, cfg, s2_mean)
            obj_parts.append(w * (cfg.lambda_mean * m_loss + c_loss))
            mean_parts.append(m_loss)
            cross_parts.append(c_loss)
    model.train()
    return (
        torch.cat(obj_parts).mean().item(),
        torch.cat(mean_parts).mean().item(),
        torch.cat(cross_parts).mean().item(),
    )


def fit_v2(
    model: V2StructModel,
    panel: Q3Panel,
    aug: dict,
    target: np.ndarray,
    cfg: V2Config,
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_frac: float = 0.15,
    seed: int = 0,
    meetings_per_batch: int = 4,
):
    """Fit ``model`` on ``target`` (e.g. CAPM residuals) with optional shock-weighted loss."""
    val_m = validation_meetings(panel, val_frac)
    train_m = panel.meetings()[: -len(val_m)]
    tr_idx = _meeting_rows(panel, train_m)
    va_idx = _meeting_rows(panel, val_m)

    train_panel = panel.subset(tr_idx)
    val_panel = panel.subset(va_idx)
    train_aug = {k: v[tr_idx] for k, v in aug.items()}
    val_aug = {k: v[va_idx] for k, v in aug.items()}

    scaler = _V2Scaler.fit(train_aug)
    device = next(model.parameters()).device
    train_batch = scaler.transform(train_aug, train_panel, target[tr_idx], device)
    val_batch = scaler.transform(val_aug, val_panel, target[va_idx], device)

    train_groups = _meeting_index_groups(train_panel.meeting)
    val_groups = _meeting_index_groups(val_panel.meeting)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay
    )
    rng = np.random.default_rng(seed)
    best, best_epoch, stale = math.inf, 0, 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    train_curve, val_curve = [], []

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        order = rng.permutation(len(train_groups))
        for start in range(0, len(order), meetings_per_batch):
            chosen = order[start : start + meetings_per_batch]
            rows = torch.as_tensor(np.concatenate([train_groups[j] for j in chosen]), device=device)
            optimizer.zero_grad(set_to_none=True)
            mu, _, _, _ = v2_outputs(model, train_batch, rows, cfg.shock_split)
            codes = train_batch["codes"][rows]
            inverse, width, count = _segments(codes)
            y = train_batch["target"][rows]
            y_bar = _segment_mean(y, inverse, width, count)
            m_bar = _segment_mean(mu, inverse, width, count)
            m_loss = (y_bar - m_bar) ** 2
            c_loss = _segment_mean(((y - y_bar[inverse]) - (mu - m_bar[inverse])) ** 2, inverse, width, count)
            w = _meeting_weights(train_batch, rows, inverse, width, count, cfg, scaler.s2_mean).to(mu.dtype)
            loss = (w * (cfg.lambda_mean * m_loss + c_loss)).mean() * (len(chosen) / meetings_per_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        train_curve.append(evaluate_v2(model, train_batch, train_groups, cfg, scaler.s2_mean)[0])
        val_curve.append(evaluate_v2(model, val_batch, val_groups, cfg, scaler.s2_mean)[0])
        if val_curve[-1] < best:
            best, best_epoch, stale = val_curve[-1], epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return argparse.Namespace(
        scaler=scaler,
        train_loss=train_curve,
        val_loss=val_curve,
        best_epoch=best_epoch,
        best_val_loss=best,
        train_meetings=pd.DatetimeIndex(train_m),
        val_meetings=pd.DatetimeIndex(val_m),
    )


def predict_v2(model: V2StructModel, panel: Q3Panel, aug: dict, scaler: _V2Scaler, cfg: V2Config):
    device = next(model.parameters()).device
    dummy_target = np.zeros(len(panel.target), dtype=np.float32)
    batch = scaler.transform(aug, panel, dummy_target, device)
    n = len(panel.target)
    n_shocks = aug["shock_mat"].shape[1]
    mu_out = np.empty(n, dtype=np.float64)
    a_out = np.empty(n, dtype=np.float64)
    b_eff_out = np.empty(n, dtype=np.float64)
    b_mat_out = np.empty((n, n_shocks), dtype=np.float64)

    model.eval()
    with torch.no_grad():
        for idx in _chunks(_meeting_index_groups(panel.meeting)):
            rows = torch.as_tensor(idx, device=device)
            mu, a, b_eff, b_mat = v2_outputs(model, batch, rows, cfg.shock_split)
            mu_out[idx] = mu.double().cpu().numpy()
            a_out[idx] = a.double().cpu().numpy()
            b_eff_out[idx] = b_eff.double().cpu().numpy()
            b_mat_out[idx] = b_mat.double().cpu().numpy()
    return mu_out, a_out, b_eff_out, b_mat_out


class V2TwoPartArm:
    """Modular two-part arm: base model (CAPM or pooled) + ensemble of demeaned V2StructModels."""

    def __init__(
        self,
        seed: int = 0,
        cfg: V2Config | None = None,
        *,
        device: str | None = None,
        macro_tables: MacroTables | None = None,
        encoder_fn=None,
        **cfg_kwargs,
    ):
        self.seed = int(seed)
        self.cfg = cfg if cfg is not None else V2Config(**cfg_kwargs)
        self.device = device
        self.macro_tables = macro_tables
        self.encoder_fn = encoder_fn
        self.base_model = None
        self.nets: list[ tuple[V2StructModel, _V2Scaler] ] = []
        self.fit_result = None

    def _tables(self) -> MacroTables:
        return self.macro_tables if self.macro_tables is not None else load_macro_tables()

    def _make_encoder(self, n_channels: int) -> nn.Module:
        if self.encoder_fn is not None:
            return self.encoder_fn(n_channels)
        return PatchTransformerEncoder(n_channels=n_channels)

    def fit(self, panel: Q3Panel, seed: int | None = None, **_) -> None:
        seed = self.seed if seed is None else int(seed)
        self.base_model = CAPMPooled() if self.cfg.base == "capm" else PooledOLS()
        self.base_model.fit(panel)
        mu_base, _, _ = self.base_model.predict(panel)
        residual_target = (np.asarray(panel.target, dtype=np.float64) - mu_base).astype(np.float32)

        aug = _augment_inputs(panel, self.cfg, self._tables())
        n_channels = aug["history"].shape[-1]
        n_fund = aug["fundamentals"].shape[1]
        n_ctx = aug["context"].shape[1]
        n_shocks = aug["shock_mat"].shape[1]

        self.nets = []
        fit_results = []
        for j in range(self.cfg.members):
            member_seed = seed * 100 + j
            seed_everything(member_seed)
            model = V2StructModel(
                self._make_encoder(n_channels),
                n_fund=n_fund,
                n_ctx=n_ctx,
                n_shocks=n_shocks,
                regime_film=self.cfg.regime,
                b_only=self.cfg.b_only,
            ).to(resolve_device(self.device))
            seed_everything(member_seed)
            res = fit_v2(model, panel, aug, residual_target, self.cfg, seed=member_seed)
            self.nets.append((model, res.scaler))
            fit_results.append(res)
        self.fit_result = fit_results[0]

    def predict(self, panel: Q3Panel):
        if self.base_model is None or not self.nets:
            raise RuntimeError("fit V2TwoPartArm before predict")
        aug = _augment_inputs(panel, self.cfg, self._tables())
        codes = pd.factorize(pd.DatetimeIndex(panel.meeting))[0]

        def demean_1d(x: np.ndarray) -> np.ndarray:
            return x - pd.Series(x).groupby(codes).transform("mean").to_numpy()

        def demean_2d(x: np.ndarray) -> np.ndarray:
            return np.column_stack([demean_1d(x[:, k]) for k in range(x.shape[1])])

        a_devs, b_eff_devs, b_mat_devs = [], [], []
        for model, scaler in self.nets:
            _, a_m, b_eff_m, b_mat_m = predict_v2(model, panel, aug, scaler, self.cfg)
            a_devs.append(np.zeros_like(a_m) if self.cfg.b_only else demean_1d(a_m))
            b_eff_devs.append(demean_1d(b_eff_m))
            b_mat_devs.append(demean_2d(b_mat_m))

        da = np.mean(a_devs, axis=0)
        db_eff = np.mean(b_eff_devs, axis=0)
        db_mat = np.mean(b_mat_devs, axis=0)

        mu_base, a_base, b_base = self.base_model.predict(panel)
        shock_mat = aug["shock_mat"].astype(np.float64)
        d_shock_resp = (db_mat * shock_mat).sum(axis=-1)

        mu = mu_base + da + d_shock_resp
        a = a_base + da
        if self.cfg.shock_split == "target_slope":  # slope response is part of a; b multiplies s
            a = a + db_mat[:, 1] * shock_mat[:, 1]
        b = b_base + db_eff
        return mu, a, b


class SectorTwoShock:
    """``capm_two_shock`` plus sector intercepts and sector x policy / information loadings, by OLS.

    Needs a panel built with ``sector=True`` (GICS one-hot in the last fundamentals columns; an
    unknown sector gets no sector terms). ``slope=True`` adds sector x slope change, kept in ``a``
    like ``capm_target_slope``, so ``mu = a + b*s`` holds."""

    def __init__(self, slope: bool = False, table=None):
        self.slope = slope
        self.table = table
        self.fit_result = None

    def _parts(self, panel: Q3Panel):
        from fedcore.q3.baselines import _beta
        from fedenc.workflow import GICS_SECTORS

        D = np.nan_to_num(np.asarray(panel.fundamentals[:, -len(GICS_SECTORS):], dtype=np.float64))
        S = split_shock(panel, "policy_info", self.table)
        slope = split_shock(panel, "target_slope", self.table)[:, 1] if self.slope else np.zeros(len(D))
        return _beta(panel), D, S, slope

    def _design(self, beta, D, S, slope):
        cols = [np.ones(len(beta)), beta, beta * S[:, 0], beta * S[:, 1], D, D * S[:, [0]], D * S[:, [1]]]
        if self.slope:
            cols += [beta * slope, D * slope[:, None]]
        return np.column_stack(cols)

    def fit(self, panel: Q3Panel, **kwargs) -> None:
        del kwargs
        parts = self._parts(panel)
        self.coef_, *_ = np.linalg.lstsq(self._design(*parts), np.asarray(panel.target, dtype=np.float64), rcond=None)

    def predict(self, panel: Q3Panel):
        beta, D, S, slope = self._parts(panel)
        k = D.shape[1]
        c, d, kp, ki = self.coef_[:4]
        g_a, g_p, g_i = self.coef_[4:4 + k], self.coef_[4 + k:4 + 2 * k], self.coef_[4 + 2 * k:4 + 3 * k]
        a = c + d * beta + D @ g_a
        if self.slope:
            a = a + self.coef_[4 + 3 * k] * beta * slope + (D @ self.coef_[5 + 3 * k:5 + 4 * k]) * slope
        info = S[:, 1] != 0.0
        b = np.where(info, ki * beta + D @ g_i, kp * beta + D @ g_p)
        s = np.asarray(panel.shock, dtype=np.float64)
        return a + b * s, a, b


def ablation_makers(device: str | None = None, members: int = 3) -> dict:
    """Modular progressive ablation arms + references."""
    return {
        "v1_two_part": lambda seed: V2TwoPartArm(seed, base="pooled", members=members, device=device),
        "v2_capm_base": lambda seed: V2TwoPartArm(seed, base="capm", members=members, device=device),
        "v2_macro_hist": lambda seed: V2TwoPartArm(
            seed, base="capm", history_macro=True, members=members, device=device
        ),
        "v2_regime": lambda seed: V2TwoPartArm(
            seed, base="capm", history_macro=True, regime=True, members=members, device=device
        ),
        "v2_policy_info": lambda seed: V2TwoPartArm(
            seed, base="capm", history_macro=True, regime=True, shock_split="policy_info",
            members=members, device=device,
        ),
        "v2_target_slope": lambda seed: V2TwoPartArm(
            seed, base="capm", history_macro=True, regime=True, shock_split="target_slope",
            members=members, device=device,
        ),
        "v2_shock_weighted": lambda seed: V2TwoPartArm(
            seed, base="capm", history_macro=True, regime=True, shock_split="policy_info",
            shock_weight=2.0, members=members, device=device,
        ),
        "v2_b_only": lambda seed: V2TwoPartArm(
            seed, base="capm", history_macro=True, regime=True, shock_split="policy_info",
            shock_weight=2.0, b_only=True, members=members, device=device,
        ),
        **REFERENCE_MAKERS,
    }


def run_ablation(
    through: str = "2021-12-31",
    seeds: tuple[int, ...] = (0, 1, 2),
    members: int = 3,
    min_train: int = 60,
    test_size: int = 16,
    device: str | None = None,
    arms: list[str] | None = None,
    store=None,
):
    """Run the modular v2 ablation suite on shared walk-forward folds and print diagnostics. Training."""
    from fedcore.q3.real import UNIVERSES
    from fedcore.results import ResultStore
    from fedenc import diagnostics
    from fedenc.config import RESULTS
    from fedenc.runs import save_runs
    from fedenc.workflow import bounded_panel

    panel, through_ts = bounded_panel(through)
    store = store or ResultStore(RESULTS)
    all_makers = ablation_makers(device=device, members=members)
    if arms is not None:
        keep = list(dict.fromkeys([*arms, *REFERENCE_MAKERS]))
        makers = {k: all_makers[k] for k in keep if k in all_makers}
    else:
        makers = all_makers

    prefix = f"v2-ablation-to{through_ts.year}"
    summary = save_runs(
        panel,
        makers,
        prefix=prefix,
        params=dict(through=str(through_ts.date()), members=members),
        min_train=min_train,
        test_size=test_size,
        seeds=seeds,
        store=store,
        datasets=UNIVERSES["sp500"].inputs,
    )
    runs = {arm: Path(path) for arm, path in zip(summary["arm"], summary["path"])}
    tables = diagnostics.run(runs, V2_REFERENCES, store=store, panel=panel, label=f"{prefix}-diagnostics")
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
        for name, table in tables.items():
            print(f"\n== {name}\n{table.to_string(index=False)}", flush=True)
    return summary, tables


def run_single(
    name: str,
    cfg: V2Config,
    *,
    through: str = "2021-12-31",
    seeds: tuple[int, ...] = (0, 1, 2),
    min_train: int = 60,
    test_size: int = 16,
    device: str | None = None,
    store=None,
):
    """Run one custom modular configuration against the CAPM references and pooled. Training."""
    from fedcore.q3.real import UNIVERSES
    from fedcore.results import ResultStore
    from fedenc import diagnostics
    from fedenc.config import RESULTS
    from fedenc.runs import save_runs
    from fedenc.workflow import bounded_panel

    panel, through_ts = bounded_panel(through)
    store = store or ResultStore(RESULTS)
    makers = {
        name: lambda seed: V2TwoPartArm(seed, cfg, device=device),
        **REFERENCE_MAKERS,
    }
    prefix = f"v2-{name}-to{through_ts.year}"
    summary = save_runs(
        panel,
        makers,
        prefix=prefix,
        params=dict(through=str(through_ts.date()), **dataclasses.asdict(cfg)),
        min_train=min_train,
        test_size=test_size,
        seeds=seeds,
        store=store,
        datasets=UNIVERSES["sp500"].inputs,
    )
    runs = {arm: Path(path) for arm, path in zip(summary["arm"], summary["path"])}
    tables = diagnostics.run(runs, V2_REFERENCES, store=store, panel=panel, label=f"{prefix}-diagnostics")
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
        for tname, table in tables.items():
            print(f"\n== {tname}\n{table.to_string(index=False)}", flush=True)
    return summary, tables


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    q_ab = sub.add_parser("ablation", help="run the modular v2 ablation suite")
    q_ab.add_argument("--through", default="2021-12-31")
    q_ab.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    q_ab.add_argument("--members", type=int, default=3)
    q_ab.add_argument("--min-train", type=int, default=60)
    q_ab.add_argument("--test-size", type=int, default=16)
    q_ab.add_argument("--arms", nargs="+", help="optional subset of arm names to run")
    q_ab.add_argument("--device")

    q_s = sub.add_parser("single", help="run one custom modular configuration")
    q_s.add_argument("--name", default="v2_custom")
    q_s.add_argument("--through", default="2021-12-31")
    q_s.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    q_s.add_argument("--members", type=int, default=3)
    q_s.add_argument("--min-train", type=int, default=60)
    q_s.add_argument("--test-size", type=int, default=16)
    q_s.add_argument("--base", default="capm", choices=BASES)
    q_s.add_argument("--history-macro", action="store_true")
    q_s.add_argument("--regime", action="store_true")
    q_s.add_argument("--shock-split", default="none", choices=SHOCK_SPLITS)
    q_s.add_argument("--shock-weight", type=float, default=0.0)
    q_s.add_argument("--w-max", type=float, default=W_MAX_DEFAULT)
    q_s.add_argument("--b-only", action="store_true")
    q_s.add_argument("--lambda-mean", type=float, default=LAMBDA_CROSS_FIRM)
    q_s.add_argument("--device")

    args = p.parse_args(argv)
    if args.command == "ablation":
        run_ablation(
            through=args.through,
            seeds=tuple(args.seeds),
            members=args.members,
            min_train=args.min_train,
            test_size=args.test_size,
            device=args.device,
            arms=args.arms,
        )
    else:
        cfg = V2Config(
            base=args.base,
            history_macro=args.history_macro,
            regime=args.regime,
            shock_split=args.shock_split,
            shock_weight=args.shock_weight,
            w_max=args.w_max,
            b_only=args.b_only,
            lambda_mean=args.lambda_mean,
            members=args.members,
        )
        run_single(
            args.name,
            cfg,
            through=args.through,
            seeds=tuple(args.seeds),
            min_train=args.min_train,
            test_size=args.test_size,
            device=args.device,
        )


if __name__ == "__main__":
    main()

