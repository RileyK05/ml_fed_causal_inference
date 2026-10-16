"""Inputs shared by the cheap tests (docs/encoder/cheap-test-menu.md).

``CheapData`` holds the Q3 firm-meeting panel plus the firm x trading-day grid it was cut
from, the daily factors (market, 1-year yield change, SMB, HML), every FOMC statement date,
and earnings report dates. ``load_real`` builds it from the registered WRDS datasets;
``synthetic_data`` builds a small consistent stand-in for unit tests and smoke runs.

Units: returns and factors in percent, ``dy`` in basis points, shocks ``s`` in 10bp.
Gaps stay NaN: nothing is filled.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fedcore.q3.contracts import L, Q3Panel, RET


@dataclass(frozen=True)
class Config:
    """Splits and budgets. Defaults follow docs/encoder/cheap-test-menu.md, section 10."""

    dev_meetings: int = 120         # development period: the first 120 meetings (through 2009-03-18)
    inner_min_train: int = 72       # inner walk-forward inside the development period
    inner_test: int = 12
    outer_min_train: int = 120      # the planned outer design (R1 only)
    outer_test: int = 16
    val_meetings: int = 18          # single train/validation split inside dev (C2, C4): last 15%
    ridge_alpha: float = 1.0        # same as the ridge_interact baseline
    n_boot: int = 2000
    n_perm: int = 200
    window: int = 63                # trailing loading window (trading days)
    min_obs: int = 40               # minimum observed days for a window loading
    sample_every: int = 21          # firm-date sampling step for R5 / C3
    horizons: tuple = (21, 63, 126)
    r1_effects: tuple = (0.0, 0.25, 0.5, 1.0, 2.0)   # planted sd of b, % per 10bp
    r2_min_meetings: int = 80
    r8_min_meetings: int = 30
    synthetic_firms: int = 150      # S1-S5 simulated panel size
    synthetic_meetings: int = 160
    seed: int = 0


@dataclass
class CheapData:
    panel: Q3Panel
    rows: pd.DataFrame              # meeting, permno, gsector (one row per panel row)
    calendar: pd.DatetimeIndex
    permnos: np.ndarray             # sorted firm ids of the grid
    channels: np.ndarray            # (F, D, C) float32, NaN where unobserved
    observed: np.ndarray            # (F, D) bool
    mkt: np.ndarray                 # (D,) market return, %
    dy: np.ndarray                  # (D,) 1-year yield change, bp
    smb: np.ndarray                 # (D,) %
    hml: np.ndarray                 # (D,) %
    fomc: pd.DatetimeIndex          # every statement date, scheduled or not
    earnings: pd.DataFrame          # permno, rdq
    fund_names: tuple
    context_names: tuple
    source: str = "real"
    stmt: pd.DataFrame | None = None        # announcement_date, STMT (audit A6)
    y1: pd.Series | None = None             # 1-year yield level by date (audit A6)
    releases: pd.DatetimeIndex | None = None  # macro release dates (R7)
    datasets: tuple = ()
    _memo: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ indexing
    @property
    def ret(self) -> np.ndarray:
        return self.channels[:, :, RET]

    def meetings(self) -> pd.DatetimeIndex:
        return self.panel.meetings()

    def meeting_table(self) -> pd.DataFrame:
        """One row per meeting: date, s, calendar position, era (5-year bin)."""
        if "meeting_table" not in self._memo:
            m = pd.DataFrame({"meeting": pd.to_datetime(self.panel.meeting), "s": self.panel.shock.astype(float)})
            m = m.drop_duplicates("meeting").sort_values("meeting").reset_index(drop=True)
            m["pos"] = self.calendar.get_indexer(m["meeting"])
            m["era"] = (m["meeting"].dt.year // 5) * 5
            self._memo["meeting_table"] = m
        return self._memo["meeting_table"]

    def row_index(self) -> tuple[np.ndarray, np.ndarray]:
        """(grid firm index, calendar position of the meeting) for every panel row."""
        if "row_index" not in self._memo:
            firm = np.searchsorted(self.permnos, np.asarray(self.panel.firm_id, dtype=np.int64))
            pos = self.calendar.get_indexer(pd.to_datetime(self.panel.meeting))
            self._memo["row_index"] = (firm, pos)
        return self._memo["row_index"]

    def ordinary_days(self) -> np.ndarray:
        """(D,) True on trading days that are not an FOMC statement date."""
        return ~self.calendar.isin(self.fomc)

    def dev_rows(self, cfg: Config) -> np.ndarray:
        """Boolean mask of panel rows in the development period."""
        dev = self.meetings()[: cfg.dev_meetings]
        return np.isin(np.asarray(self.panel.meeting, dtype="datetime64[ns]"), dev.to_numpy(dtype="datetime64[ns]"))

    def dev_end_pos(self, cfg: Config) -> int:
        """Calendar position of the first meeting after the development period (exclusive bound)."""
        m = self.meetings()
        if len(m) <= cfg.dev_meetings:
            return len(self.calendar)
        return int(self.calendar.get_indexer([m[cfg.dev_meetings]])[0])

    def factors(self, names=("mkt", "dy")) -> np.ndarray:
        return np.column_stack([getattr(self, n) for n in names]).astype(np.float64)


# ---------------------------------------------------------------------- window loadings

def window_ols(
    data: CheapData,
    firm: np.ndarray,
    end: np.ndarray,
    width: int,
    factors=("mkt", "dy"),
    day_ok: np.ndarray | None = None,
    min_obs: int = 40,
    chunk: int = 8000,
) -> dict:
    """OLS of the firm's return on an intercept and ``factors`` over the ``width`` trading
    days ending at ``end`` (inclusive), one regression per (firm, end) pair.

    A day enters when the firm is observed, every factor is finite and ``day_ok`` holds.
    Returns ``coef`` and ``se`` shaped (N, 1 + k) (intercept first), ``nobs`` (N,), and
    ``cond`` (N,), the condition number of the factor design. Fewer than ``min_obs`` days
    gives NaN.
    """
    x_all = data.factors(factors)
    k = x_all.shape[1] + 1
    finite = np.isfinite(x_all).all(axis=1)
    if day_ok is not None:
        finite &= np.asarray(day_ok, dtype=bool)
    firm = np.asarray(firm, dtype=np.int64)
    end = np.asarray(end, dtype=np.int64)
    n = firm.size
    coef = np.full((n, k), np.nan)
    se = np.full((n, k), np.nan)
    nobs = np.zeros(n, dtype=np.int64)
    cond = np.full(n, np.nan)
    offsets = np.arange(-width + 1, 1)
    ret = data.ret
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        days = end[lo:hi, None] + offsets[None, :]
        inside = (days >= 0) & (days < ret.shape[1])
        d = np.clip(days, 0, ret.shape[1] - 1)
        f = firm[lo:hi, None]
        valid = inside & data.observed[f, d] & finite[d]
        y = np.where(valid, ret[f, d].astype(np.float64), 0.0)
        x = np.concatenate((np.ones(d.shape + (1,)), x_all[d]), axis=-1)
        x = np.where(valid[..., None], x, 0.0)
        xtx = np.einsum("nwi,nwj->nij", x, x)
        xty = np.einsum("nwi,nw->ni", x, y)
        cnt = valid.sum(axis=1)
        ok = cnt >= max(min_obs, k + 1)
        safe = np.where(ok[:, None, None], xtx, np.eye(k)[None])
        try:
            inv = np.linalg.inv(safe)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(safe)
        b = np.einsum("nij,nj->ni", inv, xty)
        resid = np.where(valid, y - np.einsum("nwi,ni->nw", x, b), 0.0)
        dof = np.maximum(cnt - k, 1)
        s2 = (resid ** 2).sum(axis=1) / dof
        var = s2[:, None] * np.diagonal(inv, axis1=1, axis2=2)
        sv = np.linalg.svd(safe[:, 1:, 1:] if k > 1 else safe, compute_uv=False)
        c = sv[:, 0] / np.maximum(sv[:, -1], 1e-300)
        coef[lo:hi] = np.where(ok[:, None], b, np.nan)
        se[lo:hi] = np.where(ok[:, None], np.sqrt(np.clip(var, 0.0, None)), np.nan)
        nobs[lo:hi] = cnt
        cond[lo:hi] = np.where(ok, c, np.nan)
    return {"coef": coef, "se": se, "nobs": nobs, "cond": cond}


# ---------------------------------------------------------------------- real data

REAL_DATASETS = (
    "crsp_daily", "sp500_members", "ff_factors_daily", "ccm_link", "comp_fundq", "comp_gics",
    "usmpd_statements", "mps_surprises", "treasury_1y", "bls_macro_release_dates",
)


def load_real() -> CheapData:
    """The cached real panel plus the daily grid and factors. Loads ~1 GB into memory."""
    from fedcore.data import load
    from fedcore.q3.real import CONTEXT_NAMES, FUNDAMENTAL_NAMES, _grid, load_real_panel

    panel, rows, _ = load_real_panel()
    daily = load("crsp_daily", columns=["permno", "dlyret", "dlyvol", "dlyprc", "shrout"])
    ff = load("ff_factors_daily")
    grid = _grid(daily, ff[["date", "mktrf", "rf"]])
    del daily
    calendar = grid["calendar"]
    ffd = ff.set_index("date").reindex(calendar)

    y1 = load("treasury_1y").rename(columns={"Date": "date", "SVENY01": "y1"}).set_index("date")["y1"]
    level = y1.reindex(calendar).to_numpy(dtype=np.float64)
    dy = np.full(calendar.size, np.nan)
    dy[1:] = 100.0 * (level[1:] - level[:-1])  # bp; NaN when either day is missing

    stmt_all = load("usmpd_statements")
    fomc = pd.DatetimeIndex(pd.to_datetime(stmt_all["Date"]).dt.normalize().unique())
    mps = load("mps_surprises").rename(columns={"Date": "announcement_date"})[["announcement_date", "STMT"]]

    fundq = load("comp_fundq", columns=["gvkey", "rdq"])
    link = load("ccm_link")
    earnings = _earnings_permnos(fundq, link)
    macro = load("bls_macro_release_dates")  # CPI from 1999-06, jobs reports from 1994-02
    releases = pd.DatetimeIndex(pd.to_datetime(macro["release_date"]).unique())

    return CheapData(
        panel=panel, rows=rows.reset_index(drop=True), calendar=calendar, permnos=grid["permnos"],
        channels=grid["channels"], observed=grid["observed"], mkt=grid["mkt"], dy=dy,
        smb=100.0 * ffd["smb"].to_numpy(dtype=np.float64), hml=100.0 * ffd["hml"].to_numpy(dtype=np.float64),
        fomc=fomc, earnings=earnings, fund_names=tuple(FUNDAMENTAL_NAMES), context_names=tuple(CONTEXT_NAMES),
        source="real", stmt=mps, y1=y1, releases=releases, datasets=REAL_DATASETS,
    )


def _earnings_permnos(fundq: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    """Earnings report dates mapped to the permno linked on that date (primary links first)."""
    fq = fundq.dropna(subset=["rdq"]).assign(gvkey=lambda d: d["gvkey"].astype(str))
    fq = fq.assign(rdq=pd.to_datetime(fq["rdq"])).drop_duplicates(["gvkey", "rdq"])
    lk = link.assign(gvkey=link["gvkey"].astype(str), linkdt=pd.to_datetime(link["linkdt"]),
                     linkenddt=pd.to_datetime(link["linkenddt"]))
    j = fq.merge(lk, on="gvkey")
    ok = (j["linkdt"] <= j["rdq"]) & (j["linkenddt"].fillna(pd.Timestamp.max) >= j["rdq"])
    j = j[ok].sort_values(["gvkey", "rdq", "linkprim"], ascending=[True, True, False])
    j = j.drop_duplicates(["gvkey", "rdq"])
    return pd.DataFrame({"permno": j["lpermno"].astype("int64").to_numpy(), "rdq": j["rdq"].to_numpy()})


# ---------------------------------------------------------------------- synthetic stand-in

def synthetic_data(n_firms: int = 40, n_meetings: int = 60, seed: int = 0) -> CheapData:
    """A small, internally consistent CheapData for tests and smoke runs (no files read).

    Built like ``make_synthetic_panel``: one simulated universe, meetings every 30 trading
    days, the meeting-day return written into the grid so history, target and grid agree.
    """
    from fedcore.q3.synthetic import MEETING_GAP, _simulate, synthetic_components

    rng = np.random.default_rng(seed)
    pos = L + np.arange(n_meetings) * MEETING_GAP
    n_days = int(pos[-1]) + 2
    sim = _simulate(n_firms, n_days, rng)
    channels = sim["channels"].copy()
    observed = sim["observed"].copy()
    calendar = pd.bdate_range("1994-01-03", periods=n_days)
    s_m = rng.laplace(0.0, 0.2, size=n_meetings)

    fund_f = sim["fundamentals"].astype(np.float32)
    ctx_m = rng.normal(0.0, 1.0, size=(n_meetings, 4)).astype(np.float32)
    parts = []
    for j, p in enumerate(pos):  # in order, so later histories see earlier meeting-day returns
        one = Q3Panel(
            meeting=np.repeat(calendar[p].to_datetime64(), n_firms).astype("datetime64[ns]"),
            firm_id=np.arange(n_firms, dtype=np.int64),
            history=channels[:, p - L : p].copy(), history_mask=observed[:, p - L : p].copy(),
            fundamentals=fund_f, fundamentals_mask=np.ones(fund_f.shape, bool),
            context=np.repeat(ctx_m[j : j + 1], n_firms, axis=0),
            shock=np.full(n_firms, s_m[j], np.float32), target=np.zeros(n_firms, np.float32),
        )
        comp = synthetic_components(one)
        target = (comp["a_true"] + comp["b_true"] * s_m[j] + rng.normal(0.0, 1.5, n_firms)).to_numpy(np.float32)
        channels[:, p, :] = np.where(np.isfinite(channels[:, p, :]), channels[:, p, :], 0.0)
        channels[:, p, RET] = target
        observed[:, p] = True
        parts.append(Q3Panel(**{**one.__dict__, "target": target}))
    panel = Q3Panel(**{
        name: np.concatenate([getattr(part, name) for part in parts])
        for name in ("meeting", "firm_id", "history", "history_mask", "fundamentals", "fundamentals_mask",
                     "context", "shock", "target")
    })
    panel.validate()
    meeting, firm_id = panel.meeting, panel.firm_id

    market = sim["market"]
    dy = 1.2 * market + rng.normal(0.0, 3.0, n_days)
    dy[pos] = 10.0 * s_m
    quarters = np.arange(40, n_days, 63)
    earn = pd.DataFrame(
        [(f, calendar[min(q + int(rng.integers(0, 10)), n_days - 1)]) for f in range(n_firms) for q in quarters],
        columns=["permno", "rdq"],
    )
    rows = pd.DataFrame({
        "meeting": pd.to_datetime(meeting), "permno": firm_id,
        "gsector": np.array(["10", "20", "30"])[firm_id % 3],
    })
    return CheapData(
        panel=panel, rows=rows, calendar=calendar, permnos=np.arange(n_firms, dtype=np.int64),
        channels=channels, observed=observed, mkt=market, dy=dy,
        smb=rng.normal(0.0, 0.5, n_days), hml=rng.normal(0.0, 0.5, n_days),
        fomc=pd.DatetimeIndex(calendar[pos]), earnings=earn,
        fund_names=("leverage", "log_size", "profitability"), context_names=("rate", "vix", "slope", "credit"),
        source="synthetic", stmt=pd.DataFrame({"announcement_date": calendar[pos], "STMT": s_m / 10.0}),
        releases=pd.DatetimeIndex([]),
    )
