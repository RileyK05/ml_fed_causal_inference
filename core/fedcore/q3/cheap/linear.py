"""Linear building blocks for the cheap tests: the ridge design, within-meeting OLS,
meeting-clustered standard errors, per-meeting scoring, and a fast meeting bootstrap.

The ridge design matches ``fedcore.q3.baselines.RidgeInteract``: summary features,
fundamentals and context, median-imputed and clipped at the median ± 5 IQR, then z-scored,
with every statistic fit on the training rows only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from fedcore.protocol import walk_forward
from fedcore.q3.features import last_observed, summary_feature_names, summary_features
from fedcore.q3.train import CLIP, _columns

from fedcore.q3.cheap.data import CheapData, Config

KINDS = ("a", "ab0", "abz")  # a(z) | a(z) + b0*s | a(z) + b(z)*s


# ---------------------------------------------------------------------- features

def blocks(data: CheapData) -> tuple[np.ndarray, list[str]]:
    """(N, P) summary features + fundamentals + context for every panel row; computed once."""
    if "blocks" not in data._memo:
        p = data.panel
        feats = summary_features(p)
        x = np.concatenate((feats, p.fundamentals.astype(np.float64), p.context.astype(np.float64)), axis=1)
        names = summary_feature_names(p.history.shape[2]) + list(data.fund_names) + list(data.context_names)
        data._memo["blocks"] = (x, names)
    return data._memo["blocks"]


def state_features(data: CheapData) -> pd.DataFrame:
    """Named per-row state variables used by the interaction tests (R3, R4, R6, R11)."""
    if "state" not in data._memo:
        from fedcore.q3.contracts import BETA63

        p = data.panel
        out = {
            "rvol21": last_observed(p.history[:, :, 2], p.history_mask),
            "beta63": last_observed(p.history[:, :, BETA63], p.history_mask),
        }
        for k, name in enumerate(data.fund_names):
            out[name] = p.fundamentals[:, k].astype(np.float64)
        data._memo["state"] = pd.DataFrame(out)
    return data._memo["state"]


class Standardizer:
    """Median impute, clip at median ± CLIP*IQR, z-score. Fit on training rows only."""

    def fit(self, x: np.ndarray) -> "Standardizer":
        x = np.asarray(x, dtype=np.float64)
        self.median_, iqr = _columns(x, np.isfinite(x))
        self.lo_, self.hi_ = self.median_ - CLIP * iqr, self.median_ + CLIP * iqr
        z = self._clip(x)
        self.mean_ = z.mean(axis=0)
        sd = z.std(axis=0)
        self.sd_ = np.where(sd < 1e-8, 1.0, sd)
        return self

    def _clip(self, x: np.ndarray) -> np.ndarray:
        x = np.array(x, dtype=np.float64, copy=True)
        bad = ~np.isfinite(x)
        x[bad] = np.broadcast_to(self.median_, x.shape)[bad]
        return np.clip(x, self.lo_, self.hi_)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (self._clip(x) - self.mean_) / self.sd_


def ridge_fit_predict(kind: str, z: np.ndarray, s: np.ndarray, y: np.ndarray, train: np.ndarray,
                      test: np.ndarray, alpha: float = 1.0, offset: np.ndarray | None = None) -> dict:
    """Fit one of KINDS on ``train`` rows and predict ``test`` rows: mu, a, b on test.

    ``offset`` (optional, per row) is a known shock loading subtracted before fitting and
    added back: the model then learns a correction to ``offset * s`` (used by S4).
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    std = Standardizer().fit(z[train])
    zt, ze = std.transform(z[train]), std.transform(z[test])
    st, se = s[train], s[test]
    off_t = 0.0 if offset is None else offset[train] * st
    off_e = 0.0 if offset is None else offset[test] * se

    def design(zz, ss):
        if kind == "a":
            return zz
        if kind == "ab0":
            return np.column_stack((zz, ss))
        return np.column_stack((zz, ss[:, None] * zz, ss))

    model = Ridge(alpha=alpha).fit(design(zt, st), y[train] - off_t)
    coef, p = model.coef_, zt.shape[1]
    a = model.intercept_ + ze @ coef[:p]
    if kind == "a":
        b = np.zeros(len(se))
    elif kind == "ab0":
        b = np.full(len(se), coef[p])
    else:
        b = ze @ coef[p : 2 * p] + coef[2 * p]
    if offset is not None:
        b = b + offset[test]
    return {"mu": a + b * se, "a": a, "b": b}


# ---------------------------------------------------------------------- splits

def dev_meetings(data: CheapData, cfg: Config) -> pd.DatetimeIndex:
    return data.meetings()[: cfg.dev_meetings]


def inner_folds(data: CheapData, cfg: Config):
    return walk_forward(dev_meetings(data, cfg), cfg.inner_min_train, cfg.inner_test)


def outer_folds(data: CheapData, cfg: Config):
    return walk_forward(data.meetings(), cfg.outer_min_train, cfg.outer_test)


def rows_of(data: CheapData, meetings) -> np.ndarray:
    own = np.asarray(data.panel.meeting, dtype="datetime64[ns]")
    return np.flatnonzero(np.isin(own, pd.DatetimeIndex(meetings).to_numpy(dtype="datetime64[ns]")))


# ---------------------------------------------------------------------- scoring

def meeting_mse(meeting: np.ndarray, y: np.ndarray, mu: np.ndarray, centered: bool = False) -> pd.Series:
    """Per-meeting MSE (or within-meeting centered MSE), indexed by meeting date."""
    df = pd.DataFrame({"m": pd.to_datetime(meeting), "y": y, "mu": mu})
    if centered:
        g = df.groupby("m")
        df["y"] = df["y"] - g["y"].transform("mean")
        df["mu"] = df["mu"] - g["mu"].transform("mean")
    return ((df["y"] - df["mu"]) ** 2).groupby(df["m"]).mean()


def boot_mean(values, n: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    """Bootstrap the mean of one value per meeting (equivalent to ``meeting_bootstrap`` with
    one row per meeting, vectorized)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"estimate": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_meetings": 0}
    rng = np.random.default_rng(seed)
    draws = v[rng.integers(0, v.size, size=(n, v.size))].mean(axis=1)
    a = (1 - level) / 2
    return {
        "estimate": float(v.mean()), "se": float(draws.std(ddof=1)),
        "ci_low": float(np.quantile(draws, a)), "ci_high": float(np.quantile(draws, 1 - a)),
        "n_meetings": int(v.size),
    }


def paired(left: pd.Series, right: pd.Series, cfg: Config) -> dict:
    """Mean per-meeting difference left - right with a bootstrap CI. Negative: left is better."""
    both = left.index.intersection(right.index)
    return boot_mean((left.loc[both] - right.loc[both]).to_numpy(), cfg.n_boot, cfg.seed)


# ---------------------------------------------------------------------- within-meeting OLS

def demean(groups: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Subtract the group (meeting) mean from each column."""
    x = np.asarray(x, dtype=np.float64)
    two = x.ndim == 2
    df = pd.DataFrame(x if two else x[:, None])
    out = df - df.groupby(pd.factorize(groups)[0]).transform("mean")
    return out.to_numpy() if two else out.to_numpy()[:, 0]


def ols_cluster(x: np.ndarray, y: np.ndarray, groups: np.ndarray, intercept: bool = False) -> dict:
    """OLS with meeting-clustered (CR1) standard errors. Returns coef, se, t, vcov, n, n_groups."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if intercept:
        x = np.column_stack((np.ones(len(x)), x))
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
    x, y, g = x[ok], y[ok], np.asarray(groups)[ok]
    n, k = x.shape
    xtx_inv = np.linalg.pinv(x.T @ x)
    coef = xtx_inv @ x.T @ y
    e = y - x @ coef
    codes, uniq = pd.factorize(g)
    scores = np.zeros((len(uniq), k))
    np.add.at(scores, codes, x * e[:, None])
    meat = scores.T @ scores
    c = len(uniq)
    adj = (c / max(c - 1, 1)) * ((n - 1) / max(n - k, 1))
    vcov = adj * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.clip(np.diag(vcov), 0.0, None))
    return {"coef": coef, "se": se, "t": coef / np.where(se > 0, se, np.nan), "vcov": vcov,
            "n": int(n), "n_groups": int(c), "resid": e, "rows": np.flatnonzero(ok)}


def wald_p(coef: np.ndarray, vcov: np.ndarray) -> float:
    """p-value of the joint test that every coefficient is zero."""
    from scipy.stats import chi2

    stat = float(coef @ np.linalg.pinv(vcov) @ coef)
    return float(chi2.sf(stat, df=len(coef)))
