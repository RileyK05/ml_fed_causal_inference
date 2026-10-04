"""Tabular Q3 baselines. Each exposes ``fit(panel, **kwargs)`` and ``predict(panel) -> (mu, a, b)``.

``pooled`` is one intercept and one shock loading.
``ridge_interact`` is ridge on features, fundamentals, context, and those columns
interacted with the shock. ``b_hat`` is the shock-coefficient part on each row.
``lgbm`` uses the same columns; ``b_hat`` is the finite difference of the
prediction at s+0.5 and s-0.5.
``summary_nn`` is ``ResponseModel(SummaryEncoder)`` through the shared trainer.

Feature medians, and the ridge z-scores, are fit on the training panel only.
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import Ridge

from fedcore.q3.contracts import Q3Panel, SummaryEncoder
from fedcore.q3.features import summary_features
from fedcore.q3.model import ResponseModel
from fedcore.q3.train import fit as train_fit
from fedcore.q3.train import predict as train_predict
from fedcore.q3.train import seed_everything

RIDGE_ALPHA = 1.0


def _blocks(panel: Q3Panel) -> np.ndarray:
    feats = summary_features(panel)
    fund = np.asarray(panel.fundamentals, dtype=np.float64)
    ctx = np.asarray(panel.context, dtype=np.float64)
    return np.concatenate((feats, fund, ctx), axis=1)


def _column_median(values: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(values, axis=0)
    return np.where(np.isfinite(med), med, 0.0).astype(np.float64)


def _impute(values: np.ndarray, median: np.ndarray) -> np.ndarray:
    out = np.array(values, dtype=np.float64, copy=True)
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = np.broadcast_to(median, out.shape)[bad]
    return out


class _Design:
    """Median-imputed [Z, s·Z, s]. Medians come from the fit panel only."""

    def fit(self, panel: Q3Panel) -> "_Design":
        self.median_ = _column_median(_blocks(panel))
        return self

    def Z(self, panel: Q3Panel) -> np.ndarray:
        return _impute(_blocks(panel), self.median_)

    def matrix(self, panel: Q3Panel, shock: np.ndarray | None = None) -> np.ndarray:
        z = self.Z(panel)
        s = np.asarray(panel.shock if shock is None else shock, dtype=np.float64).reshape(-1)
        return np.column_stack((z, s[:, None] * z, s))


class PooledOLS:
    """r = a + b·s, one a and one b for every firm and meeting."""

    def fit(self, panel: Q3Panel, **kwargs) -> None:
        del kwargs
        s = np.asarray(panel.shock, dtype=np.float64)
        y = np.asarray(panel.target, dtype=np.float64)
        design = np.column_stack((np.ones(len(s)), s))
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        self.a_ = float(coef[0])
        self.b_ = float(coef[1])
        self.fit_result = None

    def predict(self, panel: Q3Panel):
        s = np.asarray(panel.shock, dtype=np.float64)
        mu = self.a_ + self.b_ * s
        return mu, np.full(len(s), self.a_), np.full(len(s), self.b_)


class RidgeInteract:
    """Ridge on [Z, s·Z, s]. Z is median-imputed and z-scored on the training fold.

    ``a_hat + b_hat * s`` equals the prediction. ``b_hat`` collects every coefficient
    that multiplies s.
    """

    def __init__(self, seed: int = 0, alpha: float = RIDGE_ALPHA):
        self.seed = int(seed)
        self.alpha = float(alpha)
        self.fit_result = None

    def fit(self, panel: Q3Panel, **kwargs) -> None:
        seed = int(kwargs.get("seed", self.seed))
        self.design_ = _Design().fit(panel)
        z = self.design_.Z(panel)
        self.mean_ = z.mean(axis=0)
        std = z.std(axis=0)
        self.std_ = np.where(std < 1e-8, 1.0, std)
        s = np.asarray(panel.shock, dtype=np.float64)
        z_std = (z - self.mean_) / self.std_
        x = np.column_stack((z_std, s[:, None] * z_std, s))
        self.model_ = Ridge(alpha=self.alpha, random_state=seed)
        self.model_.fit(x, np.asarray(panel.target, dtype=np.float64))
        self.n_z_ = z.shape[1]

    def predict(self, panel: Q3Panel):
        z = (self.design_.Z(panel) - self.mean_) / self.std_
        s = np.asarray(panel.shock, dtype=np.float64)
        x = np.column_stack((z, s[:, None] * z, s))
        mu = np.asarray(self.model_.predict(x), dtype=np.float64)
        coef = np.asarray(self.model_.coef_, dtype=np.float64)
        n = self.n_z_
        b = z @ coef[n : 2 * n] + coef[2 * n]
        a = mu - b * s
        return mu, a, b


class LGBMBaseline:
    """LightGBM on the same columns as ridge. b_hat is a central finite difference in s."""

    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        self.fit_result = None

    def fit(self, panel: Q3Panel, **kwargs) -> None:
        seed = int(kwargs.get("seed", self.seed))
        self.design_ = _Design().fit(panel)
        x = self.design_.matrix(panel)
        self.model_ = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            deterministic=True,
            n_jobs=1,
            verbosity=-1,
        )
        self.model_.fit(x, np.asarray(panel.target, dtype=np.float64))

    def predict(self, panel: Q3Panel):
        s = np.asarray(panel.shock, dtype=np.float64)
        mu = np.asarray(self.model_.predict(self.design_.matrix(panel, s)), dtype=np.float64)
        hi = np.asarray(self.model_.predict(self.design_.matrix(panel, s + 0.5)), dtype=np.float64)
        lo = np.asarray(self.model_.predict(self.design_.matrix(panel, s - 0.5)), dtype=np.float64)
        b = hi - lo
        a = mu - b * s
        return mu, a, b


class SummaryNN:
    """The response network with SummaryEncoder in place of a sequence encoder."""

    def __init__(self, seed: int = 0, out_dim: int = 32, **train_kwargs):
        self.seed = int(seed)
        self.out_dim = int(out_dim)
        self.train_kwargs = train_kwargs
        self.model = None
        self.scaler = None
        self.fit_result = None

    def fit(self, panel: Q3Panel, **kwargs) -> None:
        kw = dict(self.train_kwargs)
        kw.update(kwargs)
        seed = int(kw.pop("seed", self.seed))
        seed_everything(seed)
        encoder = SummaryEncoder(n_channels=panel.history.shape[-1], out_dim=self.out_dim)
        self.model = ResponseModel(
            encoder,
            n_fund=panel.fundamentals.shape[1],
            n_ctx=panel.context.shape[1],
        )
        self.fit_result = train_fit(self.model, panel, seed=seed, **kw)
        self.scaler = self.fit_result.scaler

    def predict(self, panel: Q3Panel):
        if self.model is None or self.scaler is None:
            raise RuntimeError("fit SummaryNN before predict")
        return train_predict(self.model, panel, self.scaler)


BASELINES = {
    "pooled": lambda seed: PooledOLS(),
    "ridge_interact": lambda seed: RidgeInteract(seed=seed),
    "lgbm": lambda seed: LGBMBaseline(seed=seed),
    "summary_nn": lambda seed: SummaryNN(seed=seed),
}
