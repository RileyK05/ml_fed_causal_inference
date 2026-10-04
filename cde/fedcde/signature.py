"""Truncated path signatures in numpy (depth 3), with an optional lead-lag transform.

``iisignature`` does not build on Windows, so the signature is implemented here. The
depth-2 terms are the increments plus the Levy areas; a unit test in
``cde/tests/test_cde.py`` checks them against a hand-computed 2-D example.

Missing days are NaN in the path. Increments touching a NaN are set to zero, which is
the "hold flat across a gap" convention used everywhere in this repo. The signature
therefore depends only on observed values.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from fedcore.q3.contracts import Q3Panel
from fedcore.q3.train import validation_meetings

from fedcde.encoder_cde import build_path


def _increments(path: np.ndarray) -> np.ndarray:
    """First differences along time, prepended with a zero row. NaN -> 0. (N, T, C)."""
    path = np.asarray(path)
    if path.ndim != 3:
        raise ValueError(f"path must be (N, T, C), got {path.shape}")
    d = np.diff(path, axis=1, prepend=path[:, :1, :])
    return np.where(np.isfinite(d), d, 0.0)


def _lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """Interleave each channel with its lag: (N, T, C) -> (N, T, 2C).

    The lag of the first time step is the value itself, so the first increment of the
    lagged channel is zero.
    """
    path = np.asarray(path)
    n, t, c = path.shape
    lagged = np.empty_like(path)
    lagged[:, 0, :] = path[:, 0, :]
    lagged[:, 1:, :] = path[:, :-1, :]
    return np.stack((path, lagged), axis=-1).reshape(n, t, 2 * c)


def truncated_signature(
    path: np.ndarray,
    depth: int = 3,
    *,
    lead_lag: bool = False,
    chunk: int = 4096,
) -> np.ndarray:
    """Depth-``depth`` truncated signature of a batch of paths.

    ``path`` is (N, T, C); returns (N, C + C^2 + ... + C^depth) float64. Terms are
    ordered depth by depth, and within a depth by the channel index tuple in
    C-order (e.g. depth 2 is (i, j) with j fastest).
    """
    if depth < 1:
        raise ValueError("depth must be at least 1")
    path = np.asarray(path, dtype=np.float64)
    if lead_lag:
        path = _lead_lag_transform(path)
    n, t, c = path.shape
    blocks: list[np.ndarray] = []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        blocks.append(_signature_block(path[start:stop], t, c, depth))
    return np.concatenate(blocks, axis=0)


def _signature_block(path: np.ndarray, t: int, c: int, depth: int) -> np.ndarray:
    d = _increments(path)  # (B, T, C)
    b = d.shape[0]
    terms: list[np.ndarray] = []

    # Inclusive cumsum: A_t = sum_{u<=t} d_u. Exclusive: A_prev_t = A_{t-1} = A_t - d_t.
    # The signature integrates the *previous* level's value at u-1 against d_u, so every
    # level below uses the exclusive cumsum of the level above (strict s < t).
    a = np.cumsum(d, axis=1)
    a_prev = a - d  # (B, T, C); a_prev[:, 0, :] = 0

    # Depth 1: total displacement.
    terms.append(a[:, -1, :])

    if depth >= 2:
        # S2[b, t, i, j] = sum_{u<=t} a_prev[b, u, i] d[b, u, j]
        s2 = np.empty((b, t, c, c), dtype=np.float64)
        for j in range(c):
            s2[:, :, :, j] = np.cumsum(a_prev * d[:, :, j : j + 1], axis=1)
        terms.append(s2[:, -1, :, :].reshape(b, c * c))

    if depth >= 3:
        # S2_prev[b, t, i, j] = S2[b, t, i, j] - a_prev[b, t, i] d[b, t, j]
        s2_prev = s2 - a_prev[:, :, :, None] * d[:, :, None, :]
        # S3[b, i, j, k] = sum_t S2_prev[b, t, i, j] d[b, t, k]
        s3 = np.einsum("btij,btk->bijk", s2_prev, d, optimize=True)
        terms.append(s3.reshape(b, c * c * c))

    return np.concatenate(terms, axis=1)


def signature_n_channels(n_channels: int, depth: int = 3, *, lead_lag: bool = False) -> int:
    """Number of signature terms for a given channel count."""
    c = 2 * n_channels if lead_lag else n_channels
    return sum(c**d for d in range(1, depth + 1))


ALPHA_GRID = (0.1, 1.0, 10.0, 100.0)


def _median_impute(values: np.ndarray, median: np.ndarray) -> np.ndarray:
    out = np.array(values, dtype=np.float64, copy=True)
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = np.broadcast_to(median, out.shape)[bad]
    return out


def _meeting_mse(mu: np.ndarray, target: np.ndarray, meeting: np.ndarray) -> float:
    """Mean over meetings of within-meeting MSE (the trainer's loss)."""
    values = []
    for day in np.unique(meeting):
        sel = meeting == day
        values.append(float(np.mean((mu[sel] - target[sel]) ** 2)))
    return float(np.mean(values))


class SigRidge:
    """Ridge on [sig, fundamentals, context, s*sig, s*fundamentals, s*context, s].

    The signature block is the depth-3 truncated signature of the path over the full
    window and over the last 63 days. ``b_hat`` is the shock-coefficient part, the same
    construction as ``ridge_interact``. The ridge alpha is chosen on the trainer's
    validation slice (last 15 percent of the training meetings) before refitting on the
    full training panel.
    """

    def __init__(
        self,
        seed: int = 0,
        alpha_grid: tuple[float, ...] = ALPHA_GRID,
        window: int = 252,
        short_window: int = 63,
        lead_lag: bool = False,
    ):
        self.seed = int(seed)
        self.alpha_grid = tuple(float(a) for a in alpha_grid)
        self.window = int(window)
        self.short_window = int(short_window)
        self.lead_lag = bool(lead_lag)
        self.fit_result = None

    def _sig_features(self, panel: Q3Panel) -> np.ndarray:
        import torch

        x = torch.from_numpy(np.ascontiguousarray(panel.history)).float()
        mask = torch.from_numpy(panel.history_mask)
        path = build_path(x, mask, window=self.window)
        full = truncated_signature(path.numpy(), depth=3, lead_lag=self.lead_lag)
        short = truncated_signature(path.numpy()[:, -self.short_window :, :], depth=3, lead_lag=self.lead_lag)
        return np.concatenate((full, short), axis=1)

    def _blocks(self, panel: Q3Panel) -> np.ndarray:
        sig = self._sig_features(panel)
        fund = np.asarray(panel.fundamentals, dtype=np.float64)
        ctx = np.asarray(panel.context, dtype=np.float64)
        return np.concatenate((sig, fund, ctx), axis=1)

    def fit(self, panel: Q3Panel, **kwargs) -> None:
        del kwargs
        meetings = panel.meetings()
        val_m = validation_meetings(panel, 0.15)
        n_val = len(val_m)
        train_m = meetings[:-n_val]

        def rows(sel):
            return np.flatnonzero(np.isin(np.asarray(panel.meeting, dtype="datetime64[ns]"), sel))

        sub_train = panel.subset(rows(train_m.to_numpy()))
        val = panel.subset(rows(val_m.to_numpy()))

        z_sub = self._blocks(sub_train)
        z_val = self._blocks(val)
        self.median_ = np.nanmedian(z_sub, axis=0)
        self.median_ = np.where(np.isfinite(self.median_), self.median_, 0.0)
        z_sub = _median_impute(z_sub, self.median_)
        z_val = _median_impute(z_val, self.median_)
        self.mean_ = z_sub.mean(axis=0)
        std = z_sub.std(axis=0)
        self.std_ = np.where(std < 1e-8, 1.0, std)
        z_sub = (z_sub - self.mean_) / self.std_
        z_val = (z_val - self.mean_) / self.std_

        s_sub = np.asarray(sub_train.shock, dtype=np.float64)
        s_val = np.asarray(val.shock, dtype=np.float64)
        y_sub = np.asarray(sub_train.target, dtype=np.float64)
        y_val = np.asarray(val.target, dtype=np.float64)

        best_alpha, best_loss = self.alpha_grid[0], np.inf
        for alpha in self.alpha_grid:
            model = Ridge(alpha=alpha, random_state=self.seed)
            model.fit(np.column_stack((z_sub, s_sub[:, None] * z_sub, s_sub)), y_sub)
            mu_val = model.predict(np.column_stack((z_val, s_val[:, None] * z_val, s_val)))
            loss = _meeting_mse(mu_val, y_val, np.asarray(val.meeting, dtype="datetime64[ns]"))
            if loss < best_loss:
                best_alpha, best_loss = alpha, loss

        z_full = _median_impute(self._blocks(panel), self.median_)
        z_full = (z_full - self.mean_) / self.std_
        s_full = np.asarray(panel.shock, dtype=np.float64)
        self.alpha_ = best_alpha
        self.model_ = Ridge(alpha=best_alpha, random_state=self.seed)
        self.model_.fit(np.column_stack((z_full, s_full[:, None] * z_full, s_full)), np.asarray(panel.target, dtype=np.float64))
        self.n_z_ = z_full.shape[1]
        return self

    def predict(self, panel: Q3Panel):
        z = _median_impute(self._blocks(panel), self.median_)
        z = (z - self.mean_) / self.std_
        s = np.asarray(panel.shock, dtype=np.float64)
        x = np.column_stack((z, s[:, None] * z, s))
        mu = np.asarray(self.model_.predict(x), dtype=np.float64)
        coef = np.asarray(self.model_.coef_, dtype=np.float64)
        n = self.n_z_
        b = z @ coef[n : 2 * n] + coef[2 * n]
        a = mu - b * s
        return mu, a, b
