"""Q3 v3: a two-beta CAPM base plus a restricted correction (docs/encoder/q3-v3-roadmap_a.md).

    mu = mu_base + P_m(delta_a) + P_m(delta_b) * s

``mu_base`` is ``CAPMBetas(252, both=True)``. ``P_m`` removes, within each meeting, the span of a
basis built from pre-event data: ``"centered"`` removes the meeting mean only, ``"beta"`` removes
the span of ``[1, beta63, beta_smooth]`` (rank-aware, exact). For a fully observed meeting,
raw-return MSE then equals the projected-residual MSE plus a constant, so no target purification
is needed.

Fixes from docs/reports/encoder-review-2026-10-17.md:

1. The base is fit on the inner-training block only; inner-validation outcomes never touch it.
   The deployed base stays that inner-training fit (declared: no refit).
2. Training, early stopping and prediction all score the deployed prediction (base + projected
   correction); there is no auxiliary meeting-mean term.
3. One shock (the panel's STMT surprise), so ``a_hat`` is pre-event drift and ``b_hat`` the
   loading on s; no realized announcement-window move is an input.
4. No macro tables (nothing is zero-filled); every member's selected epoch is kept.

Arms (``ARMS``): A0 two-beta CAPM, A1 summary-feature MLP projected, A2 transformer centered,
A3 transformer projected. CPU ridge corrections (``ResidualRidge``): R0 equal weights, R1 a
5-year recency half-life, R2 with within-meeting ranks.

    python -m fedenc.v3 run --arms A1 A2 A3 --device cuda     # training: needs the user's go-ahead
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from fedcore.q3.contracts import BETA63, Q3Panel, SummaryEncoder
from fedcore.q3.train import (_columns, _meeting_index_groups, _meeting_rows, _scale, resolve_device,
                              seed_everything, validation_meetings)
from fedenc.decouple import _chunks, _segment_mean, _segments
from fedenc.encoder_transformer import PatchTransformerEncoder
from fedenc.v2 import CAPMBetas, V2StructModel

BASES = ("centered", "beta")
HALF_LIFE_YEARS = 5.0


# ----------------------------------------------------------------------------- projection

def base_betas(panel: Q3Panel) -> tuple[np.ndarray, np.ndarray]:
    """The two betas ``CAPMBetas(252, both=True)`` uses: last 63-day and the 252-day average."""
    b63, bw = CAPMBetas(252, both=True)._betas(panel)
    return np.asarray(b63, dtype=np.float64), np.asarray(bw, dtype=np.float64)


def basis_rows(panel: Q3Panel, kind: str, tol: float = 1e-8) -> np.ndarray:
    """(n, 3) per-row slices of each meeting's orthonormal basis Q_m (zero-padded columns).

    ``P_m v = v - Q_m (Q_m' v)``. ``kind="centered"``: Q_m = 1/sqrt(n_m). ``kind="beta"``: an
    orthonormal basis of span[1, beta63, beta_smooth] by SVD, dropping directions with singular
    value below ``tol`` times the largest (nearly collinear betas)."""
    if kind not in BASES:
        raise ValueError(f"kind must be one of {BASES}")
    n = len(panel.target)
    out = np.zeros((n, 3), dtype=np.float64)
    b63, bw = base_betas(panel)
    for rows in _meeting_index_groups(panel.meeting):
        ones = np.ones(len(rows))
        if kind == "centered":
            out[rows, 0] = ones / math.sqrt(len(rows))
            continue
        B = np.column_stack((ones, b63[rows], bw[rows]))
        U, S, _ = np.linalg.svd(B, full_matrices=False)
        keep = S > tol * S.max()
        out[rows, : int(keep.sum())] = U[:, keep]
    return out


def project(v: torch.Tensor, q: torch.Tensor, inverse: torch.Tensor, width: int) -> torch.Tensor:
    """P_m v for rows of whole meetings: v (r,), q (r, 3) basis rows, ``inverse`` meeting codes."""
    qtv = torch.zeros(width, q.shape[1], dtype=v.dtype, device=v.device).index_add_(0, inverse, q * v[:, None])
    return v - (q * qtv[inverse]).sum(dim=1)


# ----------------------------------------------------------------------------- network arm

class _Scaler:
    """Median/IQR scaling of history, fundamentals and context, fit on the inner-training rows."""

    def __init__(self, panel: Q3Panel):
        self.h = _columns(panel.history, panel.history_mask)
        self.f = _columns(panel.fundamentals, panel.fundamentals_mask)
        self.c = _columns(panel.context, np.ones(panel.context.shape, dtype=bool))

    def tensors(self, panel: Q3Panel, device) -> dict:
        h = np.where(panel.history_mask[..., None], _scale(panel.history, *self.h), np.float32(0.0))
        f = np.where(panel.fundamentals_mask, _scale(panel.fundamentals, *self.f), np.float32(0.0))
        c = _scale(panel.context, *self.c)
        t = lambda x, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(x), dtype=dt, device=device)
        return dict(history=t(h), history_mask=t(panel.history_mask, torch.bool), fundamentals=t(f),
                    fundamentals_mask=t(panel.fundamentals_mask), context=t(c))


def _make_model(encoder: nn.Module, panel: Q3Panel) -> V2StructModel:
    model = V2StructModel(encoder, n_fund=panel.fundamentals.shape[1], n_ctx=panel.context.shape[1], n_shocks=1)
    for head in (model.head_a, model.head_b):  # zero correction at the start: predicts the base exactly
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
    return model


def _correction(model, x: dict, rows: torch.Tensor, q: torch.Tensor, codes: torch.Tensor, s: torch.Tensor):
    """Projected (delta_a, delta_b) for ``rows`` (whole meetings)."""
    a_raw, b_mat = model(*(x[k][rows] for k in ("history", "history_mask", "fundamentals",
                                                 "fundamentals_mask", "context")))
    inverse, width, _ = _segments(codes[rows])
    da = project(a_raw.double(), q[rows], inverse, width)
    db = project(b_mat[:, 0].double(), q[rows], inverse, width)
    return da, db, inverse, width


def _deployed_loss(model, x, rows, q, codes, s, e):
    """Equal-meeting mean of the deployed squared error on residual ``e`` (= y - mu_base)."""
    da, db, inverse, width = _correction(model, x, rows, q, codes, s)
    r = e[rows] - da - db * s[rows]
    _, _, count = _segments(codes[rows])
    return _segment_mean(r ** 2, inverse, width, count).mean()


class V3Arm:
    """Two-beta CAPM base (inner-training fit) + an ensemble of projected corrections."""

    def __init__(self, seed: int = 0, *, encoder: str = "transformer", basis: str = "beta", members: int = 3,
                 lr: float = 1e-3, weight_decay: float = 1e-4, max_epochs: int = 40, patience: int = 10,
                 meetings_per_batch: int = 4, val_frac: float = 0.15, device: str | None = None, encoder_fn=None):
        if basis not in BASES:
            raise ValueError(f"basis must be one of {BASES}")
        self.seed, self.encoder, self.basis, self.members = int(seed), encoder, basis, int(members)
        self.lr, self.weight_decay, self.max_epochs, self.patience = lr, weight_decay, max_epochs, patience
        self.meetings_per_batch, self.val_frac, self.device, self.encoder_fn = meetings_per_batch, val_frac, device, encoder_fn
        self.fit_result = None

    def config(self) -> dict:
        return dict(encoder=self.encoder, basis=self.basis, members=self.members, lr=self.lr,
                    weight_decay=self.weight_decay, max_epochs=self.max_epochs, patience=self.patience,
                    meetings_per_batch=self.meetings_per_batch, val_frac=self.val_frac, base="capm_beta_both")

    def _encoder(self, n_channels):
        if self.encoder_fn is not None:
            return self.encoder_fn(n_channels)
        if self.encoder == "summary":
            return SummaryEncoder(n_channels=n_channels, out_dim=32)
        return PatchTransformerEncoder(n_channels=n_channels)

    def fit(self, panel: Q3Panel, seed: int | None = None, **_) -> None:
        seed = self.seed if seed is None else int(seed)
        val_m = validation_meetings(panel, self.val_frac)
        tr_idx = _meeting_rows(panel, panel.meetings()[: -len(val_m)])
        va_idx = _meeting_rows(panel, val_m)
        train, val = panel.subset(tr_idx), panel.subset(va_idx)
        self.base = CAPMBetas(252, both=True)
        self.base.fit(train)  # review fix 1: inner validation never touches the base
        self.scaler = _Scaler(train)
        device = resolve_device(self.device)
        data = {name: self._data(p, device) for name, p in (("train", train), ("val", val))}
        groups = {name: _meeting_index_groups(p.meeting) for name, p in (("train", train), ("val", val))}
        self.nets, records = [], []
        for j in range(self.members):
            member_seed = seed * 100 + j
            seed_everything(member_seed)
            model = _make_model(self._encoder(panel.history.shape[-1]), panel).to(device)
            records.append(self._train(model, data, groups, member_seed))
            self.nets.append(model)
        first = records[0]
        self.fit_result = argparse.Namespace(
            train_loss=first["train"], val_loss=first["val"], best_epoch=first["best_epoch"],
            best_val_loss=first["best"], val_meetings=pd.DatetimeIndex(val_m),
            member_best_epochs=[r["best_epoch"] for r in records], member_best_val=[r["best"] for r in records])

    def _data(self, p: Q3Panel, device) -> dict:
        x = self.scaler.tensors(p, device)
        mu_base, _, _ = self.base.predict(p)
        t = lambda v, dt=torch.float64: torch.as_tensor(np.asarray(v), dtype=dt, device=device)
        return dict(x=x, q=t(basis_rows(p, self.basis)), codes=t(pd.factorize(pd.DatetimeIndex(p.meeting))[0], torch.long),
                    s=t(p.shock), e=t(np.asarray(p.target, dtype=np.float64) - mu_base))

    def _evaluate(self, model, d, groups) -> float:
        model.eval()
        with torch.no_grad():
            parts = [_deployed_loss(model, d["x"], torch.as_tensor(idx, device=d["e"].device), d["q"], d["codes"],
                                    d["s"], d["e"]) * len(np.unique(d["codes"][idx].cpu().numpy()))
                     for idx in _chunks(groups)]
        model.train()
        return float(sum(parts).item() / len(groups))  # equal weight per meeting

    def _train(self, model, data, groups, seed) -> dict:
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        rng = np.random.default_rng(seed)
        tr, g = data["train"], groups["train"]
        best, best_epoch, stale = self._evaluate(model, data["val"], groups["val"]), 0, 0  # epoch 0 = the base
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        curve_tr, curve_va = [], []
        for epoch in range(1, self.max_epochs + 1):
            model.train()
            order = rng.permutation(len(g))
            for start in range(0, len(order), self.meetings_per_batch):
                rows = torch.as_tensor(np.concatenate([g[k] for k in order[start:start + self.meetings_per_batch]]),
                                       device=tr["e"].device)
                opt.zero_grad(set_to_none=True)
                loss = _deployed_loss(model, tr["x"], rows, tr["q"], tr["codes"], tr["s"], tr["e"])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            curve_tr.append(self._evaluate(model, tr, g))
            curve_va.append(self._evaluate(model, data["val"], groups["val"]))
            if curve_va[-1] < best:
                best, best_epoch, stale = curve_va[-1], epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        model.load_state_dict(best_state)
        model.eval()
        return dict(train=curve_tr, val=curve_va, best=best, best_epoch=best_epoch)

    def corrections(self, panel: Q3Panel) -> tuple[np.ndarray, np.ndarray]:
        """Member-averaged projected (delta_a, delta_b)."""
        device = next(self.nets[0].parameters()).device
        d = self._data(panel, device)
        da, db = np.zeros(len(panel.target)), np.zeros(len(panel.target))
        with torch.no_grad():
            for model in self.nets:
                for idx in _chunks(_meeting_index_groups(panel.meeting)):
                    rows = torch.as_tensor(idx, device=device)
                    a, b, _, _ = _correction(model, d["x"], rows, d["q"], d["codes"], d["s"])
                    da[idx] += a.cpu().numpy() / len(self.nets)
                    db[idx] += b.cpu().numpy() / len(self.nets)
        return da, db

    def predict(self, panel: Q3Panel):
        _, a0, b0 = self.base.predict(panel)
        da, db = self.corrections(panel)
        a, b = a0 + da, b0 + db
        return a + b * np.asarray(panel.shock, dtype=np.float64), a, b


# ----------------------------------------------------------------------------- CPU ridge corrections

def recency_weights(meetings: pd.DatetimeIndex, cutoff: pd.Timestamp, half_life: float = HALF_LIFE_YEARS):
    """2^(-years before ``cutoff`` / half_life), normalized to mean 1 over the given meetings."""
    years = (pd.Timestamp(cutoff) - pd.DatetimeIndex(meetings)).days / 365.25
    w = np.power(2.0, -np.asarray(years, dtype=np.float64) / half_life)
    return w / w.mean()


class ResidualRidge:
    """Two-beta CAPM + a ridge correction on firm features, centered within each meeting.

    Correction: ``Zc w_a + (Zc * s) w_b`` with Z the summary features and fundamentals
    (median-imputed, clipped, z-scored on training rows) and Zc Z minus its meeting mean.
    ``ranks`` appends within-meeting percentile ranks of every column (pre-event values only).
    ``recency`` weights meetings by a 5-year half-life from the training cutoff, in the base and
    the ridge alike. The penalty is chosen on the last ``val_frac`` of training meetings with the
    base fit on the rest, then base and ridge are refit on all training meetings (declared)."""

    ALPHAS = tuple(10.0 ** np.arange(0, 7.5, 0.5))

    def __init__(self, ranks: bool = False, recency: bool = False, val_frac: float = 0.15):
        self.ranks, self.recency, self.val_frac = ranks, recency, val_frac
        self.fit_result = None

    def _features(self, panel: Q3Panel, fit: bool) -> np.ndarray:
        from fedcore.q3.features import summary_features

        Z = np.column_stack((summary_features(panel), np.asarray(panel.fundamentals, dtype=np.float64)))
        if fit:
            with np.errstate(all="ignore"):
                self.med_ = np.where(np.isfinite(np.nanmedian(Z, axis=0)), np.nanmedian(Z, axis=0), 0.0)
                q75, q25 = np.nanpercentile(Z, [75, 25], axis=0)
            iqr = np.nan_to_num(q75 - q25, nan=1.0)
            self.iqr_ = np.where(iqr > 1e-12, iqr, 1.0)
        Z = np.where(np.isfinite(Z), Z, self.med_)
        Z = np.clip((Z - self.med_) / self.iqr_, -5.0, 5.0)
        codes = pd.factorize(pd.DatetimeIndex(panel.meeting))[0]
        if self.ranks:
            R = pd.DataFrame(Z).groupby(codes).rank(pct=True).to_numpy() - 0.5
            Z = np.column_stack((Z, R))
        Zc = Z - pd.DataFrame(Z).groupby(codes).transform("mean").to_numpy()
        s = np.asarray(panel.shock, dtype=np.float64)[:, None]
        return np.column_stack((Zc, Zc * s))

    def _weights(self, panel: Q3Panel, cutoff) -> np.ndarray:
        if not self.recency:
            return np.ones(len(panel.target))
        m = pd.DatetimeIndex(panel.meeting)
        um = panel.meetings()
        per = pd.Series(recency_weights(um, cutoff), index=um)
        return per.reindex(m).to_numpy()

    @staticmethod
    def _wls_base(panel: Q3Panel, w: np.ndarray) -> CAPMBetas:
        base = CAPMBetas(252, both=True)
        X, _ = base._design(panel)
        sw = np.sqrt(w)[:, None]
        base.coef_, *_ = np.linalg.lstsq(X * sw, np.asarray(panel.target, dtype=np.float64) * sw[:, 0], rcond=None)
        return base

    def _ridge(self, F, e, w, alpha):
        Fw = F * w[:, None]
        sd = np.where(F.std(axis=0) > 1e-12, F.std(axis=0), 1.0)
        A = (F / sd).T @ (Fw / sd) + alpha * np.eye(F.shape[1])
        return np.linalg.solve(A, (Fw / sd).T @ e) / sd

    def fit(self, panel: Q3Panel, **kwargs) -> None:
        del kwargs
        cutoff = panel.meetings()[-1]
        val_m = validation_meetings(panel, self.val_frac)
        inner = panel.subset(_meeting_rows(panel, panel.meetings()[: -len(val_m)]))
        val = panel.subset(_meeting_rows(panel, val_m))
        w_in = self._weights(inner, inner.meetings()[-1])
        base = self._wls_base(inner, w_in)
        F_in = self._features(inner, fit=True)
        e_in = np.asarray(inner.target, dtype=np.float64) - base.predict(inner)[0]
        F_va = self._features(val, fit=False)
        e_va = np.asarray(val.target, dtype=np.float64) - base.predict(val)[0]
        codes = pd.factorize(pd.DatetimeIndex(val.meeting))[0]

        def val_mse(c):  # equal-meeting MSE on the inner validation block
            r = (e_va - F_va @ c) ** 2
            return pd.Series(r).groupby(codes).mean().mean()

        scores = [val_mse(self._ridge(F_in, e_in, w_in, al)) for al in self.ALPHAS]
        self.alpha_ = float(self.ALPHAS[int(np.argmin(scores))])
        w = self._weights(panel, cutoff)
        self.base = self._wls_base(panel, w)
        F = self._features(panel, fit=True)
        e = np.asarray(panel.target, dtype=np.float64) - self.base.predict(panel)[0]
        self.coef_ = self._ridge(F, e, w, self.alpha_)

    def predict(self, panel: Q3Panel):
        _, a0, b0 = self.base.predict(panel)
        F = self._features(panel, fit=False)
        k = F.shape[1] // 2
        s = np.asarray(panel.shock, dtype=np.float64)
        a = a0 + F[:, :k] @ self.coef_[:k]
        # F[:, k:] is Zc * s, so the b part is Zc @ w_b
        b = b0 + (F[:, :k] @ self.coef_[k:])
        return a + b * s, a, b


# ----------------------------------------------------------------------------- arms and runner

def makers(device=None, members=3) -> dict:
    from fedcore.q3.baselines import BASELINES

    return {
        "A0": lambda seed: CAPMBetas(252, both=True),
        "A1": lambda seed: V3Arm(seed, encoder="summary", basis="beta", members=members, device=device),
        "A2": lambda seed: V3Arm(seed, encoder="transformer", basis="centered", members=members, device=device),
        "A3": lambda seed: V3Arm(seed, encoder="transformer", basis="beta", members=members, device=device),
        "R0": lambda seed: ResidualRidge(),
        "R1": lambda seed: ResidualRidge(recency=True),
        "R2": lambda seed: ResidualRidge(ranks=True),
        "capm_pooled": BASELINES["capm_pooled"],
    }


def run(arms, through="2021-12-31", seeds=(0, 1, 2), members=3, min_train=60, test_size=16, device=None, store=None):
    """Walk-forward on development meetings (same 161 test meetings as v2). Training for A1-A3."""
    from fedcore.q3.real import UNIVERSES
    from fedcore.results import ResultStore
    from fedenc.config import RESULTS
    from fedenc.runs import save_runs
    from fedenc.workflow import bounded_panel

    if pd.Timestamp(through) > pd.Timestamp("2021-12-31"):
        raise ValueError("v3 development runs stop at 2021-12-31")
    panel, through_ts = bounded_panel(through)
    every = makers(device=device, members=members)
    chosen = {a: every[a] for a in arms}
    configs = {a: (m(0).config() if hasattr(m(0), "config") else {}) for a, m in chosen.items()}
    neural = {"A1", "A2", "A3"}
    return save_runs(panel, chosen, prefix=f"v3-to{through_ts.year}",
                     params=dict(through=str(through_ts.date()), configs=configs,
                                 plan="docs/encoder/q3-v3-results.md#pre-stated-criteria"),
                     min_train=min_train, test_size=test_size,
                     seeds=seeds if neural & set(arms) else (0,), store=store or ResultStore(RESULTS),
                     datasets=UNIVERSES["sp500"].inputs)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arms", nargs="+", required=True, choices=list(makers()))
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    r.add_argument("--members", type=int, default=3)
    r.add_argument("--device")
    args = p.parse_args(argv)
    run(args.arms, seeds=tuple(args.seeds), members=args.members, device=args.device)


if __name__ == "__main__":
    main()
