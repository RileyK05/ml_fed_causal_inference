"""Q3 v4: firm shock sensitivity beyond beta, scored on large surprises (docs/encoder/q3-v4-plan.md).

    mu = mu_base + delta_a + P_m(delta_b) * s

``mu_base`` is ``CAPMBetas(252, both=True)``. ``P_m`` is v3's exact within-meeting projection off
span[1, beta63, beta_smooth]. b-only arms fix ``delta_a = 0``; joint arms (``*j``) also learn a
projected ``delta_a``, as matched controls.

Two-stage fit (plan section 4). Stage 1: on the first 85% of the training window, with the base
and scalers fit there only, choose the ridge penalty or each member's stopping step on the last
15%. Stage 2: refit the base and scalers on the whole window, then the correction at that penalty
or for exactly that many optimizer steps. Predictions save ``mu_base``, ``delta_a`` and
``delta_b``; folds save every member's selected epoch and step count.

Arms: B0 two-beta CAPM; B1 ridge on all summary features and fundamentals; B2 ridge on six named
balance-sheet and state variables; B3c B2 plus own 63-day relative return and sector effects;
B3 B3c plus leave-one-out sector peer stress; B4/B4j summary MLP b-only / joint; B5/B5j transformer
b-only / joint. Every correction is projected off the beta span.

    python -m fedenc.v4 run --arms B0 B1 B2 B3c B3 --seeds 0      # CPU
    python -m fedenc.v4 run --arms B5 --device cuda                 # training: needs the user's go-ahead
    python -m fedenc.v4 score                                        # read-only scoring of saved runs
"""
from __future__ import annotations

import argparse
import hashlib
import math

import numpy as np
import pandas as pd
import torch
from torch import nn

from fedcore.q3.contracts import Q3Panel, SummaryEncoder
from fedcore.q3.train import _meeting_index_groups, _meeting_rows, resolve_device, seed_everything, validation_meetings
from fedenc.encoder_transformer import PatchTransformerEncoder
from fedenc.v2 import CAPMBetas, V2StructModel
from fedenc.v3 import _Scaler, _chunks, _correction, _deployed_loss, basis_rows

NAMED = ("cash", "leverage", "log_size", "book_to_market", "ret_rel_mean_252", "ret_rel_drawdown_252")
OWN63 = "ret_rel_mean_63"
STANDOUT = (pd.Timestamp("2004-01-28"), pd.Timestamp("2011-09-21"))
THRESHOLDS = (0.1, 0.25, 0.5)  # |s| in 10bp units: 1, 2.5, 5bp
PRIMARY = 0.25
GUARDRAIL = 0.005
ALPHAS = tuple(10.0 ** np.arange(-2, 7.5, 0.5))


def panel_fingerprint(panel: Q3Panel) -> str:
    """Hash of the arrays a run is scored on (meetings, firms, target, shock)."""
    h = hashlib.sha256()
    for arr in (np.asarray(panel.meeting, dtype="datetime64[ns]").astype(np.int64), np.asarray(panel.firm_id),
                np.asarray(panel.target, dtype=np.float32), np.asarray(panel.shock, dtype=np.float32)):
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:16]


def _codes(panel: Q3Panel) -> np.ndarray:
    return pd.factorize(pd.DatetimeIndex(panel.meeting))[0]


def project_columns(V: np.ndarray, q: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """P_m applied to every column of V (n, k): V - Q_m (Q_m' V), meeting by meeting."""
    qtv = np.zeros((codes.max() + 1, q.shape[1], V.shape[1]))
    np.add.at(qtv, codes, q[:, :, None] * V[:, None, :])
    return V - np.einsum("nr,nrk->nk", q, qtv[codes])


def _inner_split(panel: Q3Panel, val_frac: float):
    val_m = validation_meetings(panel, val_frac)
    return (panel.subset(_meeting_rows(panel, panel.meetings()[: -len(val_m)])),
            panel.subset(_meeting_rows(panel, val_m)))


# ----------------------------------------------------------------------------- ridge arms

class SectorMap:
    """GICS sector per (meeting, firm), from the panel's row metadata (as of the meeting from 2001)."""

    def __init__(self, rows: pd.DataFrame):
        self.map = pd.Series(rows["gsector"].astype("string").str[:2].to_numpy(),
                             index=pd.MultiIndex.from_arrays([pd.DatetimeIndex(rows["meeting"]),
                                                              rows["permno"].astype(int)]))

    def __call__(self, panel: Q3Panel) -> np.ndarray:
        idx = pd.MultiIndex.from_arrays([pd.DatetimeIndex(panel.meeting), np.asarray(panel.firm_id).astype(int)])
        return self.map.reindex(idx).fillna("unknown").to_numpy(dtype=object)


def peer_stress(own: np.ndarray, sector: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Leave-one-out mean of ``own`` over the firm's sector in the same meeting (NaN if alone)."""
    df = pd.DataFrame({"m": codes, "g": sector, "x": own})
    ok = df["x"].notna()
    grp = df[ok].groupby(["m", "g"])["x"]
    tot = grp.transform("sum").reindex(df.index)
    cnt = grp.transform("count").reindex(df.index)
    loo = (tot - df["x"]) / (cnt - 1)
    loo[(cnt <= 1) | ~ok] = np.nan
    return loo.to_numpy(dtype=np.float64)


class V4Ridge:
    """Two-beta base + ridge correction to b only, projected off the beta span.

    ``features="all"`` uses every summary feature and fundamental; otherwise the named columns.
    ``sector`` adds own 63-day relative return and sector dummies; ``peer`` adds the leave-one-out
    sector mean of the 63-day relative return. All enter as P_m(Z) * s."""

    def __init__(self, features="named", sector=False, peer=False, sector_map: SectorMap | None = None,
                 val_frac: float = 0.15):
        if (sector or peer) and sector_map is None:
            raise ValueError("sector effects and peer stress need a sector_map")
        self.features, self.sector, self.peer, self.sector_map, self.val_frac = features, sector, peer, sector_map, val_frac
        self.fit_result = None

    def config(self) -> dict:
        return dict(model="ridge_b_only", features=self.features, sector=self.sector, peer=self.peer,
                    val_frac=self.val_frac, base="capm_beta_both", projection="beta")

    def _raw(self, panel: Q3Panel) -> tuple[np.ndarray, list[str]]:
        from fedcore.q3.features import summary_feature_names, summary_features
        from fedcore.q3.real import FUNDAMENTAL_NAMES

        allf = pd.DataFrame(np.column_stack((summary_features(panel), np.asarray(panel.fundamentals, dtype=np.float64))),
                            columns=summary_feature_names() + list(FUNDAMENTAL_NAMES))
        cols = list(allf.columns) if self.features == "all" else list(NAMED)
        Z = allf[cols].to_numpy(dtype=np.float64)
        names = list(cols)
        if self.sector or self.peer:
            own = allf[OWN63].to_numpy(dtype=np.float64)
            if self.sector:
                Z, names = np.column_stack((Z, own)), names + [OWN63]
            if self.peer:
                Z = np.column_stack((Z, peer_stress(own, self.sector_map(panel), _codes(panel))))
                names = names + ["peer_" + OWN63]
        return Z, names

    def _design(self, panel: Q3Panel, fit: bool) -> np.ndarray:
        Z, _ = self._raw(panel)
        if fit:
            with np.errstate(all="ignore"):
                med = np.nanmedian(Z, axis=0)
                q75, q25 = np.nanpercentile(Z, [75, 25], axis=0)
            self.med_ = np.where(np.isfinite(med), med, 0.0)
            iqr = np.nan_to_num(q75 - q25, nan=1.0)
            self.iqr_ = np.where(iqr > 1e-12, iqr, 1.0)
        Z = np.clip((np.where(np.isfinite(Z), Z, self.med_) - self.med_) / self.iqr_, -5.0, 5.0)
        if self.sector:
            sec = self.sector_map(panel)
            if fit:
                self.sectors_ = sorted(set(sec) - {"unknown"})
            Z = np.column_stack((Z, np.stack([sec == g for g in self.sectors_], axis=1).astype(np.float64)))
        return project_columns(Z, basis_rows(panel, "beta"), _codes(panel))

    @staticmethod
    def _solve(PZ, s, e, alpha, sd=None):
        F = PZ * s[:, None]
        sd = np.where(F.std(axis=0) > 1e-12, F.std(axis=0), 1.0) if sd is None else sd
        Fs = F / sd
        return np.linalg.solve(Fs.T @ Fs + alpha * np.eye(F.shape[1]), Fs.T @ e) / sd

    def fit(self, panel: Q3Panel, **kwargs) -> None:
        del kwargs
        inner, val = _inner_split(panel, self.val_frac)
        base = CAPMBetas(252, both=True)
        base.fit(inner)
        PZ_in = self._design(inner, fit=True)
        PZ_va = self._design(val, fit=False)
        s_in, s_va = (np.asarray(p.shock, dtype=np.float64) for p in (inner, val))
        e_in = np.asarray(inner.target, dtype=np.float64) - base.predict(inner)[0]
        e_va = np.asarray(val.target, dtype=np.float64) - base.predict(val)[0]
        codes = _codes(val)
        scores = []
        for al in ALPHAS:
            w = self._solve(PZ_in, s_in, e_in, al)
            scores.append(pd.Series((e_va - (PZ_va @ w) * s_va) ** 2).groupby(codes).mean().mean())
        self.alpha_ = float(ALPHAS[int(np.argmin(scores))])
        self.base = CAPMBetas(252, both=True)
        self.base.fit(panel)
        PZ = self._design(panel, fit=True)
        e = np.asarray(panel.target, dtype=np.float64) - self.base.predict(panel)[0]
        self.w_ = self._solve(PZ, np.asarray(panel.shock, dtype=np.float64), e, self.alpha_)

    def prediction_extras(self, panel: Q3Panel) -> dict:
        mu0, _, _ = self.base.predict(panel)
        return dict(mu_base=mu0, delta_a=np.zeros(len(mu0)), delta_b=self._design(panel, fit=False) @ self.w_)

    def predict(self, panel: Q3Panel):
        _, a0, b0 = self.base.predict(panel)
        b = b0 + self._design(panel, fit=False) @ self.w_
        return a0 + b * np.asarray(panel.shock, dtype=np.float64), a0, b


# ----------------------------------------------------------------------------- neural arms

class V4Arm:
    """Two-beta base + an ensemble of projected corrections (b only, or a and b), two-stage fit."""

    def __init__(self, seed: int = 0, *, encoder="transformer", b_only=True, members=3, lr=1e-3,
                 weight_decay=1e-4, max_epochs=40, patience=10, meetings_per_batch=4, val_frac=0.15,
                 device=None, encoder_fn=None):
        self.seed, self.encoder, self.b_only, self.members = int(seed), encoder, bool(b_only), int(members)
        self.lr, self.weight_decay, self.max_epochs, self.patience = lr, weight_decay, max_epochs, patience
        self.meetings_per_batch, self.val_frac, self.device, self.encoder_fn = meetings_per_batch, val_frac, device, encoder_fn
        self.fit_result = None
        self.track_train = True  # the v5 text arm skips the per-epoch training-set score (cost)

    def config(self) -> dict:
        return dict(model="neural", encoder=self.encoder, b_only=self.b_only, members=self.members, lr=self.lr,
                    weight_decay=self.weight_decay, max_epochs=self.max_epochs, patience=self.patience,
                    meetings_per_batch=self.meetings_per_batch, val_frac=self.val_frac, base="capm_beta_both",
                    projection="beta", refit="step-matched")

    def _new_model(self, panel: Q3Panel, device):
        if self.encoder_fn is not None:
            enc = self.encoder_fn(panel.history.shape[-1])
        elif self.encoder == "summary":
            enc = SummaryEncoder(n_channels=panel.history.shape[-1], out_dim=32)
        else:
            enc = PatchTransformerEncoder(n_channels=panel.history.shape[-1])
        model = V2StructModel(enc, n_fund=panel.fundamentals.shape[1], n_ctx=panel.context.shape[1], n_shocks=1,
                              b_only=self.b_only)
        for head in (model.head_b, model.head_a):
            if head is not None:  # zero correction at the start: predicts the base exactly
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        return model.to(device)

    def _optimizer(self, model):
        return torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    @staticmethod
    def _data(p: Q3Panel, base, scaler, device) -> dict:
        t = lambda v, dt=torch.float64: torch.as_tensor(np.asarray(v), dtype=dt, device=device)
        return dict(x=scaler.tensors(p, device), q=t(basis_rows(p, "beta")), codes=t(_codes(p), torch.long),
                    s=t(p.shock), e=t(np.asarray(p.target, dtype=np.float64) - base.predict(p)[0]))

    @staticmethod
    def _eval(model, d, groups) -> float:
        model.eval()
        with torch.no_grad():
            tot = sum(_deployed_loss(model, d["x"], torch.as_tensor(idx, device=d["e"].device), d["q"], d["codes"],
                                     d["s"], d["e"]).item() * len(np.unique(d["codes"][idx].cpu().numpy()))
                      for idx in _chunks(groups))
        model.train()
        return tot / len(groups)

    def _step(self, model, opt, d, rows):
        opt.zero_grad(set_to_none=True)
        loss = _deployed_loss(model, d["x"], rows, d["q"], d["codes"], d["s"], d["e"])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    def _batches(self, groups, rng, device):
        order = rng.permutation(len(groups))
        for start in range(0, len(order), self.meetings_per_batch):
            yield torch.as_tensor(np.concatenate([groups[k] for k in order[start:start + self.meetings_per_batch]]),
                                  device=device)

    def fit(self, panel: Q3Panel, seed: int | None = None, **_) -> None:
        seed = self.seed if seed is None else int(seed)
        device = resolve_device(self.device)
        inner, val = _inner_split(panel, self.val_frac)
        base_in = CAPMBetas(252, both=True)
        base_in.fit(inner)
        self.base_inner_ = base_in  # kept for the inner-validation isolation test
        sc_in = _Scaler(inner)
        d_in, d_va = self._data(inner, base_in, sc_in, device), self._data(val, base_in, sc_in, device)
        g_in, g_va = _meeting_index_groups(inner.meeting), _meeting_index_groups(val.meeting)
        steps_per_epoch = math.ceil(len(g_in) / self.meetings_per_batch)
        self.base = CAPMBetas(252, both=True)
        self.base.fit(panel)
        self.scaler = _Scaler(panel)
        d_full, g_full = self._data(panel, self.base, self.scaler, device), _meeting_index_groups(panel.meeting)
        self.nets, selection, first_curves = [], [], None
        for j in range(self.members):
            member_seed = seed * 100 + j
            # stage 1: choose the stopping point on the inner validation block
            seed_everything(member_seed)
            model = self._new_model(panel, device)
            opt = self._optimizer(model)
            rng = np.random.default_rng(member_seed)
            best, best_epoch, stale, tr_curve, va_curve = self._eval(model, d_va, g_va), 0, 0, [], []
            for epoch in range(1, self.max_epochs + 1):
                for rows in self._batches(g_in, rng, device):
                    self._step(model, opt, d_in, rows)
                tr_curve.append(self._eval(model, d_in, g_in) if self.track_train else float("nan"))
                va_curve.append(self._eval(model, d_va, g_va))
                if va_curve[-1] < best:
                    best, best_epoch, stale = va_curve[-1], epoch, 0
                else:
                    stale += 1
                    if stale >= self.patience:
                        break
            n_steps = best_epoch * steps_per_epoch
            # stage 2: refit on the whole window for exactly the selected number of optimizer steps
            seed_everything(member_seed)
            model = self._new_model(panel, device)
            opt = self._optimizer(model)
            rng, done = np.random.default_rng(member_seed), 0
            while done < n_steps:
                for rows in self._batches(g_full, rng, device):
                    if done >= n_steps:
                        break
                    self._step(model, opt, d_full, rows)
                    done += 1
            model.eval()
            self.nets.append(model)
            selection.append(dict(epoch=best_epoch, steps=n_steps, val=round(best, 6)))
            if first_curves is None:
                first_curves = (tr_curve, va_curve, best, best_epoch)
        tr, va, best, best_epoch = first_curves
        self.fit_result = argparse.Namespace(train_loss=tr, val_loss=va, best_epoch=best_epoch, best_val_loss=best,
                                             val_meetings=val.meetings(), member_selection=selection)

    def corrections(self, panel: Q3Panel):
        device = next(self.nets[0].parameters()).device
        d = self._data(panel, self.base, self.scaler, device)
        da, db = np.zeros(len(panel.target)), np.zeros(len(panel.target))
        with torch.no_grad():
            for model in self.nets:
                for idx in _chunks(_meeting_index_groups(panel.meeting)):
                    a, b, _, _ = _correction(model, d["x"], torch.as_tensor(idx, device=device), d["q"], d["codes"], d["s"])
                    da[idx] += a.cpu().numpy() / len(self.nets)
                    db[idx] += b.cpu().numpy() / len(self.nets)
        return da, db

    def prediction_extras(self, panel: Q3Panel) -> dict:
        da, db = self.corrections(panel)
        return dict(mu_base=self.base.predict(panel)[0], delta_a=da, delta_b=db)

    def predict(self, panel: Q3Panel):
        _, a0, b0 = self.base.predict(panel)
        da, db = self.corrections(panel)
        a, b = a0 + da, b0 + db
        return a + b * np.asarray(panel.shock, dtype=np.float64), a, b


# ----------------------------------------------------------------------------- runner

NEURAL = ("B4", "B4j", "B5", "B5j")


def makers(device=None, members=3, sector_map=None) -> dict:
    return {
        "B0": lambda seed: CAPMBetas(252, both=True),
        "B1": lambda seed: V4Ridge(features="all"),
        "B2": lambda seed: V4Ridge(),
        "B3c": lambda seed: V4Ridge(sector=True, sector_map=sector_map),
        "B3": lambda seed: V4Ridge(sector=True, peer=True, sector_map=sector_map),
        "B4": lambda seed: V4Arm(seed, encoder="summary", b_only=True, members=members, device=device),
        "B4j": lambda seed: V4Arm(seed, encoder="summary", b_only=False, members=members, device=device),
        "B5": lambda seed: V4Arm(seed, encoder="transformer", b_only=True, members=members, device=device),
        "B5j": lambda seed: V4Arm(seed, encoder="transformer", b_only=False, members=members, device=device),
    }


def _run(arms, panel, rows, prefix, params, min_train, test_size, seeds, members, device, store, role):
    from fedcore.q3.real import UNIVERSES
    from fedcore.results import ResultStore
    from fedenc.config import RESULTS
    from fedenc.runs import save_runs

    every = makers(device=device, members=members, sector_map=SectorMap(rows))
    chosen = {a: every[a] for a in arms}
    configs = {a: (m(0).config() if hasattr(m(0), "config") else {"model": "capm_beta_both"}) for a, m in chosen.items()}
    return save_runs(panel, chosen, prefix=prefix, role=role,
                     params=dict(params, configs=configs, panel_fingerprint=panel_fingerprint(panel),
                                 plan="docs/encoder/q3-v4-plan.md"),
                     min_train=min_train, test_size=test_size, seeds=seeds, store=store or ResultStore(RESULTS),
                     datasets=UNIVERSES["sp500"].inputs)


def run(arms, through="2021-12-31", seeds=(0, 1, 2), members=3, device=None, store=None):
    """Development walk-forward (the v2/v3 161 test meetings). Training for B4-B5j."""
    from fedenc.workflow import bounded_panel

    if pd.Timestamp(through) > pd.Timestamp("2021-12-31"):
        raise ValueError("v4 development runs stop at 2021-12-31")
    panel, through_ts, rows = bounded_panel(through, with_rows=True)
    return _run(arms, panel, rows, f"v4-to{through_ts.year}", dict(through=str(through_ts.date())), 60, 16,
                seeds, members, device, store, "exploratory")


def run_final(arms, seeds=(6, 7, 8), members=3, device=None, store=None):
    """The one 2022-2025 check of an arm that passed on development data (plan section 5). Training."""
    from fedcore.protocol import walk_forward
    from fedenc.workflow import bounded_panel

    panel, through, rows = bounded_panel("2025-12-31", with_rows=True)
    meetings = panel.meetings()
    n_dev = int((meetings <= pd.Timestamp("2021-12-31")).sum())
    folds = walk_forward(meetings, n_dev, len(meetings) - n_dev)
    if len(folds) != 1 or folds[0].test.min() <= pd.Timestamp("2021-12-31"):
        raise RuntimeError("the final check must be one fold: train through 2021, test 2022-2025")
    return _run(list(dict.fromkeys(["B0", *arms])), panel, rows, "v4-final",
                dict(through=str(through.date()), dev_end="2021-12-31"), n_dev, len(meetings) - n_dev,
                seeds, members, device, store, "robustness")


# ----------------------------------------------------------------------------- scoring

def _load(prefix: str, arm: str):
    from fedenc.config import RESULTS

    hits = sorted((RESULTS / "q3_susceptibility").glob(f"*_{prefix}-{arm.lower()}"))
    if not hits:
        return None
    p = pd.read_csv(hits[-1] / "tables" / "predictions.csv", parse_dates=["meeting"])
    cols = [c for c in ("mu", "mu_base", "delta_a", "delta_b") if c in p.columns]
    g = p.groupby(["meeting", "firm_id"])
    out = g[cols].mean().join(g["target"].first())
    return out, hits[-1].name


def _boot(x: pd.Series, n=10000, seed=0) -> tuple[float, float, float]:
    v = x.dropna().to_numpy()
    d = np.random.default_rng(seed).choice(v, (n, v.size)).mean(axis=1)
    return float(v.mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def _fmt(t):
    return f"{t[0]:+.4f} ({t[1]:+.4f}, {t[2]:+.4f})"


def score(prefix="v4-to2021", arms=("B0", "B1", "B2", "B3c", "B3", "B4", "B4j", "B5", "B5j"), save=True):
    """Every pre-stated v4 comparison from saved predictions. Read-only apart from the score run."""
    from fedcore.data import load
    from fedcore.q3.features import summary_feature_names, summary_features
    from fedcore.q3.real import FUNDAMENTAL_NAMES
    from fedenc.workflow import bounded_panel

    loaded = {a: _load(prefix, a) for a in arms}
    loaded = {a: v for a, v in loaded.items() if v is not None}
    if "B0" not in loaded:
        raise RuntimeError(f"no B0 run with prefix {prefix}")
    P = {a: v[0] for a, v in loaded.items()}
    df = pd.DataFrame({a: v["mu"] for a, v in P.items()}).join(P["B0"]["target"]).dropna()
    final = prefix.startswith("v4-final")
    panel, _, rows = bounded_panel("2025-12-31" if final else "2021-12-31", with_rows=True)
    s = pd.Series(panel.shock, index=pd.DatetimeIndex(panel.meeting)).groupby(level=0).first()
    meet = df.index.get_level_values(0).unique()
    S = s.reindex(meet).abs()
    dm = lambda c: df[c] - df.groupby(level=0)[c].transform("mean")
    tot = {a: ((df.target - df[a]) ** 2).groupby(level=0).mean() for a in P}
    cs = {a: ((dm("target") - dm(a)) ** 2).groupby(level=0).mean() for a in P}

    # earnings flag: Compustat report date on the meeting day or the trading day before
    f = load("comp_fundq")[["gvkey", "rdq"]].dropna()
    f = f.assign(gvkey=f["gvkey"].astype("string"), rdq=pd.to_datetime(f["rdq"])).drop_duplicates()
    cal = pd.DatetimeIndex(sorted(pd.to_datetime(load("ff_factors_daily", columns=["mktrf"])["date"])))
    gv = pd.Series(rows["gvkey"].astype("string").to_numpy(),
                   index=pd.MultiIndex.from_arrays([pd.DatetimeIndex(rows["meeting"]), rows["permno"].astype(int)]))
    key = pd.DataFrame({"meeting": df.index.get_level_values(0), "firm_id": df.index.get_level_values(1)})
    key["gvkey"] = gv.reindex(pd.MultiIndex.from_frame(key)).to_numpy()
    key["prev"] = key["meeting"].map({m: cal[cal.searchsorted(m) - 1] for m in meet})
    hit = key.merge(f, on="gvkey")
    hit = hit[(hit.rdq == hit.meeting) | (hit.rdq == hit.prev)][["meeting", "firm_id"]].drop_duplicates()
    earn = pd.Series(False, index=df.index)
    earn.loc[pd.MultiIndex.from_frame(hit)] = True
    keep = ~earn
    dmk = lambda c: df.loc[keep, c] - df.loc[keep].groupby(level=0)[c].transform("mean")
    cs_ex = {a: ((dmk("target") - dmk(a)) ** 2).groupby(level=0).mean() for a in P}

    big = meet[S.to_numpy() >= PRIMARY]
    big_x = big[~big.isin(STANDOUT)]
    rows_out = []
    for a in P:
        if a == "B0":
            continue
        d_cs, d_tot, d_ex = cs[a] - cs["B0"], tot[a] - tot["B0"], cs_ex[a] - cs_ex["B0"]
        prim, prim_x = _boot(d_cs[big]), _boot(d_cs[big_x])
        guard = float(d_tot.mean())
        r = dict(arm=a, primary_2p5bp=_fmt(prim), primary_wo_standout=_fmt(prim_x), all_meetings_raw=_fmt(_boot(d_tot)),
                 guardrail_ok=guard <= GUARDRAIL,
                 supported=bool(prim[2] < 0 and prim_x[2] < 0 and guard <= GUARDRAIL))
        for thr in THRESHOLDS:
            r[f"cs_ge_{thr * 10:g}bp"] = _fmt(_boot(d_cs[meet[S.to_numpy() >= thr]]))
        dbig = d_cs[big]
        r["cs_2p5bp_2001_08"] = round(float(dbig[dbig.index.year <= 2008].mean()), 4) if not final else np.nan
        r["cs_2p5bp_2009_21"] = round(float(dbig[dbig.index.year >= 2009].mean()), 4) if not final else np.nan
        r["cs_2p5bp_no_earnings_days"] = _fmt(_boot(d_ex[big]))
        if "delta_b" in P[a]:
            x = (P[a]["delta_b"] * s.reindex(P[a].index.get_level_values(0)).to_numpy()).reindex(df.index)
            x = x - x.groupby(level=0).transform("mean")
            miss = dm("target") - dm("B0")
            r["bxs_slope"] = round(float((x * miss).sum() / (x ** 2).sum()), 2) if float((x ** 2).sum()) > 0 else np.nan
            r["bxs_sd"] = round(float(x.std()), 4)
        r["run"] = loaded[a][1]
        rows_out.append(r)
    table = pd.DataFrame(rows_out)

    # pre-stated pairwise comparisons (primary slice, firm-vs-firm)
    pairs = [("B2", "B1"), ("B2", "B4"), ("B2", "B5"), ("B3", "B3c"), ("B4", "B4j"), ("B5", "B5j"), ("B5", "B4")]
    pair_rows = [dict(comparison=f"{x} - {y}", slice_2p5bp=_fmt(_boot((cs[x] - cs[y])[big])),
                      wo_standout=_fmt(_boot((cs[x] - cs[y])[big_x])), all_meetings_raw=_fmt(_boot(tot[x] - tot[y])))
                 for x, y in pairs if x in P and y in P]
    pairs_t = pd.DataFrame(pair_rows)

    # what each b correction predicts: within-meeting correlations with the named variables
    X = pd.DataFrame(np.column_stack((summary_features(panel), panel.fundamentals)),
                     columns=summary_feature_names() + list(FUNDAMENTAL_NAMES),
                     index=pd.MultiIndex.from_arrays([pd.DatetimeIndex(panel.meeting), panel.firm_id.astype(int)]))
    corr_rows = []
    for a in P:
        if "delta_b" not in P[a] or float(P[a]["delta_b"].abs().sum()) == 0:
            continue
        db = P[a]["delta_b"] - P[a]["delta_b"].groupby(level=0).transform("mean")
        Xc = X.reindex(db.index)[list(NAMED)]
        Xc = Xc - Xc.groupby(level=0).transform("mean")
        corr_rows.append({"arm": a, **{c: round(float(np.corrcoef(Xc[c].fillna(0), db)[0, 1]), 2) for c in NAMED}})
    corr_t = pd.DataFrame(corr_rows)
    counts = pd.DataFrame([dict(meetings=len(meet), ge_2p5bp=len(big), ge_2p5bp_wo_standout=len(big_x),
                                ge_5bp=int((S >= 0.5).sum()), earnings_rows=int(earn.sum()), rows=len(df))])
    mse = pd.DataFrame([dict(arm=a, mse=float(tot[a].mean()), cross_firm=float(cs[a].mean()),
                             cross_firm_2p5bp=float(cs[a][big].mean())) for a in P])
    with pd.option_context("display.width", 300, "display.max_columns", 40, "display.max_colwidth", 60):
        print(counts.to_string(index=False))
        print("\n== MSE\n" + mse.to_string(index=False, float_format="{:.4f}".format))
        print("\n== vs B0 (negative = better)\n" + table.drop(columns=["run"]).to_string(index=False))
        print("\n== pre-stated pairs\n" + pairs_t.to_string(index=False))
        print("\n== b correction vs named variables (within-meeting correlation)\n" + corr_t.to_string(index=False))
    if save:
        from fedcore.questions import specs
        from fedcore.results import ResultStore
        from fedenc.config import RESULTS

        ResultStore(RESULTS).save(specs()["Q3"], f"{prefix}-score", role="exploratory",
                                  params=dict(prefix=prefix, primary=PRIMARY, guardrail=GUARDRAIL,
                                              standout=[str(d.date()) for d in STANDOUT],
                                              runs={a: v[1] for a, v in loaded.items()}),
                                  metrics={f"{r['arm']}_supported": int(r["supported"]) for r in rows_out},
                                  tables=dict(counts=counts, mse=mse, vs_b0=table, pairs=pairs_t, b_correlations=corr_t),
                                  notes="Read-only scoring of saved v4 predictions (docs/encoder/q3-v4-plan.md).")
    return table, pairs_t, corr_t


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arms", nargs="+", required=True, choices=list(makers()))
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    r.add_argument("--members", type=int, default=3)
    r.add_argument("--device")
    f = sub.add_parser("final")
    f.add_argument("--arms", nargs="+", required=True, choices=list(makers()))
    f.add_argument("--device")
    sc = sub.add_parser("score")
    sc.add_argument("--prefix", default="v4-to2021")
    sc.add_argument("--no-save", action="store_true")
    args = p.parse_args(argv)
    if args.command == "run":
        run(args.arms, seeds=tuple(args.seeds), members=args.members, device=args.device)
    elif args.command == "final":
        run_final(args.arms, device=args.device)
    else:
        score(prefix=args.prefix, save=not args.no_save)


if __name__ == "__main__":
    main()
