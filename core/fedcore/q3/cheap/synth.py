"""S1-S5: synthetic stress tests with linear models, where the true a and b are known.

Histories, fundamentals and context come from ``make_synthetic_panel``; the target is
rewritten per case. Surprises are drawn per meeting from a Laplace distribution with
median |s| of about 0.14 (1.4bp), matching the real median. Success here is evidence
about the method, not about the real relationship's form.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fedcore.protocol import walk_forward
from fedcore.q3.features import summary_features
from fedcore.q3.synthetic import make_synthetic_panel, synthetic_components

from fedcore.q3.cheap.base import Result, check
from fedcore.q3.cheap.data import Config
from fedcore.q3.cheap.linear import Standardizer, boot_mean, meeting_mse, ridge_fit_predict

LAPLACE_SCALE = 0.2   # median |s| = 0.2 * ln 2 = 0.14
NOISE_SD = 1.5


def _base(cfg: Config, seed: int) -> dict:
    panel = make_synthetic_panel(n_firms=cfg.synthetic_firms, n_meetings=cfg.synthetic_meetings, noise="easy", seed=seed)
    comp = synthetic_components(panel)
    z = np.concatenate((summary_features(panel), panel.fundamentals.astype(np.float64),
                        panel.context.astype(np.float64)), axis=1)
    rng = np.random.default_rng(seed + 1)
    meetings = panel.meetings()
    s_m = pd.Series(rng.laplace(0.0, LAPLACE_SCALE, len(meetings)), index=meetings)
    meeting = pd.to_datetime(panel.meeting)
    std = lambda v: Standardizer().fit(v[:, None]).transform(v[:, None])[:, 0]
    return {
        "z": z, "meeting": panel.meeting, "meetings": meetings, "firm": panel.firm_id,
        "s": s_m.reindex(meeting).to_numpy(), "lev": std(comp["leverage"].to_numpy()),
        "rvol": std(comp["rvol63"].to_numpy()), "rng": rng,
        "folds": walk_forward(meetings, int(len(meetings) * 0.6), max(1, len(meetings) // 8)),
    }


def _evaluate(base: dict, y, a_true, b_true, cfg: Config, *, z=None, offset=None) -> dict:
    """Out-of-fold ab0 vs abz: per-meeting MSE, paired gain, and recovery of a, b and mu."""
    z = base["z"] if z is None else z
    s = base["s"]
    mse = {"ab0": [], "abz": []}
    b_hat, idx, mu_hat = [], [], []
    for fold in base["folds"]:
        own = np.asarray(base["meeting"], dtype="datetime64[ns]")
        tr = np.flatnonzero(np.isin(own, fold.train.to_numpy(dtype="datetime64[ns]")))
        te = np.flatnonzero(np.isin(own, fold.test.to_numpy(dtype="datetime64[ns]")))
        for kind in mse:
            p = ridge_fit_predict(kind, z, s, y, tr, te, alpha=cfg.ridge_alpha, offset=offset)
            mse[kind].append(meeting_mse(base["meeting"][te], y[te], p["mu"]))
            if kind == "abz":
                b_hat.append(p["b"])
                mu_hat.append(p["mu"])
                idx.append(te)
    idx = np.concatenate(idx)
    b_hat, mu_hat = np.concatenate(b_hat), np.concatenate(mu_hat)
    left, right = pd.concat(mse["abz"]), pd.concat(mse["ab0"])
    gain = boot_mean((left - right.loc[left.index]).to_numpy(), cfg.n_boot, cfg.seed)
    bt, at = b_true[idx], a_true[idx]
    mu_true = at + bt * s[idx]
    return {
        "abz_minus_ab0": gain["estimate"], "ci_low": gain["ci_low"], "ci_high": gain["ci_high"],
        "b_corr": float(np.corrcoef(b_hat, bt)[0, 1]) if np.std(bt) > 1e-9 else np.nan,
        "b_rmse": float(np.sqrt(np.mean((b_hat - bt) ** 2))), "b_hat_sd": float(np.std(b_hat)),
        "b_true_sd": float(np.std(bt)), "mu_rmse_vs_truth": float(np.sqrt(np.mean((mu_hat - mu_true) ** 2))),
    }


def _baseline_a(base: dict) -> np.ndarray:
    return 1.0 * base["lev"] + 0.5 * base["rvol"]


@check("S1", "Synthetic: varying a, constant b (false heterogeneity?)", "synthetic", uses_real=False)
def s1(_data, cfg: Config) -> Result:
    base = _base(cfg, cfg.seed)
    a = _baseline_a(base)
    b = np.full_like(a, -1.0)
    y = a + b * base["s"] + base["rng"].normal(0.0, NOISE_SD, a.size)
    r = _evaluate(base, y, a, b, cfg)
    return Result({"b_hat_sd": r["b_hat_sd"], "abz_minus_ab0": r["abz_minus_ab0"], "ci_low": r["ci_low"]},
                  {"result": pd.DataFrame([r])},
                  "Only the baseline varies; true b is -1 everywhere. b_hat_sd is the invented heterogeneity; "
                  "abz should not beat ab0 (ci_low should be at or above 0).")


@check("S2", "Synthetic: varying b, strong baseline, mostly quiet shocks", "synthetic", uses_real=False)
def s2(_data, cfg: Config) -> Result:
    base = _base(cfg, cfg.seed)
    a = _baseline_a(base)
    rows = []
    for sd_b in (0.25, 0.5, 1.0, 2.0):
        b = -1.0 + sd_b * (base["lev"] + base["rvol"]) / np.sqrt(2.0)
        y = a + b * base["s"] + base["rng"].normal(0.0, NOISE_SD, a.size)
        rows.append({"true_sd_b": sd_b, **_evaluate(base, y, a, b, cfg)})
    table = pd.DataFrame(rows)
    return Result({f"b_corr_sd{r.true_sd_b}": r.b_corr for r in table.itertuples()}, {"by_effect": table},
                  "b depends on leverage and volatility; the baseline is strong and surprises are mostly quiet. "
                  "How much b heterogeneity can linear interactions recover, and does the baseline suffer?")


@check("S3", "Synthetic: shared meeting noise", "synthetic", uses_real=False)
def s3(_data, cfg: Config) -> Result:
    base = _base(cfg, cfg.seed)
    a = _baseline_a(base)
    b = -1.0 + 1.0 * (base["lev"] + base["rvol"]) / np.sqrt(2.0)
    codes = pd.factorize(pd.to_datetime(base["meeting"]))[0]
    rows = []
    for sd_c in (0.0, 1.0, 2.0, 4.0):
        common = base["rng"].normal(0.0, sd_c, codes.max() + 1)[codes] if sd_c > 0 else 0.0
        y = a + b * base["s"] + common + base["rng"].normal(0.0, NOISE_SD, a.size)
        rows.append({"common_noise_sd": sd_c, **_evaluate(base, y, a, b, cfg)})
    table = pd.DataFrame(rows)
    return Result({f"b_corr_common{r.common_noise_sd}": r.b_corr for r in table.itertuples()}, {"by_noise": table},
                  "A residual shared by every firm in a meeting, at increasing size. Per-meeting MSE rises, but "
                  "within-meeting b recovery should hold if the shared part does not correlate with firm state.")


@check("S4", "Synthetic: helpful vs misleading ordinary-day exposure", "synthetic", uses_real=False)
def s4(_data, cfg: Config) -> Result:
    base = _base(cfg, cfg.seed)
    rng = base["rng"]
    n_firms = int(np.max(base["firm"])) + 1
    u = rng.normal(0.0, 1.0, n_firms)                      # latent Fed sensitivity driver, unobserved
    a = _baseline_a(base)
    b = -1.0 + 0.7 * u[base["firm"]]
    y = a + b * base["s"] + rng.normal(0.0, NOISE_SD, a.size)
    rows = []
    for rho, label in ((1.0, "related"), (0.0, "unrelated"), (-1.0, "opposed")):
        v = rng.normal(0.0, 1.0, n_firms)
        ordinary = rho * u + np.sqrt(max(1 - rho ** 2, 0.0)) * v
        f = (ordinary + rng.normal(0.0, 0.3, n_firms))[base["firm"]]   # the "pre-trained" estimate
        none = _evaluate(base, y, a, b, cfg)
        feat = _evaluate(base, y, a, b, cfg, z=np.column_stack((base["z"], f)))
        prior = _evaluate(base, y, a, b, cfg, offset=-1.0 + 0.7 * f)
        for how, r in (("no_transfer", none), ("as_feature", feat), ("as_prior_offset", prior)):
            rows.append({"relation": label, "rho": rho, "transfer": how, **r})
    table = pd.DataFrame(rows)
    piv = table.pivot_table(index="relation", columns="transfer", values="b_rmse")
    return Result({f"b_rmse_{rel}_{how}": float(piv.loc[rel, how]) for rel in piv.index for how in piv.columns},
                  {"transfer": table},
                  "Fed sensitivity depends on a latent firm trait. The ordinary-day exposure is related, unrelated "
                  "or opposed to it. As a learned feature its sign is estimated, so 'opposed' still helps; used as "
                  "a fixed prior (offset) it causes negative transfer when unrelated or opposed. A frozen "
                  "pre-trained encoder sits between the two.")


@check("S5", "Synthetic: shock-dependent noise and saturating response", "synthetic", uses_real=False)
def s5(_data, cfg: Config) -> Result:
    base = _base(cfg, cfg.seed)
    rng = base["rng"]
    a = _baseline_a(base)
    b = -1.0 + 1.0 * (base["lev"] + base["rvol"]) / np.sqrt(2.0)
    s = base["s"]
    k = 0.3
    cases = {
        "heteroskedastic": (a + b * s + rng.normal(0.0, 1.0, a.size) * NOISE_SD * (1.0 + 3.0 * np.abs(s)), b * s),
        "saturating": (a + b * k * np.tanh(s / k) + rng.normal(0.0, NOISE_SD, a.size), b * k * np.tanh(s / k)),
    }
    rows, calib = [], []
    for name, (y, shock_part) in cases.items():
        r = _evaluate(base, y, a, b, cfg)
        own = np.asarray(base["meeting"], dtype="datetime64[ns]")
        tr = np.flatnonzero(np.isin(own, base["folds"][0].train.to_numpy(dtype="datetime64[ns]")))
        te = np.flatnonzero(~np.isin(own, base["folds"][0].train.to_numpy(dtype="datetime64[ns]")))
        p = ridge_fit_predict("abz", base["z"], s, y, tr, te, alpha=cfg.ridge_alpha)
        resid2 = (y[te] - p["mu"]) ** 2
        slope = float(np.polyfit(np.abs(s[te]), resid2, 1)[0])
        q = pd.qcut(np.abs(s[te]), 5, labels=False, duplicates="drop")
        for bin_ in np.unique(q):
            sel = q == bin_
            pred_part = p["b"][sel] * s[te][sel]
            true_part = shock_part[te][sel]
            calib.append({"case": name, "abs_s_quintile": int(bin_), "mean_abs_s": float(np.abs(s[te][sel]).mean()),
                          "mean_abs_pred_shock_part": float(np.abs(pred_part).mean()),
                          "mean_abs_true_shock_part": float(np.abs(true_part).mean())})
        rows.append({"case": name, "resid2_slope_on_abs_s": slope, **r})
    calib = pd.DataFrame(calib)
    top = calib[calib["abs_s_quintile"] == calib["abs_s_quintile"].max()]
    over = (top["mean_abs_pred_shock_part"] / top["mean_abs_true_shock_part"]).to_numpy()
    return Result({"overswing_top_quintile_saturating": float(over[-1]) if len(over) else np.nan,
                   "resid2_slope_heteroskedastic": float(rows[0]["resid2_slope_on_abs_s"])},
                  {"cases": pd.DataFrame(rows), "calibration_by_abs_s": calib},
                  "Heteroskedastic: noise grows with |s|, linear mean; the residual^2 slope on |s| should be "
                  "positive. Saturating: response flattens past 3bp; a ratio above 1 in the top |s| quintile is "
                  "overswing from the linear head.")
