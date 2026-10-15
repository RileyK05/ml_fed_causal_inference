"""Diagnostics on saved Q3 predictions: where the MSE comes from, and what a and b learned.

No fitting. Reads ``predictions.csv`` from saved runs, maps each meeting's shock and each
firm's pre-meeting traits from the real panel, and reports:

1. MSE split per meeting into the meeting-average part and the firm-to-firm part
   (total = mean part + cross-firm part, exactly), overall, by test window and crisis vs not,
   with paired meeting-bootstrap ranges against references.
2. Anatomy: within-meeting correlations of a and b with each other, with 63-day beta and
   21-day volatility, and how much of b a linear mix of simple traits explains.
3. Persistence of each firm's relative b from one meeting to the next (same fold only).
4. b grade variants, meeting-clustered: raw; after subtracting a; controlling for
   beta x shock; controlling for beta x the meeting's realized average return. All meetings
   and large-shock meetings (|s| >= 0.25, i.e. >= 2.5bp).

Statistics are computed per seed and then averaged, as in ``daily.b_grade``.

    python -m fedenc.diagnostics NAME=RUN_DIR [NAME=RUN_DIR ...] [--reference pooled scratch]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from fedcore.q3.contracts import BETA63, RET, RVOL21
from fedcore.q3.real import FUNDAMENTAL_NAMES

LARGE_SHOCK = 0.25
CRISIS_FOLD = 3  # 2007-10 to 2009-03
N_BOOT = 10_000


# ----------------------------------------------------------------------------- inputs

def panel_frames(panel):
    """(shock per meeting, per-row traits as of the prior close) from a Q3 panel."""
    meeting = pd.DatetimeIndex(panel.meeting)
    shock = pd.Series(panel.shock.astype(np.float64), index=meeting).groupby(level=0).first()
    hist, mask = panel.history.astype(np.float64), panel.history_mask
    ret = np.where(mask, hist[:, :, RET], np.nan)
    traits = pd.DataFrame({
        "meeting": meeting, "firm_id": panel.firm_id,
        "beta63": hist[:, -1, BETA63], "vol21": hist[:, -1, RVOL21],
        "ret21": np.nansum(ret[:, -21:], axis=1), "mom252": np.nansum(ret, axis=1),
        **{name: panel.fundamentals[:, k].astype(np.float64)
           for k, name in enumerate(FUNDAMENTAL_NAMES[:panel.fundamentals.shape[1]])},
    })
    return shock, traits


def load_predictions(run_dir, shock, traits):
    df = pd.read_csv(f"{run_dir}/tables/predictions.csv", parse_dates=["meeting"])
    df["s"] = df["meeting"].map(shock)
    if df["s"].isna().any():
        raise ValueError(f"{run_dir}: meetings without a shock")
    return df.merge(traits, on=["meeting", "firm_id"], how="left", validate="many_to_one")


def _demean(df, cols):
    g = df.groupby(["seed", "meeting"])
    return {c: df[c] - g[c].transform("mean") for c in cols}


# ----------------------------------------------------------------------------- 1. MSE split

def meeting_parts(df) -> pd.DataFrame:
    """Per (seed, meeting): total, mean part, cross-firm part, and a-only / b-only versions."""
    d = _demean(df, ["target", "mu", "a_hat", "b_hat"])
    yc, ac, bs = d["target"], d["a_hat"], d["b_hat"] * df["s"]
    frame = pd.DataFrame({
        "seed": df["seed"], "meeting": df["meeting"], "fold": df["fold"],
        "total": (df["target"] - df["mu"]) ** 2,
        "cs": (yc - d["mu"]) ** 2,
        "cs_zero": yc ** 2, "cs_a_only": (yc - ac) ** 2, "cs_b_only": (yc - bs) ** 2,
        "y": df["target"], "m": df["mu"],
    })
    out = frame.groupby(["seed", "meeting"]).agg(
        fold=("fold", "first"), total=("total", "mean"), cs=("cs", "mean"), cs_zero=("cs_zero", "mean"),
        cs_a_only=("cs_a_only", "mean"), cs_b_only=("cs_b_only", "mean"), ybar=("y", "mean"), mbar=("m", "mean"))
    out["mean_part"] = (out["ybar"] - out["mbar"]) ** 2
    gap = (out["total"] - out["mean_part"] - out["cs"]).abs().max()
    if gap > 1e-5:
        raise AssertionError(f"MSE split does not add up (max gap {gap})")
    return out.drop(columns=["ybar", "mbar"]).reset_index()


def per_meeting(parts) -> pd.DataFrame:
    """Seed-average of each per-meeting quantity."""
    return parts.groupby("meeting").agg(fold=("fold", "first"), **{
        c: (c, "mean") for c in ("total", "mean_part", "cs", "cs_zero", "cs_a_only", "cs_b_only")})


def paired_range(left, right, *, n=N_BOOT, seed=0):
    """Mean of (left - right) over meetings with a meeting-bootstrap 95% range."""
    diff = (left - right).dropna().to_numpy()
    draws = np.random.default_rng(seed).choice(diff, size=(n, diff.size), replace=True).mean(axis=1)
    return float(diff.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def mse_table(meetings: dict, references=("pooled", "scratch")) -> pd.DataFrame:
    rows = []
    for name, pm in meetings.items():
        row = {"model": name, **{c: pm[c].mean() for c in ("total", "mean_part", "cs", "cs_a_only", "cs_b_only")},
               "total_win0_2": pm.loc[pm.fold != CRISIS_FOLD, "total"].mean(),
               "total_win3": pm.loc[pm.fold == CRISIS_FOLD, "total"].mean()}
        for ref in references:
            if ref in meetings and ref != name:
                for part in ("total", "mean_part", "cs"):
                    est, lo, hi = paired_range(pm[part], meetings[ref][part])
                    row[f"{part}_vs_{ref}"] = f"{est:+.3f} ({lo:+.3f}, {hi:+.3f})"
        rows.append(row)
    return pd.DataFrame(rows)


def window_table(meetings: dict) -> pd.DataFrame:
    rows = []
    for name, pm in meetings.items():
        for fold, g in pm.groupby("fold"):
            rows.append({"model": name, "window": int(fold), "meetings": len(g),
                         "total": g["total"].mean(), "mean_part": g["mean_part"].mean(), "cs": g["cs"].mean()})
    return pd.DataFrame(rows)


SHOCK_BINS = (0.0, 0.1, 0.25, 0.5, np.inf)  # |s| in 10bp units: <1bp, 1-2.5bp, 2.5-5bp, >=5bp
SHOCK_LABELS = ("tiny <1bp", "small 1-2.5bp", "medium 2.5-5bp", "large >=5bp")


def shock_table(meetings: dict, shock, reference="pooled") -> pd.DataFrame:
    """Error by surprise size: b's cross-firm effect scales with s^2, so large surprises are
    where a real b should show. Paired ranges against ``reference`` within each bin."""
    rows = []
    for name, pm in meetings.items():
        size = pd.cut(pm.index.map(shock).to_numpy().__abs__(), SHOCK_BINS, labels=SHOCK_LABELS, right=False)
        for label in SHOCK_LABELS:
            idx = pm.index[size == label]
            if not len(idx):
                continue
            g = pm.loc[idx]
            row = {"model": name, "surprise": label, "meetings": len(idx),
                   "crisis_meetings": int((g.fold == CRISIS_FOLD).sum()), "total": g["total"].mean(),
                   "mean_part": g["mean_part"].mean(), "cs": g["cs"].mean(),
                   "share_of_error": g["total"].sum() / pm["total"].sum()}
            if reference in meetings and name != reference:
                ref = meetings[reference].loc[idx]
                for part in ("total", "cs"):
                    est, lo, hi = paired_range(g[part], ref[part])
                    row[f"{part}_vs_{reference}"] = f"{est:+.3f} ({lo:+.3f}, {hi:+.3f})"
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 2-3. anatomy

def _within_corr(df, x, y, method="pearson"):
    vals = []
    for _, g in df.groupby(["seed", "meeting"]):
        g = g[[x, y]].dropna()
        if len(g) > 2 and g[x].std() > 1e-12 and g[y].std() > 1e-12:
            vals.append(g[x].corr(g[y], method=method))
    return float(np.mean(vals)) if vals else np.nan


def surrogate_r2(df, features=("beta63", "vol21", "ret21", "mom252", "leverage", "log_size",
                               "profitability", "cash", "book_to_market")):
    """Mean within-meeting R^2 of b_hat on a linear mix of simple pre-meeting traits."""
    vals = []
    for _, g in df.groupby(["seed", "meeting"]):
        g = g[["b_hat", *features]].dropna()
        if len(g) <= len(features) + 2 or g["b_hat"].std() < 1e-12:
            continue
        X = np.column_stack([np.ones(len(g)), g[list(features)].to_numpy()])
        y = g["b_hat"].to_numpy()
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        vals.append(1.0 - np.sum((y - X @ coef) ** 2) / np.sum((y - y.mean()) ** 2))
    return float(np.mean(vals)) if vals else np.nan


def persistence(df, lag=1):
    """Within the same fold and seed: corr of each firm's relative b at meeting k and k-lag."""
    d = df.assign(b_rel=_demean(df, ["b_hat"])["b_hat"])
    vals, flips = [], []
    for (seed, fold), g in d.groupby(["seed", "fold"]):
        wide = g.pivot_table(index="firm_id", columns="meeting", values="b_rel")
        cols = sorted(wide.columns)
        for k in range(lag, len(cols)):
            pair = wide[[cols[k - lag], cols[k]]].dropna()
            if len(pair) > 2 and pair.std().min() > 1e-12:
                vals.append(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
                flips.append(float((np.sign(pair.iloc[:, 0]) != np.sign(pair.iloc[:, 1])).mean()))
    return (float(np.mean(vals)) if vals else np.nan, float(np.mean(flips)) if flips else np.nan)


def anatomy_table(frames: dict) -> pd.DataFrame:
    rows = []
    for name, df in frames.items():
        if df.groupby(["seed", "meeting"])["b_hat"].std().max() < 1e-12:
            continue  # pooled: b does not vary across firms
        p1, flip = persistence(df, 1)
        p4, _ = persistence(df, 4)
        rows.append({
            "model": name,
            "corr_a_b": _within_corr(df, "a_hat", "b_hat"),
            "spearman_a_b": _within_corr(df, "a_hat", "b_hat", "spearman"),
            "corr_b_beta": _within_corr(df, "b_hat", "beta63"), "corr_b_vol": _within_corr(df, "b_hat", "vol21"),
            "corr_a_beta": _within_corr(df, "a_hat", "beta63"), "corr_a_vol": _within_corr(df, "a_hat", "vol21"),
            "corr_b_leverage": _within_corr(df, "b_hat", "leverage"),
            "corr_b_size": _within_corr(df, "b_hat", "log_size"),
            "surrogate_r2_b": surrogate_r2(df),
            "b_persist_next": p1, "b_persist_4": p4, "b_sign_flip_next": flip,
            "b_spread": float(df.groupby(["seed", "meeting"])["b_hat"].std().mean()),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 4. b grades

def ols_clustered(X, y, groups):
    """No-intercept OLS (inputs already demeaned by meeting); meeting-clustered SEs."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    xtx_inv = np.linalg.pinv(X.T @ X)
    coef = xtx_inv @ X.T @ y
    resid = y - X @ coef
    scores = pd.DataFrame(X * resid[:, None]).groupby(np.asarray(groups)).sum().to_numpy()
    se = np.sqrt(np.diag(xtx_inv @ (scores.T @ scores) @ xtx_inv))
    return coef, se


def b_grades(df, large_only=False) -> dict:
    """Four b-grade specifications plus the unrestricted a-and-b version; seed-averaged."""
    df = df[df["s"].abs() >= LARGE_SHOCK] if large_only else df
    out = {}
    for seed, g in df.groupby("seed"):
        g = g.dropna(subset=["beta63"])
        d = _demean(g, ["target", "a_hat", "b_hat", "beta63"])
        ybar = g.groupby(["seed", "meeting"])["target"].transform("mean")
        yc, ac, x = d["target"].to_numpy(), d["a_hat"].to_numpy(), (d["b_hat"] * g["s"]).to_numpy()
        beta_s, beta_m = (d["beta63"] * g["s"]).to_numpy(), (d["beta63"] * ybar).to_numpy()
        if np.sum(x * x) < 1e-12:
            return {}
        m = g["meeting"].to_numpy()
        specs = {
            "raw": ([x], yc), "minus_a": ([x], yc - ac), "a_and_b": ([ac, x], yc),
            "ctrl_beta_shock": ([x, beta_s], yc - ac), "ctrl_beta_market": ([x, beta_m], yc - ac),
        }
        for name, (cols, y) in specs.items():
            coef, se = ols_clustered(np.column_stack(cols), y, m)
            k = 1 if name == "a_and_b" else 0  # coefficient on b x shock
            out.setdefault(name, []).append((coef[k], se[k]))
            if name == "a_and_b":
                out.setdefault("a_coef", []).append((coef[0], se[0]))
            if name == "ctrl_beta_market":
                out.setdefault("beta_market_coef", []).append((coef[1], se[1]))
    result = {}
    for name, vals in out.items():
        est, se = np.mean([v[0] for v in vals]), np.mean([v[1] for v in vals])
        t = est / se if se > 0 else np.nan  # se is 0 when a regressor is constant (e.g. flat a)
        result[name] = f"{est:+.2f} (t {t:+.1f})"
    result["meetings"] = int(df["meeting"].nunique())
    return result


def rank_ic(df, large_only=False):
    df = df[df["s"].abs() >= LARGE_SHOCK] if large_only else df
    return _within_corr(df, "mu", "target", "spearman")


def grade_table(frames: dict) -> pd.DataFrame:
    rows = []
    for name, df in frames.items():
        for subset, large in (("all", False), ("large_shock", True)):
            row = {"model": name, "meetings": subset, "rank_ic": rank_ic(df, large)}
            row.update(b_grades(df, large))
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- driver

def run(runs: dict, references=("pooled", "scratch"), store=None, panel=None, keep=None, label="diagnostics"):
    """``runs`` maps a display name to a saved run directory. Saves one exploratory run.

    ``keep``: optional (meeting, firm_id) frame; only those rows are scored (e.g. the S&P 500
    members of a broader universe, so scores compare with the S&P-only panel)."""
    if panel is None:
        from fedenc.workflow import bounded_panel

        panel, _ = bounded_panel("2009-03-18")
    shock, traits = panel_frames(panel)
    frames = {name: load_predictions(path, shock, traits) for name, path in runs.items()}
    if keep is not None:
        keep = keep[["meeting", "firm_id"]].assign(meeting=lambda d: pd.to_datetime(d["meeting"]))
        frames = {name: df.merge(keep, on=["meeting", "firm_id"], validate="many_to_one")
                  for name, df in frames.items()}
    meetings = {name: per_meeting(meeting_parts(df)) for name, df in frames.items()}
    tables = {
        "mse-split": mse_table(meetings, references), "mse-by-window": window_table(meetings),
        "mse-by-surprise": shock_table(meetings, shock, references[0]),
        "anatomy": anatomy_table(frames), "b-grades": grade_table(frames),
    }
    if store is not None:
        from fedcore.questions import specs

        store.save(specs()["Q3"], label, role="exploratory",
                   params=dict(runs={k: str(v) for k, v in runs.items()}, large_shock=LARGE_SHOCK,
                               n_boot=N_BOOT, references=list(references), rows_kept=None if keep is None
                               else int(len(keep))),
                   tables=tables, notes="No fitting: computed from saved predictions.")
    return tables


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="NAME=RUN_DIR")
    p.add_argument("--reference", nargs="+", default=["pooled", "scratch"])
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args(argv)
    runs = dict(item.split("=", 1) for item in args.runs)
    store = None
    if not args.no_save:
        from fedcore.results import ResultStore
        from fedenc.config import RESULTS

        store = ResultStore(RESULTS)
    tables = run(runs, tuple(args.reference), store=store)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
        for name, table in tables.items():
            print(f"\n== {name}\n{table.to_string(index=False)}")


if __name__ == "__main__":
    main()
