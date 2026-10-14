"""A1-A7: descriptive data checks and audits. Nothing is fitted."""
from __future__ import annotations

import numpy as np
import pandas as pd

from fedcore.q3.contracts import BETA63, DLOGVOL, L, RET, RET_REL, RVOL21

from fedcore.q3.cheap.base import Result, check
from fedcore.q3.cheap.data import CheapData, Config
from fedcore.q3.cheap.linear import dev_meetings, inner_folds, outer_folds

QUIET_BP = 1.0  # |surprise| below 1bp counts as a quiet meeting


def _abs_stats(bp: np.ndarray, prefix: str) -> dict:
    v = np.asarray(bp, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {f"{prefix}_n": 0}
    a = np.abs(v)
    return {
        f"{prefix}_n": int(v.size), f"{prefix}_signed_mean_bp": float(v.mean()),
        f"{prefix}_abs_p50_bp": float(np.median(a)), f"{prefix}_abs_p90_bp": float(np.quantile(a, 0.9)),
        f"{prefix}_abs_max_bp": float(a.max()),
    }


@check("A1", "Shock distribution by era and direction", "descriptive")
def a1(data: CheapData, cfg: Config) -> Result:
    m = data.meeting_table()
    bp = 10.0 * m["s"].to_numpy()
    ordinary = data.ordinary_days()
    span = (data.calendar >= m["meeting"].min()) & (data.calendar <= m["meeting"].max())
    era_day = (data.calendar.year // 5) * 5
    rows = []
    for era in sorted(m["era"].unique()):
        sel = (m["era"] == era).to_numpy()
        days = span & ordinary & (era_day == era)
        rows.append({
            "era": int(era), "meetings": int(sel.sum()),
            "hawkish": int((bp[sel] > 0).sum()), "dovish": int((bp[sel] < 0).sum()),
            "quiet_under_1bp": int((np.abs(bp[sel]) < QUIET_BP).sum()),
            **_abs_stats(bp[sel], "fed"), **_abs_stats(data.dy[days], "ordinary"),
        })
    by_era = pd.DataFrame(rows)
    dev = m.index < cfg.dev_meetings
    split = pd.DataFrame([
        {"period": name, "meetings": int(sel.sum()), "hawkish_share": float((bp[sel] > 0).mean()),
         **_abs_stats(bp[sel], "fed")}
        for name, sel in (("development", dev), ("outer", ~dev), ("all", np.ones(len(m), bool)))
        if sel.any()
    ])
    metrics = {**_abs_stats(bp, "fed"), **_abs_stats(data.dy[span & ordinary], "ordinary")}
    return Result(metrics, {"by_era": by_era, "by_period": split},
                  "Signed surprises in bp (positive = hawkish). Ordinary days exclude every FOMC statement date.")


@check("A2", "Shock concentration by training fold", "descriptive")
def a2(data: CheapData, cfg: Config) -> Result:
    m = data.meeting_table().set_index("meeting")["s"]
    rows, top = [], []
    for design, folds in (("inner", inner_folds(data, cfg)), ("outer", outer_folds(data, cfg))):
        for fold in folds:
            s = m.loc[fold.train]
            c = ((s - s.mean()) ** 2).sort_values(ascending=False)
            total = float(c.sum())
            row = {"design": design, "fold": fold.k, "n_train": len(s), "train_end": fold.train.max()}
            for k in (1, 5, 10):
                row[f"top{k}_share"] = float(c.head(k).sum() / total) if total > 0 else np.nan
            rows.append(row)
            if fold.k == 0:
                top += [{"design": design, "rank": i + 1, "meeting": d, "s_bp": 10.0 * float(m.loc[d]),
                         "share": float(c.loc[d] / total)} for i, d in enumerate(c.head(10).index)]
    table = pd.DataFrame(rows)
    metrics = {f"{d}_fold0_top5_share": float(table.loc[(table.design == d) & (table.fold == 0), "top5_share"].iloc[0])
               for d in ("inner", "outer") if ((table.design == d) & (table.fold == 0)).any()}
    return Result(metrics, {"by_fold": table, "fold0_top_meetings": pd.DataFrame(top)},
                  "Shares of centered shock sum of squares: leverage concentration, not fitted influence.")


@check("A3", "Market-rate co-movement on ordinary days", "descriptive")
def a3(data: CheapData, cfg: Config) -> Result:
    f = pd.DataFrame({"mkt": data.mkt, "dy": data.dy, "smb": data.smb, "hml": data.hml}, index=data.calendar)
    m = data.meetings()
    f = f[data.ordinary_days() & (data.calendar >= m.min()) & (data.calendar <= m.max())].dropna()
    rows = []
    for era, g in f.groupby((f.index.year // 5) * 5):
        x = (g - g.mean()) / g.std(ddof=0)
        sv = np.linalg.svd(x.to_numpy(), compute_uv=False)
        rows.append({
            "era": int(era), "days": len(g), "corr_mkt_dy": float(g["mkt"].corr(g["dy"])),
            "same_sign_share": float((np.sign(g["mkt"]) == np.sign(g["dy"])).mean()),
            "cond_mkt_dy_smb_hml": float(sv[0] / max(sv[-1], 1e-12)),
        })
    roll = f["mkt"].rolling(63).corr(f["dy"]).dropna()
    metrics = {"corr_mkt_dy_all": float(f["mkt"].corr(f["dy"])),
               "rolling63_corr_positive_share": float((roll > 0).mean()) if len(roll) else np.nan}
    return Result(metrics, {"by_era": pd.DataFrame(rows)},
                  "Correlation of daily market return and 1-year yield change; the sign flips across eras.")


@check("A4", "Meeting-wide vs within-meeting return variation", "descriptive")
def a4(data: CheapData, cfg: Config) -> Result:
    df = pd.DataFrame({"m": pd.to_datetime(data.panel.meeting), "y": data.panel.target.astype(float)})
    dev = set(dev_meetings(data, cfg))
    rows = []
    for name, part in (("development", df[df.m.isin(dev)]), ("outer", df[~df.m.isin(dev)]), ("all", df)):
        if part.empty:
            continue
        mean = part.groupby("m")["y"].transform("mean")
        total = float(((part.y - part.y.mean()) ** 2).mean())
        between = float(((mean - part.y.mean()) ** 2).mean())
        rows.append({"period": name, "rows": len(part), "total_var": total, "between_share": between / total})
    era = df.assign(era=(df.m.dt.year // 5) * 5)
    by_era = []
    for e, part in era.groupby("era"):
        mean = part.groupby("m")["y"].transform("mean")
        total = float(((part.y - part.y.mean()) ** 2).mean())
        by_era.append({"era": int(e), "between_share": float(((mean - part.y.mean()) ** 2).mean()) / total})
    table = pd.DataFrame(rows)
    return Result({f"between_share_{r.period}": r.between_share for r in table.itertuples()},
                  {"by_period": table, "by_era": pd.DataFrame(by_era)},
                  "between_share = variance of meeting means / total variance. Not a signal/noise split.")


@check("A5", "Ordinary-day corpus vs Fed-day rows", "descriptive")
def a5(data: CheapData, cfg: Config) -> Result:
    p = data.panel
    dev = data.dev_rows(cfg)
    names = {RET: "ret", DLOGVOL: "dlogvol", RVOL21: "rvol21", RET_REL: "ret_rel", BETA63: "beta63"}
    rng = np.random.default_rng(cfg.seed)
    end = data.dev_end_pos(cfg)
    ok = data.observed[:, L:end] & data.ordinary_days()[None, L:end]
    f_idx, d_idx = np.nonzero(ok)
    pick = rng.choice(f_idx.size, size=min(200_000, f_idx.size), replace=False) if f_idx.size else np.array([], int)
    f_idx, d_idx = f_idx[pick], d_idx[pick] + L
    rows = []
    for c, name in names.items():
        fed = p.history[dev, -1, c].astype(np.float64)
        ordv = data.channels[f_idx, d_idx, c].astype(np.float64)
        for label, v in (("fed_rows_prior_day", fed), ("ordinary_firm_days", ordv)):
            v = v[np.isfinite(v)]
            rows.append({"channel": name, "population": label, "n": v.size,
                         **({f"p{q}": float(np.quantile(v, q / 100)) for q in (10, 50, 90)} if v.size else {})})
    cover = data.observed[f_idx[:, None], d_idx[:, None] + np.arange(-L + 1, 1)].mean(axis=1) if f_idx.size else []
    miss = pd.DataFrame([
        {"population": "fed_rows", "history_observed_share": float(p.history_mask[dev].mean())},
        {"population": "ordinary_firm_days", "history_observed_share": float(np.mean(cover)) if len(cover) else np.nan},
    ])
    return Result({"fed_history_observed_share": float(p.history_mask[dev].mean())},
                  {"channel_quantiles": pd.DataFrame(rows), "coverage": miss},
                  "Development period only. Ordinary sample: up to 200k observed firm-days, every firm ever in "
                  "the grid (not only index members). Fundamentals have no daily ordinary-day counterpart here.")


@check("A6", "Timing and label audit", "descriptive")
def a6(data: CheapData, cfg: Config) -> Result:
    p = data.panel
    firm, pos = data.row_index()
    rng = np.random.default_rng(cfg.seed)
    k = rng.choice(len(firm), size=min(5000, len(firm)), replace=False)
    checks = []

    def add(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    add("meeting_dates_are_trading_days", (pos >= 0).all(), f"{int((pos < 0).sum())} rows off-calendar")
    prior = data.ret[firm[k], pos[k] - 1]
    hist = p.history[k, -1, RET]
    same = np.isclose(hist, prior, atol=1e-4) | (np.isnan(hist) & np.isnan(prior))
    add("history_ends_prior_close", same.all(), f"{int((~same).sum())} of {k.size} sampled rows differ")
    grid_day = data.ret[firm[k], pos[k]]
    known = np.isfinite(grid_day)  # the grid masks zero-volume days; the panel target keeps their return
    tgt = np.isclose(p.target[k][known], grid_day[known], atol=1e-4)
    add("target_is_meeting_day_return", tgt.all(),
        f"{int((~tgt).sum())} of {int(known.sum())} comparable rows differ; {int((~known).sum())} meeting days "
        "masked in the grid (zero volume)")
    m = data.meeting_table()
    add("shock_constant_within_meeting",
        pd.Series(p.shock).groupby(pd.to_datetime(p.meeting)).nunique().max() == 1, "")
    if data.stmt is not None:
        j = m.merge(data.stmt.assign(announcement_date=pd.to_datetime(data.stmt["announcement_date"])),
                    left_on="meeting", right_on="announcement_date", how="left")
        ok = np.isclose(j["s"], 10.0 * j["STMT"], atol=1e-4)
        add("shock_is_10x_STMT", ok.all(), f"{int((~ok).sum())} meetings differ")
    if data.y1 is not None and "y1_level" in data.context_names:
        c = data.context_names.index("y1_level")
        lvl = pd.Series(p.context[:, c], index=pd.to_datetime(p.meeting)).groupby(level=0).first()
        y1 = data.y1.dropna().sort_index()
        prior_lvl = [float(y1[y1.index < d].iloc[-1]) if (y1.index < d).any() else np.nan for d in lvl.index]
        ok = np.isclose(lvl.to_numpy(), prior_lvl, atol=1e-4)
        add("context_y1_is_prior_day", ok.all(), f"{int((~ok).sum())} meetings differ")
    med = float(np.median(np.abs(10.0 * m["s"])))
    add("shock_units_plausible_bp", 0.2 <= med <= 10.0, f"median |surprise| {med:.2f}bp")
    sd = float(np.std(p.target))
    add("target_units_percent", 0.5 <= sd <= 6.0, f"target sd {sd:.2f}")
    table = pd.DataFrame(checks)
    return Result({"all_passed": bool(table["passed"].all()), "n_failed": int((~table["passed"]).sum())},
                  {"checks": table},
                  "Point-in-time fundamentals and membership are covered by core/tests/test_q3_real.py.")


@check("A7", "Pre-training window and label overlap audit", "descriptive")
def a7(data: CheapData, cfg: Config) -> Result:
    meetings = data.meetings()
    cut = {"inner_validation": meetings[min(cfg.inner_min_train, len(meetings) - 1)],
           "outer_test": meetings[min(cfg.outer_min_train, len(meetings) - 1)]}
    count = np.cumsum(data.observed, axis=1, dtype=np.int32)
    in_window = count[:, L - 1 :] - np.concatenate((np.zeros((count.shape[0], 1), np.int32), count[:, :-L]), axis=1)
    ends_all = np.arange(L - 1, data.calendar.size)
    rows = []
    for name, cutoff in cut.items():
        cpos = int(data.calendar.searchsorted(cutoff))
        stride_ok = (ends_all - (L - 1)) % 5 == 0
        firm, k = np.nonzero((in_window >= 126) & (ends_all < cpos)[None, :] & stride_ok[None, :])
        end = ends_all[k]
        for label, h in (("history_only", 0), ("same_day_response", 1), *((f"future_{x}d", x) for x in cfg.horizons)):
            last = end + h
            leaks = int((last >= cpos).sum())
            keep = last < cpos
            split = np.quantile(end[keep], 0.85) if keep.any() else 0
            val = keep & (end > split)
            train = keep & (end <= split)
            rows.append({
                "cutoff": name, "cutoff_date": cutoff, "label": label, "windows": int(end.size),
                "label_crosses_cutoff": leaks, "usable": int(keep.sum()),
                "val_windows_history_overlapping_train": int((val & (end - L + 1 <= split)).sum()),
                "train_windows_label_reaching_val": int((train & (end + h > split)).sum()),
            })
    table = pd.DataFrame(rows)
    return Result({"max_label_crossings": int(table["label_crosses_cutoff"].max())}, {"windows": table},
                  "Windows: stride 5, at least 126 observed days, ending before the cutoff. Labels that cross the "
                  "cutoff must be dropped. Validation = last 15% of window end dates; history overlap is expected "
                  "for 252-day windows. Train windows whose label reaches the validation period need an "
                  "embargo of the label horizon.")
