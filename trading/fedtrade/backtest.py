"""Per-meeting trade table and its summary, from frozen Q3 predictions and post-meeting CAR.

    python -m fedtrade.backtest dev --no-save     # development meetings (2001-2021), print only

Saving a run needs the user's go-ahead (AGENTS.md). No fitting happens here.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from fedtrade.books import DECILE, LEGS, STRATEGIES, trade_returns
from fedtrade.signals import load_q3, rate_sensitivity

BIG = 0.25           # |s| >= 2.5bp (10bp units), the big-meeting slice used throughout Q3
DRIFT_DAYS = 20
BOOT = 5000


def trades(pred: pd.DataFrame, shock: pd.Series, car: pd.Series, sector: pd.Series | None = None,
           min_abs_s: float = BIG, q: float = DECILE) -> pd.DataFrame:
    """One row per meeting with |s| >= ``min_abs_s``: s and each strategy's return (%)."""
    db = rate_sensitivity(pred, shock)
    car = car.reindex(db.index)
    sec = sector.reindex(db.index) if sector is not None else None
    rows = []
    for m in db.index.unique(0):
        s = float(shock.get(m, np.nan))
        if not np.isfinite(s) or abs(s) < min_abs_s:
            continue
        r = trade_returns(db.loc[m], car.loc[m], s, None if sec is None else sec.loc[m], q)
        rows.append(dict(meeting=m, s=s, **r))
    return pd.DataFrame(rows).set_index("meeting") if rows else pd.DataFrame(columns=["s", *STRATEGIES])


def summary(table: pd.DataFrame, cost_pct: float = 0.0, n_boot: int = BOOT, seed: int = 0) -> pd.DataFrame:
    """Per strategy and surprise direction: trades, mean, median, win rate, t, meeting-bootstrap
    95% interval of the mean and worst trade, % per trade. ``cost_pct``: round-trip cost per stock
    leg, subtracted once per leg traded."""
    rng = np.random.default_rng(seed)
    out = []
    for strat in [c for c in STRATEGIES if c in table]:
        for label, msk in (("all", table["s"] != 0), ("dovish", table["s"] < 0), ("hawkish", table["s"] > 0)):
            x = (table.loc[msk, strat] - cost_pct * LEGS[strat]).dropna().to_numpy()
            if not len(x):
                continue
            boot = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
            sd = x.std(ddof=1) if len(x) > 1 else np.nan
            out.append(dict(strategy=strat, meetings=label, trades=len(x), mean=x.mean(), median=np.median(x),
                            win=(x > 0).mean(), t=x.mean() / sd * np.sqrt(len(x)) if sd and sd > 0 else np.nan,
                            lo=np.quantile(boot, 0.025), hi=np.quantile(boot, 0.975), worst=x.min()))
    return pd.DataFrame(out)


def dev_inputs(prefix: str = "v6-to2021", arm: str = "F0s", end: str = "2021-12-31"):
    """Development predictions, surprises, 20-day CAR and GICS sector. Builds the licensed panel locally."""
    from fedcore.q3.channels import abnormal_returns
    from fedcore.q3.drift import car_table
    from fedcore.q3.real import _grid, load_real_panel, read_inputs

    panel, rows, _ = load_real_panel()
    base = read_inputs()
    grid = _grid(base.daily, base.market)
    car = car_table(rows, abnormal_returns(grid), grid["calendar"], grid["permnos"], DRIFT_DAYS, end)
    shock = pd.Series(panel.shock, index=pd.DatetimeIndex(panel.meeting)).groupby(level=0).first()
    sector = pd.Series(rows["gsector"].astype(str).to_numpy(), index=car.index)
    pred, run = load_q3(prefix, arm)
    return pred, shock, car, sector, run


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fedtrade.backtest")
    ap.add_argument("period", choices=["dev"])
    ap.add_argument("--prefix", default="v6-to2021")
    ap.add_argument("--arm", default="F0s")
    ap.add_argument("--cost", type=float, default=0.0, help="round-trip cost per stock leg, %%")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args(argv)
    pred, shock, car, sector, run = dev_inputs(a.prefix, a.arm)
    t = trades(pred, shock, car, sector)
    res = summary(t, a.cost)
    pd.set_option("display.width", 200)
    print(f"Q3 run {run}; {len(t)} big meetings; 20-day CAR, % per trade, cost {a.cost}% per leg")
    print(res.round(2).to_string(index=False))
    if not a.no_save:
        from fedcore.questions import specs
        from fedcore.results import ResultStore
        from fedtrade.config import RESULTS

        ResultStore(RESULTS).save(specs()["Q5"], f"dev-{a.prefix}-{a.arm}", role="exploratory",
                                  params=dict(prefix=a.prefix, arm=a.arm, cost=a.cost, q=DECILE, min_abs_s=BIG,
                                              drift_days=DRIFT_DAYS, q3_run=run),
                                  tables={"trades": t.reset_index(), "summary": res},
                                  datasets=["crsp_daily", "comp_gics"],
                                  notes="Rule books on frozen Q3 predictions, development meetings.")


if __name__ == "__main__":
    main()
