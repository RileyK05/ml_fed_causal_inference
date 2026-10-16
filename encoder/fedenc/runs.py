"""Fit Q3 arms on shared walk-forward folds and save one run per arm plus a comparison run.

``save_runs`` is training whenever a maker fits a network: it needs the user's go-ahead in chat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def b_grade(predictions: pd.DataFrame, shock: pd.Series) -> dict:
    """Within-meeting slope of the return on (b_hat - meeting mean) * s; 1 = calibrated b.

    ``shock`` maps meeting -> s. Meeting-clustered SE. Per seed, then averaged.
    """
    rows = []
    for seed, df in predictions.groupby("seed"):
        s = df["meeting"].map(shock).to_numpy(dtype=np.float64)
        yc = df["target"] - df.groupby("meeting")["target"].transform("mean")
        bc = df["b_hat"] - df.groupby("meeting")["b_hat"].transform("mean")
        x = (bc * s).to_numpy()
        sxx = float(x @ x)
        if sxx == 0:
            rows.append(dict(seed=seed, b_slope=np.nan, b_se=np.nan))
            continue
        slope = float(x @ yc.to_numpy()) / sxx
        score = pd.Series(x * (yc.to_numpy() - slope * x)).groupby(df["meeting"].to_numpy()).sum()
        rows.append(dict(seed=seed, b_slope=slope, b_se=float(np.sqrt((score ** 2).sum())) / sxx))
    out = pd.DataFrame(rows)
    return dict(b_slope=float(out.b_slope.mean()), b_se=float(out.b_se.mean()),
                b_t=float(out.b_slope.mean() / out.b_se.mean()))


def save_runs(panel, makers, *, prefix, params, min_train, test_size, seeds, store, datasets=None,
              role="exploratory"):
    """Fit each maker on the same folds; save runs; summary row per arm vs scratch."""
    from fedcore.q3.evaluate import compare, run_walk_forward
    from fedcore.q3.real import INPUTS
    from fedcore.questions import specs

    datasets = list(INPUTS) if datasets is None else list(datasets)
    shock = pd.Series(panel.shock, index=pd.DatetimeIndex(panel.meeting)).groupby(level=0).first()
    results, rows = {}, []
    for name, make in makers.items():
        result = run_walk_forward(panel, make, min_train=min_train, test_size=test_size, seeds=tuple(seeds))
        grade = b_grade(result.predictions, shock)
        saved = store.save(specs()["Q3"], f"{prefix}-{name}", role=role, datasets=datasets,
                           params=dict(arm=name, seeds=list(seeds), min_train=min_train,
                                       test_size=test_size, **params),
                           metrics={**result.metrics, **grade},
                           tables=dict(predictions=result.predictions, folds=result.folds, curves=result.curves))
        results[name] = result
        rows.append(dict(arm=name, run_id=saved.run_id, path=str(saved.path), mse=result.metrics["mse_per_meeting"],
                         rank_ic=result.metrics.get("rank_ic"), **grade))
        print(pd.DataFrame([rows[-1]]).to_string(index=False), flush=True)
    for row in rows:
        if row["arm"] != "tx_scratch" and "tx_scratch" in results:
            diff = compare(results[row["arm"]], results["tx_scratch"])
            row.update(mse_vs_scratch=diff["estimate"], ci_low=diff.get("ci_low"), ci_high=diff.get("ci_high"))
    summary = pd.DataFrame(rows)
    store.save(specs()["Q3"], f"{prefix}-comparison", role=role, params=params,
               tables={"comparison": summary})
    return summary
