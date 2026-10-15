"""Experiment: pretrain the full response model on ordinary days, then post-train on Fed days.

Same model, same formula, same target every day: mu = a(z) + b(z) * s, scored against
that day's raw return. On ordinary days there is no Fed shock, so s = 0 and only a (and
everything feeding it) learns. Post-training on Fed days keeps every weight and the
pretraining scaler; b turns on because s is nonzero. Nothing is discarded.

Ordinary-day rows are built with the Fed panel's own builder: same S&P members, history
channels, point-in-time fundamentals and context (1-year yield level/changes to the prior
close, the last three statement surprises before the day, 21-day market volatility).
All statement dates, scheduled or not, are excluded. Each day weighs the same in the loss,
like each meeting does on Fed days.

Per fold the pretraining uses only ordinary days before that fold's first inner validation
meeting. Exploratory: if it does not beat scratch, delete this module.

    python -m fedenc.daily run --through 2009-03-18 --every 10 --seeds 0 --device cuda

``run`` is training and needs the user's go-ahead in chat.
"""
from __future__ import annotations

import argparse
import dataclasses
import math

import numpy as np
import pandas as pd
import torch

from fedcore.data import load
from fedcore.q3 import ResponseModel
from fedcore.q3.train import (
    ChannelScaler, _forward, _meeting_index_groups, _meeting_rows, _tensors, _train_epoch,
    predict as train_predict, resolve_device, seed_everything, validation_meetings,
)

EVAL_CHUNK = 4096  # rows per evaluation pass; the shared trainer scores everything at once
from fedenc.encoder_transformer import PatchTransformerEncoder


def ordinary_day_table(dates) -> pd.DataFrame:
    """``meeting_table`` columns for ordinary days: s = 0, context as of the prior close."""
    mps = load("mps_surprises").rename(columns={"Date": "event"}).sort_values("event")
    hist = pd.DataFrame({"event": pd.to_datetime(mps["event"]), "st_prev": mps["STMT"].to_numpy(),
                         "me_prev": mps["ME"].to_numpy()})
    hist["st_mean3"] = hist["st_prev"].rolling(3).mean()  # the last three events, inclusive
    y1 = load("treasury_1y").rename(columns={"Date": "date", "SVENY01": "y1_level"}).sort_values("date")
    y1["date"] = pd.to_datetime(y1["date"])
    y1["y1_chg_21d"] = y1["y1_level"] - y1["y1_level"].shift(21)
    y1["y1_chg_63d"] = y1["y1_level"] - y1["y1_level"].shift(63)
    df = pd.DataFrame({"announcement_date": pd.DatetimeIndex(dates).sort_values()})
    # Strictly before the day, exactly as the Fed-day table uses the previous event.
    df = pd.merge_asof(df, hist, left_on="announcement_date", right_on="event",
                       allow_exact_matches=False).drop(columns="event")
    df = pd.merge_asof(df, y1[["date", "y1_level", "y1_chg_21d", "y1_chg_63d"]],
                       left_on="announcement_date", right_on="date", allow_exact_matches=False).drop(columns="date")
    df["s"] = 0.0
    return df


def ordinary_dates(market_dates, statement_dates, *, start, end, every):
    """Every ``every``-th trading day in [start, end), skipping all statement dates."""
    days = pd.DatetimeIndex(pd.to_datetime(market_dates)).sort_values().unique()
    days = days[(days >= pd.Timestamp(start)) & (days < pd.Timestamp(end))]
    days = days[~days.isin(pd.DatetimeIndex(pd.to_datetime(statement_dates)).normalize())]
    return days[::every]


def build_daily_panel(through, every=10):
    """Ordinary-day panel (s = 0) from 1994 to ``through``, every ``every``-th trading day."""
    from fedcore.q3.real import build_panel, read_inputs

    inputs = read_inputs()
    statements = pd.to_datetime(load("usmpd_statements")["Date"])
    dates = ordinary_dates(pd.to_datetime(inputs.market["date"]), statements,
                           start="1994-01-01", end=pd.Timestamp(through), every=every)
    panel, _, notes = build_panel(dataclasses.replace(inputs, meetings=ordinary_day_table(dates)))
    return panel, notes


def _subset_before(panel, cutoff):
    return panel.subset(np.flatnonzero(panel.meeting < np.datetime64(pd.Timestamp(cutoff))))


def _eval_chunked(model, batch, chunk=EVAL_CHUNK) -> float:
    """Mean over days/meetings of the within-day MSE, scored ``chunk`` rows at a time."""
    model.eval()
    codes = batch["codes"]
    width = int(codes.max().item()) + 1
    sse = torch.zeros(width, dtype=torch.float64, device=codes.device)
    count = torch.zeros(width, dtype=torch.float64, device=codes.device)
    with torch.no_grad():
        for start in range(0, codes.numel(), chunk):
            rows = slice(start, start + chunk)
            mu, _, _ = _forward(model, batch, rows)
            err = (mu.double() - batch["target"][rows].double()) ** 2
            sse.index_add_(0, codes[rows], err)
            count.index_add_(0, codes[rows], torch.ones_like(err))
    model.train()
    keep = count > 0
    return float((sse[keep] / count[keep]).mean().item())


def _loop(model, optimizer, panel, train_m, val_m, scaler, *, max_epochs, patience, seed, meetings_per_batch=4):
    """Meeting-batched epochs with chunked evaluation and early stopping on ``val_m``."""
    device = next(model.parameters()).device
    train_part = panel.subset(_meeting_rows(panel, train_m))
    train_batch = _tensors(train_part, scaler, device)
    val_batch = _tensors(panel.subset(_meeting_rows(panel, val_m)), scaler, device)
    rng = np.random.default_rng(seed)
    index = _meeting_index_groups(np.asarray(train_part.meeting))
    best_val, best_epoch, stale = math.inf, 0, 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    train_curve, val_curve = [], []
    for epoch in range(1, max_epochs + 1):
        _train_epoch(model, optimizer, train_batch, index, rng, meetings_per_batch, len(index), False)
        train_curve.append(_eval_chunked(model, train_batch))
        val_curve.append(_eval_chunked(model, val_batch))
        if val_curve[-1] < best_val:
            best_val, best_epoch, stale = val_curve[-1], epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return argparse.Namespace(train_loss=train_curve, val_loss=val_curve, best_epoch=best_epoch,
                              best_val_loss=best_val, train_meetings=pd.DatetimeIndex(train_m),
                              val_meetings=pd.DatetimeIndex(val_m))


def pretrain(model, daily, *, lr=1e-3, weight_decay=1e-4, max_epochs=20, patience=3, val_frac=0.15, seed=0):
    """Ordinary-day training of the whole model (s = 0); the scaler fits on training days only."""
    seed_everything(seed)
    val_m = validation_meetings(daily, val_frac)
    train_m = daily.meetings()[:-len(val_m)]
    scaler = ChannelScaler.fit(daily.subset(_meeting_rows(daily, train_m)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    result = _loop(model, optimizer, daily, train_m, val_m, scaler, max_epochs=max_epochs,
                   patience=patience, seed=seed)
    result.scaler = scaler
    return result


def post_train(model, panel, scaler, *, lr=1e-3, pretrained_lr=1e-4, weight_decay=1e-4,
               max_epochs=40, patience=10, val_frac=0.15, seed=0):
    """Fed-day training of a pretrained model with the pretraining scaler kept fixed.

    The b head never trained (s was 0), so it gets ``lr``; every pretrained weight gets
    ``pretrained_lr``. Early stopping on the last Fed meetings, as in the shared trainer.
    """
    seed_everything(seed)
    val_m = validation_meetings(panel, val_frac)
    train_m = panel.meetings()[:-len(val_m)]
    b_ids = {id(p) for p in model.head.b.parameters()}
    groups = [
        {"params": [p for p in model.parameters() if id(p) in b_ids], "lr": lr, "weight_decay": weight_decay},
        {"params": [p for p in model.parameters() if id(p) not in b_ids], "lr": pretrained_lr,
         "weight_decay": weight_decay},
    ]
    optimizer = torch.optim.AdamW(groups)
    return _loop(model, optimizer, panel, train_m, val_m, scaler, max_epochs=max_epochs,
                 patience=patience, seed=seed)


class DailyArm:
    """Pretrain on ordinary days before this fit's inner validation block, then post-train."""

    def __init__(self, daily_panel, seed=0, device=None, pretrain_epochs=20, pretrain_patience=3,
                 max_epochs=40, patience=10):
        self.daily_panel = daily_panel
        self.seed = int(seed)
        self.device = device
        self.pretrain_epochs, self.pretrain_patience = pretrain_epochs, pretrain_patience
        self.max_epochs, self.patience = max_epochs, patience
        self.model = self.scaler = self.fit_result = self.pretrain_result = None

    def fit(self, panel, seed=None, **_):
        seed = self.seed if seed is None else int(seed)
        cutoff = validation_meetings(panel, 0.15).min()
        daily = _subset_before(self.daily_panel, cutoff)
        if len(daily.meetings()) < 10:
            raise ValueError("fewer than 10 ordinary days before the fold's inner validation")
        assert pd.Timestamp(daily.meetings().max()) < pd.Timestamp(cutoff)
        seed_everything(seed)
        self.model = ResponseModel(PatchTransformerEncoder(), n_fund=panel.fundamentals.shape[1],
                                   n_ctx=panel.context.shape[1]).to(resolve_device(self.device))
        self.pretrain_result = pretrain(self.model, daily, max_epochs=self.pretrain_epochs,
                                        patience=self.pretrain_patience, seed=seed)
        self.scaler = self.pretrain_result.scaler
        self.fit_result = post_train(self.model, panel, self.scaler, max_epochs=self.max_epochs,
                                     patience=self.patience, seed=seed)
        self.cutoff = cutoff
        print(f"  fold cutoff {pd.Timestamp(cutoff).date()} seed {seed}: pretrain best epoch "
              f"{self.pretrain_result.best_epoch} on {len(daily.meetings())} days; post-train best epoch "
              f"{self.fit_result.best_epoch}", flush=True)

    def predict(self, panel):
        if self.model is None:
            raise RuntimeError("fit the arm before predict")
        return train_predict(self.model, panel, self.scaler)


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


def run(through="2009-03-18", every=10, seeds=(0,), min_train=60, test_size=16, device=None,
        baselines=("pooled",), store=None):
    """Daily-pretrained arm vs scratch (and baselines) on the same Fed-day folds. Training."""
    from fedcore.q3.baselines import BASELINES
    from fedcore.q3.evaluate import compare, run_walk_forward
    from fedcore.questions import specs
    from fedcore.results import ResultStore
    from fedcore.q3.real import INPUTS
    from fedenc.arms import TxArm
    from fedenc.config import RESULTS
    from fedenc.workflow import bounded_panel

    panel, through = bounded_panel(through)
    daily, notes = build_daily_panel(through, every=every)
    print(f"daily panel: {notes['n_rows']} rows, {notes['n_meetings']} days, "
          f"{notes['first_meeting']} .. {notes['last_meeting']}", flush=True)
    shock = pd.Series(panel.shock, index=pd.DatetimeIndex(panel.meeting)).groupby(level=0).first()
    store = store or ResultStore(RESULTS)
    makers = {
        "tx_daily": lambda seed: DailyArm(daily, seed=seed, device=device),
        "tx_scratch": lambda seed: TxArm(seed=seed, init="scratch", device=device, max_epochs=40, patience=10),
        **{name: BASELINES[name] for name in baselines},
    }
    results, rows = {}, []
    for name, make in makers.items():
        result = run_walk_forward(panel, make, min_train=min_train, test_size=test_size, seeds=tuple(seeds))
        grade = b_grade(result.predictions, shock)
        saved = store.save(specs()["Q3"], f"daily-exp-{name}", role="exploratory", datasets=list(INPUTS),
                           params=dict(arm=name, every=every, seeds=list(seeds), through=str(through.date()),
                                       min_train=min_train, test_size=test_size, daily_rows=notes["n_rows"],
                                       daily_days=notes["n_meetings"]),
                           metrics={**result.metrics, **grade},
                           tables=dict(predictions=result.predictions, folds=result.folds, curves=result.curves))
        results[name] = result
        row = dict(arm=name, run_id=saved.run_id, mse=result.metrics["mse_per_meeting"],
                   rank_ic=result.metrics.get("rank_ic"), **grade)
        if name != "tx_scratch" and "tx_scratch" in results:
            diff = compare(result, results["tx_scratch"])
            row.update(mse_vs_scratch=diff["estimate"], ci_low=diff.get("ci_low"), ci_high=diff.get("ci_high"))
        rows.append(row)
        print(pd.DataFrame([row]).to_string(index=False), flush=True)
    summary = pd.DataFrame(rows)
    store.save(specs()["Q3"], "daily-exp-comparison", role="exploratory",
               params=dict(every=every, seeds=list(seeds), through=str(through.date())),
               tables={"comparison": summary})
    return summary


def _compare_and_save(panel, makers, *, prefix, params, min_train, test_size, seeds, store, datasets=None):
    """Fit each maker on the same folds; save runs; summary row per arm vs scratch."""
    from fedcore.q3.evaluate import compare, run_walk_forward
    from fedcore.q3.real import INPUTS

    datasets = list(INPUTS) if datasets is None else list(datasets)
    from fedcore.questions import specs

    shock = pd.Series(panel.shock, index=pd.DatetimeIndex(panel.meeting)).groupby(level=0).first()
    results, rows = {}, []
    for name, make in makers.items():
        result = run_walk_forward(panel, make, min_train=min_train, test_size=test_size, seeds=tuple(seeds))
        grade = b_grade(result.predictions, shock)
        saved = store.save(specs()["Q3"], f"{prefix}-{name}", role="exploratory", datasets=datasets,
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
    store.save(specs()["Q3"], f"{prefix}-comparison", role="exploratory", params=params,
               tables={"comparison": summary})
    return summary


def run_inputs(through="2009-03-18", seeds=(0, 1, 2), min_train=60, test_size=16, device=None, store=None):
    """Does the model use fundamentals? Scratch vs wider branches vs no price history. Training."""
    from fedcore.q3.baselines import BASELINES
    from fedcore.results import ResultStore
    from fedenc.arms import TxArm
    from fedenc.config import RESULTS
    from fedenc.workflow import bounded_panel

    panel, through = bounded_panel(through)
    budget = dict(device=device, max_epochs=40, patience=10)
    makers = {
        "tx_scratch": lambda seed: TxArm(seed=seed, **budget),
        "tx_branch32": lambda seed: TxArm(seed=seed, branch_dim=32, **budget),
        "tx_no_history": lambda seed: TxArm(seed=seed, no_history=True, **budget),
        "pooled": BASELINES["pooled"],
    }
    return _compare_and_save(panel, makers, prefix="inputs-exp", params=dict(through=str(through.date())),
                             min_train=min_train, test_size=test_size, seeds=seeds,
                             store=store or ResultStore(RESULTS))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("inputs", help="scratch input/width variants vs scratch and pooled (training)")
    q.add_argument("--through", default="2009-03-18")
    q.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    q.add_argument("--device")
    for command in ("build", "run"):
        q = sub.add_parser(command)
        q.add_argument("--through", default="2009-03-18")
        q.add_argument("--every", type=int, default=10, help="keep every n-th ordinary trading day")
        if command == "run":
            q.add_argument("--seeds", type=int, nargs="+", default=[0])
            q.add_argument("--baselines", nargs="*", default=["pooled"])
            q.add_argument("--device")
    args = p.parse_args(argv)
    if args.command == "inputs":
        print(run_inputs(args.through, seeds=tuple(args.seeds), device=args.device).to_string(index=False))
        return
    if args.command == "build":  # builds the ordinary-day panel and reports it; fits nothing
        panel, notes = build_daily_panel(args.through, every=args.every)
        print({k: notes[k] for k in ("n_rows", "n_meetings", "n_firms", "first_meeting", "last_meeting")})
        print("history GB:", round(panel.history.nbytes / 1e9, 2))
        return
    print(run(args.through, every=args.every, seeds=tuple(args.seeds), device=args.device,
              baselines=tuple(args.baselines)).to_string(index=False))


if __name__ == "__main__":
    main()
