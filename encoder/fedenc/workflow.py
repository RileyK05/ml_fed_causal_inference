"""Explicit plan, audit, pretrain and evaluation commands for the encoder project.

Plan/audit/folds never fit. Pretrain/evaluate require the user's training go-ahead in chat.
Nothing runs on import, and nothing uses the 2026 holdout.

Per-fold pretraining: ``folds`` lists each walk-forward fold's cutoff (its first inner
validation meeting); ``pretrain-folds`` pretrains one checkpoint per cutoff, so each fold's
encoder sees every ordinary day it is allowed to; ``evaluate`` takes all of them and each
fit uses the latest checkpoint that precedes its inner validation block.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from fedcore.data.loaders import fingerprint
from fedcore.protocol import walk_forward
from fedcore.q3.evaluate import compare, run_walk_forward
from fedcore.q3.train import seed_everything, validation_meetings
from fedcore.questions import specs
from fedcore.results import ResultStore
from fedenc.arms import make_arms
from fedenc.checkpoint import load_checkpoint, pick_checkpoint, save_checkpoint
from fedenc.config import RESULTS
from fedenc.encoder_transformer import PatchTransformerEncoder
from fedenc.pretrain_return import OBJECTIVES, RETURN_OBJECTIVES, pretrain_return
from fedenc.return_corpus import real_return_corpus

PRETRAIN_DATASETS = ("crsp_daily", "ff_factors_daily", "usmpd_statements")
ADAPTS = ("full", "staged", "frozen", "top")
VAL_FRAC = 0.15  # the arms' inner validation fraction; fold cutoffs must match it


def corpus_audit(corpus, val_frac):
    train, val = corpus.split(val_frac, purge_history=True)
    return dict(
        n_rows=len(corpus), n_target_dates=len(corpus.target_dates.unique()),
        n_train_rows=len(train), n_val_rows=len(val),
        n_purged_rows=len(corpus) - len(train) - len(val),
        n_train_dates=len(corpus.target_dates[train].unique()),
        n_val_dates=len(corpus.target_dates[val].unique()),
        purge_history=True,
        first_history_end=str(corpus.history_end_dates.min().date()),
        train_target_end=str(corpus.target_dates[train].max().date()),
        val_target_start=str(corpus.target_dates[val].min().date()),
        available_through=str(corpus.target_dates.max().date()),
        cutoff=str(corpus.source.cutoff.date()),
        excluded_fomc_dates=len(corpus.excluded_dates),
        inputs={name: fingerprint(name) for name in PRETRAIN_DATASETS},
    )


def bounded_panel(through):
    """The real panel through an inclusive historical date; never the 2026 holdout."""
    from fedcore.q3.real import load_real_panel

    through = pd.Timestamp(through)
    if pd.isna(through) or through >= pd.Timestamp("2026-01-01"):
        raise ValueError("through must be a historical date before the separate 2026 holdout")
    panel, _, _ = load_real_panel()
    return panel.subset(np.flatnonzero(panel.meeting <= through.to_datetime64())), through


def fold_cutoffs(panel, min_train, test_size, val_frac=VAL_FRAC):
    """Per fold: the first inner validation meeting, which is that fold's pretraining cutoff."""
    folds = walk_forward(panel.meetings(), min_train, test_size)
    if not folds:
        raise ValueError("no evaluation folds inside the requested meeting boundary")
    rows = []
    for fold in folds:
        part = panel.subset(np.flatnonzero(np.isin(panel.meeting, fold.train)))
        rows.append(dict(fold=fold.k, cutoff=validation_meetings(part, val_frac).min(),
                         first_test=fold.test.min(), n_train_meetings=len(fold.train)))
    return pd.DataFrame(rows)


def _checkpoint_set(paths):
    items = paths if isinstance(paths, (list, tuple)) else [paths]
    cps = [load_checkpoint(path) for path in items]
    if not cps:
        raise ValueError("no checkpoints given")
    if len({cp.params["objective"] for cp in cps}) != 1 or any(cp.config != cps[0].config for cp in cps):
        raise ValueError("checkpoints in one set must share objective and architecture")
    if len({cp.available_through for cp in cps}) != len(cps):
        raise ValueError("checkpoints in one set must have distinct cutoffs")
    return cps


def evaluate_checkpoint(checkpoint, *, through, masked_checkpoint=None, min_train=60, test_size=16, seeds=(0, 1, 2),
                        arms=("tx_scratch", "tx_pre_full"), baselines=(), max_epochs=40, patience=10,
                        device=None, store=None):
    """Evaluate only explicitly bounded historical meetings; never select using holdout.

    ``checkpoint`` / ``masked_checkpoint``: one path or a list (one per fold).
    Availability is checked against every fold's inner validation before any fit.
    ``baselines`` (names from ``fedcore.q3.baselines.BASELINES``) run on the same folds.
    Model weights, scalers, predictions and curves are retained through ResultStore.
    """
    from fedcore.q3.real import INPUTS

    if min_train < 2 or test_size < 1 or max_epochs < 1 or patience < 1:
        raise ValueError("invalid evaluation fold/epoch/patience settings")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("choose distinct seeds")
    cps = _checkpoint_set(checkpoint)
    if cps[0].params["objective"] not in RETURN_OBJECTIVES:
        raise ValueError("the main checkpoint set must use a next-return objective")
    masked = _checkpoint_set(masked_checkpoint) if masked_checkpoint is not None else None
    if masked is not None:
        if masked[0].params["objective"] != OBJECTIVES[1]:
            raise ValueError("the comparison set must use masked reconstruction")
        if masked[0].config != cps[0].config:
            raise ValueError("comparison checkpoints must use identical encoder architectures")
    for cp in cps + (masked or []):
        for name, old in cp.fingerprints.items():
            if fingerprint(name) != old:
                raise ValueError(f"checkpoint input changed: {name}; use a matching data version")
    panel, through = bounded_panel(through)
    cutoffs = fold_cutoffs(panel, min_train, test_size)
    used = []
    for row in cutoffs.itertuples(index=False):
        cp = pick_checkpoint(cps, row.cutoff)
        entry = dict(fold=row.fold, cutoff=row.cutoff, checkpoint=str(cp.path),
                     available_through=cp.available_through)
        if masked is not None:
            mcp = pick_checkpoint(masked, row.cutoff)
            if any(not np.array_equal(mcp.scaler[key], cp.scaler[key]) for key in cp.scaler):
                raise ValueError("comparison checkpoints must share history normalization; "
                                 "use matching corpus/split/scaler seed")
            entry["masked_checkpoint"] = str(mcp.path)
        used.append(entry)
    budget = dict(max_epochs=max_epochs, patience=patience, device=device)
    factories = make_arms(cps, train_kwargs=budget)
    if masked is not None:
        masked_arms = make_arms(masked, train_kwargs=budget)
        factories.update({f"tx_masked_{adapt}": masked_arms[f"tx_pre_{adapt}"] for adapt in ADAPTS})
    if baselines:
        from fedcore.q3.baselines import BASELINES

        if len(set(baselines)) != len(baselines) or set(baselines) - BASELINES.keys():
            raise ValueError(f"choose distinct baselines from {list(BASELINES)}")
    if not arms or len(set(arms)) != len(arms) or set(arms) - factories.keys():
        raise ValueError(f"choose distinct arms from {list(factories)}")
    store = store or ResultStore(RESULTS)
    results, run_ids = {}, {}
    fold_args = dict(min_train=min_train, test_size=test_size, seeds=tuple(seeds))
    for name in baselines:
        result = run_walk_forward(panel, BASELINES[name], **fold_args)
        run = store.save(
            specs()["Q3"], f"baseline-{name}", role="exploratory", datasets=list(INPUTS),
            params=dict(arm=name, seeds=list(seeds), min_train=min_train, test_size=test_size,
                        through=str(through.date()), n_meetings=len(panel.meetings())),
            metrics=result.metrics, tables=dict(predictions=result.predictions, folds=result.folds,
                                                by_seed=result.by_seed),
            notes="Tabular baseline on the same walk-forward folds as the encoder arms. No 2026 holdout.",
        )
        results[name], run_ids[name] = result, run.run_id
        print(f"saved {name}: {run.path}", flush=True)
    for name in arms:
        fitted = []
        arm_set = masked if name.startswith("tx_masked_") else cps

        def factory(seed):
            arm = factories[name](seed)
            fitted.append((seed, arm))
            return arm

        result = run_walk_forward(panel, factory, **fold_args)
        state_rows, scaler_rows, stage_rows = [], [], []
        for ordinal, (seed, arm) in enumerate(fitted):
            for key, tensor in arm.model.state_dict().items():
                state_rows.append(dict(fit=ordinal, seed=seed, name=key,
                                       shape=json.dumps(list(tensor.shape)),
                                       values=json.dumps(tensor.cpu().reshape(-1).tolist())))
            for field in ("hist_median", "hist_iqr", "fund_median", "fund_iqr", "ctx_median", "ctx_iqr"):
                scaler_rows.append(dict(fit=ordinal, seed=seed, name=field,
                                        values=json.dumps(getattr(arm.scaler, field).tolist())))
            stages = getattr(arm.fit_result, "stage_epochs", None)
            loaded = arm.loaded_checkpoint
            stage_rows.append(dict(fit=ordinal, seed=seed,
                                   checkpoint=None if loaded is None else str(loaded.path),
                                   stage_epochs=json.dumps(stages),
                                   best_epoch=getattr(arm.fit_result, "best_epoch", None)))
        run = store.save(
            specs()["Q3"], name, role="exploratory", datasets=list(INPUTS),
            params=dict(arm=name, seeds=list(seeds), min_train=min_train, test_size=test_size,
                        through=str(through.date()), checkpoint_runs=[str(cp.path) for cp in arm_set],
                        checkpoint_digests=[cp.digest for cp in arm_set],
                        objective=arm_set[0].params["objective"],
                        checkpoint_available_through=[str(cp.available_through.date()) for cp in arm_set],
                        uses_pretrained_weights=name != "tx_scratch",
                        encoder_config=cps[0].config, max_epochs=max_epochs, patience=patience,
                        device=device, n_meetings=len(panel.meetings()),
                        n_fund=panel.fundamentals.shape[1], n_ctx=panel.context.shape[1]),
            metrics=result.metrics, tables=dict(predictions=result.predictions, folds=result.folds,
                                                curves=result.curves, by_seed=result.by_seed,
                                                **{"response-state": pd.DataFrame(state_rows),
                                                   "response-scalers": pd.DataFrame(scaler_rows),
                                                   "fits": pd.DataFrame(stage_rows)}),
            notes="Raw Fed-day return target; mu=a+b*s; shared meeting-weighted MSE. "
                  "Each fit uses the latest checkpoint before its inner validation block. No 2026 holdout. "
                  "fit ordinal maps to folds table row order. Paired CI is the shared meeting bootstrap.",
        )
        results[name], run_ids[name] = result, run.run_id
        print(f"saved {name}: {run.path}", flush=True)
    rows = []
    for name, result in results.items():
        row = dict(arm=name, run_id=run_ids[name], **result.metrics)
        for ref in ("tx_scratch", "pooled"):
            if ref in results and name != ref:
                row.update({f"mse_minus_{ref}_{key}": value
                            for key, value in compare(result, results[ref]).items()})
        rows.append(row)
    summary = pd.DataFrame(rows)
    store.save(specs()["Q3"], "encoder-transfer-comparison", role="exploratory",
               params=dict(through=str(through.date()), checkpoint_digests=[cp.digest for cp in cps], runs=run_ids),
               tables={"comparison": summary, "checkpoint-by-fold": pd.DataFrame(used)})
    return summary


def pretrain_one(cutoff, args):
    """Pretrain and save one checkpoint whose labels all precede ``cutoff``."""
    cutoff = pd.Timestamp(cutoff)
    if pd.isna(cutoff) or cutoff > pd.Timestamp("2026-01-01"):
        raise ValueError("pretraining cutoff must exclude the 2026 holdout")
    corpus = real_return_corpus(cutoff, stride=args.stride, min_observed=args.min_observed)
    print(json.dumps(corpus_audit(corpus, args.val_frac), indent=2), flush=True)
    if args.command == "audit":
        return None
    seed_everything(args.seed)
    result = pretrain_return(PatchTransformerEncoder(), corpus, objective=args.objective,
                             epochs=args.epochs, lr=args.lr, seed=args.seed, val_frac=args.val_frac,
                             batch_size=args.batch_size, dates_per_step=args.dates_per_step,
                             patience=args.patience, scaler_windows=args.scaler_windows,
                             train_eval_dates=args.train_eval_dates,
                             device=args.device, verbose=True)
    result.params.update(stride=args.stride, min_observed=args.min_observed)
    run = save_checkpoint(result, ResultStore(RESULTS), datasets=PRETRAIN_DATASETS)
    print(f"checkpoint run: {run.path}", flush=True)
    return run


def _fold_args(p):
    p.add_argument("--through", required=True, help="inclusive historical evaluation boundary")
    p.add_argument("--min-train", type=int, default=60)
    p.add_argument("--test-size", type=int, default=16)


def _pretrain_args(p):
    p.add_argument("--objective", choices=OBJECTIVES, default=OBJECTIVES[0])
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--min-observed", type=int, default=126)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--dates-per-step", type=int, default=4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--scaler-windows", type=int, default=4096)
    p.add_argument("--train-eval-dates", type=int, default=250, help="training dates scored for the train-loss curve")
    p.add_argument("--device")


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    for command in ("plan", "audit", "pretrain"):
        p = sub.add_parser(command)
        p.add_argument("--cutoff", required=True, help="exclusive date boundary for next-day labels")
        _pretrain_args(p)
    p = sub.add_parser("folds", help="print each fold's pretraining cutoff; loads the panel, fits nothing")
    _fold_args(p)
    p = sub.add_parser("pretrain-folds", help="one checkpoint per fold cutoff")
    _fold_args(p)
    _pretrain_args(p)
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True, nargs="+", help="saved pretraining run directories (one per fold)")
    p.add_argument("--masked-checkpoint", nargs="+", help="optional masked comparison set (one per fold)")
    _fold_args(p)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--arms", nargs="+", default=["tx_scratch", "tx_pre_full"])
    p.add_argument("--baselines", nargs="*", default=[], help="tabular baselines on the same folds, e.g. pooled ridge_interact")
    p.add_argument("--max-epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--device")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "plan":
        nextday = args.objective in RETURN_OBJECTIVES
        loss = {OBJECTIVES[0]: "log(v) + (r_next - m)^2/v",
                OBJECTIVES[2]: "(r_next - m)^2 / naive trailing variance"}.get(
            args.objective, "observed masked-patch MSE, balanced across channels")
        print(json.dumps({**vars(args), "architecture": PatchTransformerEncoder().get_config(),
                          "target": "next trading day's raw percent return" if nextday else "masked observed history entries",
                          "loss": loss,
                          "excluded": "all FOMC target dates",
                          "transfer": "encoder + history scaler; discard temporary head",
                          "purge_history": True,
                          "results": str(RESULTS), "loads_data": False, "fits": False}, indent=2))
        return
    if args.command in ("folds", "pretrain-folds"):
        panel, _ = bounded_panel(args.through)
        cutoffs = fold_cutoffs(panel, args.min_train, args.test_size)
        print(cutoffs.to_string(index=False), flush=True)
        if args.command == "pretrain-folds":
            for cutoff in cutoffs.cutoff:
                pretrain_one(cutoff, args)
        return
    if args.command == "evaluate":
        print(evaluate_checkpoint(args.checkpoint, through=args.through, min_train=args.min_train,
                                  masked_checkpoint=args.masked_checkpoint,
                                  test_size=args.test_size, seeds=args.seeds, arms=args.arms,
                                  baselines=args.baselines,
                                  max_epochs=args.max_epochs, patience=args.patience,
                                  device=args.device).to_string(index=False))
        return
    pretrain_one(args.cutoff, args)


if __name__ == "__main__":
    main()
