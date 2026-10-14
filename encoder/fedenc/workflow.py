"""Explicit plan, audit, pretrain and evaluation commands for the encoder project.

Plan/audit never fit. Pretrain/evaluate require the user's training go-ahead in chat.
Nothing runs on import, and nothing uses the 2026 holdout.
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
from fedenc.arms import TxArm, make_arms
from fedenc.checkpoint import load_checkpoint, save_checkpoint
from fedenc.config import RESULTS
from fedenc.encoder_transformer import PatchTransformerEncoder
from fedenc.pretrain_return import OBJECTIVES, pretrain_return
from fedenc.return_corpus import real_return_corpus

PRETRAIN_DATASETS = ("crsp_daily", "ff_factors_daily", "usmpd_statements")


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


def evaluate_checkpoint(checkpoint, *, through, masked_checkpoint=None, min_train=60, test_size=16, seeds=(0, 1, 2),
                        arms=("tx_scratch", "tx_pre_full"), max_epochs=40, patience=10,
                        device=None, store=None):
    """Evaluate only explicitly bounded historical meetings; never select using holdout.

    Availability is checked against every fold's inner validation before any fit.
    Model weights, scalers, predictions and curves are retained through ResultStore.
    """
    from fedcore.q3.real import INPUTS, load_real_panel

    if min_train < 2 or test_size < 1 or max_epochs < 1 or patience < 1:
        raise ValueError("invalid evaluation fold/epoch/patience settings")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("choose distinct seeds")
    cp = load_checkpoint(checkpoint)
    checkpoints = [cp]
    masked = load_checkpoint(masked_checkpoint) if masked_checkpoint is not None else None
    if masked is not None:
        if cp.params["objective"] != OBJECTIVES[0] or masked.params["objective"] != OBJECTIVES[1]:
            raise ValueError("three-way comparison requires next-return and masked checkpoints")
        if masked.config != cp.config:
            raise ValueError("comparison checkpoints must use identical encoder architectures")
        if any(not np.array_equal(masked.scaler[key], cp.scaler[key]) for key in cp.scaler):
            raise ValueError("comparison checkpoints must share history normalization; use matching corpus/split/scaler seed")
        checkpoints.append(masked)
    for checkpoint_ in checkpoints:
        for name, old in checkpoint_.fingerprints.items():
            if fingerprint(name) != old:
                raise ValueError(f"checkpoint input changed: {name}; use a matching data version")
    panel, _, _ = load_real_panel()
    through = pd.Timestamp(through)
    if pd.isna(through) or through >= pd.Timestamp("2026-01-01"):
        raise ValueError("through must be a historical date before the separate 2026 holdout")
    panel = panel.subset(np.flatnonzero(panel.meeting <= through.to_datetime64()))
    folds = walk_forward(panel.meetings(), min_train, test_size)
    if not folds:
        raise ValueError("no evaluation folds inside the requested meeting boundary")
    for fold in folds:
        part = panel.subset(np.flatnonzero(np.isin(panel.meeting, fold.train)))
        for checkpoint_ in checkpoints:
            checkpoint_.require_before(validation_meetings(part).min())
    factories = make_arms(cp, train_kwargs=dict(max_epochs=max_epochs, patience=patience, device=device))
    if masked is not None:
        factories["tx_masked_full"] = lambda seed: TxArm(
            seed=seed, init="pretrained", adapt="full", checkpoint=masked,
            max_epochs=max_epochs, patience=patience, device=device,
        )
    if not arms or len(set(arms)) != len(arms) or set(arms) - factories.keys():
        raise ValueError(f"choose distinct arms from {list(factories)}")
    store = store or ResultStore(RESULTS)
    results, run_ids = {}, {}
    for name in arms:
        fitted = []
        arm_checkpoint = masked if name == "tx_masked_full" else cp

        def factory(seed):
            arm = factories[name](seed)
            fitted.append((seed, arm))
            return arm

        result = run_walk_forward(panel, factory, min_train=min_train, test_size=test_size, seeds=tuple(seeds))
        state_rows, scaler_rows = [], []
        for ordinal, (seed, arm) in enumerate(fitted):
            for key, tensor in arm.model.state_dict().items():
                state_rows.append(dict(fit=ordinal, seed=seed, name=key,
                                       shape=json.dumps(list(tensor.shape)),
                                       values=json.dumps(tensor.cpu().reshape(-1).tolist())))
            for field in ("hist_median", "hist_iqr", "fund_median", "fund_iqr", "ctx_median", "ctx_iqr"):
                scaler_rows.append(dict(fit=ordinal, seed=seed, name=field,
                                        values=json.dumps(getattr(arm.scaler, field).tolist())))
        run = store.save(
            specs()["Q3"], name, role="exploratory", datasets=list(INPUTS),
            params=dict(arm=name, seeds=list(seeds), min_train=min_train, test_size=test_size,
                        through=str(through.date()), checkpoint_run=str(arm_checkpoint.path),
                        checkpoint_digest=arm_checkpoint.digest, objective=arm_checkpoint.params["objective"],
                        checkpoint_available_through=str(arm_checkpoint.available_through.date()),
                        uses_pretrained_weights=name != "tx_scratch",
                        encoder_config=cp.config, max_epochs=max_epochs, patience=patience,
                        device=device, n_meetings=len(panel.meetings()),
                        n_fund=panel.fundamentals.shape[1], n_ctx=panel.context.shape[1]),
            metrics=result.metrics, tables=dict(predictions=result.predictions, folds=result.folds,
                                                curves=result.curves, by_seed=result.by_seed,
                                                **{"response-state": pd.DataFrame(state_rows),
                                                   "response-scalers": pd.DataFrame(scaler_rows)}),
            notes="Raw Fed-day return target; mu=a+b*s; shared meeting-weighted MSE. "
                  "Checkpoint predates all inner validation blocks. No 2026 holdout. "
                  "fit ordinal maps to folds table row order. Paired CI is the shared meeting bootstrap.",
        )
        results[name], run_ids[name] = result, run.run_id
        print(f"saved {name}: {run.path}", flush=True)
    rows = []
    for name, result in results.items():
        row = dict(arm=name, run_id=run_ids[name], **result.metrics)
        if "tx_scratch" in results and name != "tx_scratch":
            row.update({f"mse_minus_scratch_{key}": value
                        for key, value in compare(result, results["tx_scratch"]).items()})
        rows.append(row)
    summary = pd.DataFrame(rows)
    store.save(specs()["Q3"], "encoder-transfer-comparison", role="exploratory",
               params=dict(through=str(through.date()), checkpoint_digest=cp.digest, runs=run_ids),
               tables={"comparison": summary})
    return summary


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    for command in ("plan", "audit", "pretrain"):
        p = sub.add_parser(command)
        p.add_argument("--cutoff", required=True, help="exclusive date boundary for next-day labels")
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
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True, help="saved pretraining run directory")
    p.add_argument("--masked-checkpoint", help="optional masked baseline for the same three-way comparison")
    p.add_argument("--through", required=True, help="inclusive historical evaluation boundary")
    p.add_argument("--min-train", type=int, default=60)
    p.add_argument("--test-size", type=int, default=16)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--arms", nargs="+", default=["tx_scratch", "tx_pre_full"])
    p.add_argument("--max-epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--device")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "plan":
        nextday = args.objective == OBJECTIVES[0]
        print(json.dumps({**vars(args), "architecture": PatchTransformerEncoder().get_config(),
                          "target": "next trading day's raw percent return (mean and variance)" if nextday else "masked observed history entries",
                          "loss": "log(v) + (r_next - m)^2/v" if nextday else "observed masked-patch MSE, balanced across channels",
                          "excluded": "all FOMC target dates",
                          "transfer": "encoder + history scaler; discard temporary head",
                          "purge_history": True,
                          "results": str(RESULTS), "loads_data": False, "fits": False}, indent=2))
        return
    if args.command == "evaluate":
        print(evaluate_checkpoint(args.checkpoint, through=args.through, min_train=args.min_train,
                                  masked_checkpoint=args.masked_checkpoint,
                                  test_size=args.test_size, seeds=args.seeds, arms=args.arms,
                                  max_epochs=args.max_epochs, patience=args.patience,
                                  device=args.device).to_string(index=False))
        return
    cutoff = pd.Timestamp(args.cutoff)
    if pd.isna(cutoff) or cutoff > pd.Timestamp("2026-01-01"):
        raise ValueError("pretraining cutoff must exclude the 2026 holdout")
    corpus = real_return_corpus(cutoff, stride=args.stride, min_observed=args.min_observed)
    audit = corpus_audit(corpus, args.val_frac)
    print(json.dumps(audit, indent=2), flush=True)
    if args.command == "audit":
        return
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


if __name__ == "__main__":
    main()
