"""Walk-forward scoring shared by every Q3 arm.

Folds come from ``fedcore.protocol.walk_forward`` over whole meetings.
``RankIC`` is undefined for a meeting with fewer than 3 firms or a constant
prediction or target. Those meetings stay in the MSE. The reported ``rank_ic``
is the mean over meetings where the correlation exists, and
``rank_ic_defined_frac`` records how many that was. Nothing is dropped quietly
from the primary loss.

``compare`` is the paired difference of per-meeting MSE (first result minus
baseline), averaged across seeds, with a meeting bootstrap CI. A negative
estimate means the first result has the lower error.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fedcore.protocol import meeting_bootstrap, walk_forward
from fedcore.q3.contracts import Q3Panel
from fedcore.q3.synthetic import make_synthetic_panel
from fedcore.questions import specs as question_specs
from fedcore.results import ResultStore

_METRIC_COLS = (
    "mse_per_meeting",
    "centered_mse",
    "rank_ic",
    "rank_ic_defined_frac",
    "b_corr",
    "b_rmse",
    "a_rmse",
)


@dataclass
class EvalResult:
    predictions: pd.DataFrame
    metrics: dict
    by_seed: pd.DataFrame
    folds: pd.DataFrame
    curves: pd.DataFrame
    notes: str = ""


def run_walk_forward(
    panel: Q3Panel,
    make_model,
    *,
    min_train: int = 80,
    test_size: int = 16,
    seeds: tuple[int, ...] = (0, 1, 2),
    fit_kwargs: dict | None = None,
) -> EvalResult:
    """Fit a fresh model on each fold and seed. Score only that fold's test meetings."""
    panel.validate()
    folds = walk_forward(panel.meetings(), min_train, test_size)
    if not folds:
        raise ValueError(
            f"walk_forward produced no folds (meetings={len(panel.meetings())}, "
            f"min_train={min_train}, test_size={test_size})"
        )
    pred_frames = []
    fold_rows = []
    curve_rows = []
    for fold in folds:
        train_panel = panel.subset(_meeting_rows(panel, fold.train))
        test_panel = panel.subset(_meeting_rows(panel, fold.test))
        for seed in seeds:
            model = make_model(int(seed))
            kw = dict(fit_kwargs or {})
            kw["seed"] = int(seed)
            model.fit(train_panel, **kw)
            mu, a_hat, b_hat = _as_numpy(model.predict(test_panel))
            frame = _prediction_frame(test_panel, mu, a_hat, b_hat, fold.k, int(seed))
            pred_frames.append(frame)
            fit_result = getattr(model, "fit_result", None)
            best_epoch = np.nan if fit_result is None else fit_result.best_epoch
            best_val = np.nan if fit_result is None else fit_result.best_val_loss
            n_val = np.nan if fit_result is None else len(fit_result.val_meetings)
            fold_rows.append(
                {
                    "fold": fold.k,
                    "seed": int(seed),
                    "n_train_meetings": len(fold.train),
                    "n_val_meetings": n_val,
                    "n_test_meetings": len(fold.test),
                    "n_train_rows": len(train_panel.target),
                    "n_test_rows": len(test_panel.target),
                    "train_start": fold.train.min(),
                    "train_end": fold.train.max(),
                    "test_start": fold.test.min(),
                    "test_end": fold.test.max(),
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val,
                }
            )
            if fit_result is not None:
                for epoch, (tr, va) in enumerate(zip(fit_result.train_loss, fit_result.val_loss), start=1):
                    curve_rows.append(
                        {"fold": fold.k, "seed": int(seed), "epoch": epoch, "train_loss": tr, "val_loss": va}
                    )
    predictions = pd.concat(pred_frames, ignore_index=True)
    by_seed = _metrics_by_seed(predictions)
    curves = pd.DataFrame(curve_rows, columns=["fold", "seed", "epoch", "train_loss", "val_loss"])
    return EvalResult(
        predictions=predictions,
        metrics=_average_metrics(by_seed),
        by_seed=by_seed,
        folds=pd.DataFrame(fold_rows),
        curves=curves,
    )


def compare(result: EvalResult, baseline_result: EvalResult) -> dict:
    """Meeting-bootstrap CI for the mean paired per-meeting MSE difference."""
    diff = _paired_meeting_diff(result.predictions, baseline_result.predictions)
    frame = diff.rename("diff").reset_index()
    if frame.columns[0] != "meeting":
        frame = frame.rename(columns={frame.columns[0]: "meeting"})
    return meeting_bootstrap(frame, lambda d: float(d["diff"].mean()), meeting_col="meeting")


def save(result: EvalResult, store: ResultStore, name: str, params: dict | None = None, notes: str = ""):
    """Append an exploratory run under the Q3 question. The store chooses the folder."""
    spec = question_specs()["Q3"]
    return store.save(
        spec,
        name,
        role="exploratory",
        params=_jsonable(params or {}),
        metrics=_jsonable(result.metrics),
        tables={
            "predictions": result.predictions,
            "folds": result.folds,
            "curves": result.curves,
            "by_seed": result.by_seed,
        },
        notes=notes or result.notes,
    )


def run_arms(
    arms: dict,
    store: ResultStore,
    *,
    data: str = "synthetic",
    noise: str = "realistic",
    seeds: tuple[int, ...] = (0, 1, 2),
    min_train: int | None = None,
    test_size: int | None = None,
    fit_kwargs: dict | None = None,
    **panel_kwargs,
) -> pd.DataFrame:
    """Build one panel, score every arm, save each run, and return a comparison table.

    ``data="real"`` scores the WRDS firm panel (``fedcore.q3.real``) and needs explicit
    ``min_train`` and ``test_size``; ``noise`` and
    ``panel_kwargs`` apply to synthetic data only. When ``min_train`` and ``test_size`` are left at their defaults and the panel
    has fewer than 80 meetings, both are reduced so the small end-to-end test
    still produces a fold. The saved params record that choice.
    """
    if data not in ("synthetic", "real"):
        raise ValueError(f"unknown data={data!r}")
    if not arms:
        raise ValueError("arms is empty")
    panel_inputs: dict = {}
    if data == "real":
        if panel_kwargs:
            raise ValueError(f"panel options apply to synthetic data only: {sorted(panel_kwargs)}")
        if min_train is None or test_size is None:
            raise ValueError("data='real' needs explicit min_train and test_size (docs/training-plan.md, D1)")
        from fedcore.q3.real import load_real_panel

        panel, _, meta = load_real_panel()
        noise, panel_seed = None, None
        panel_inputs = meta["inputs"]
    else:
        panel_seed = int(panel_kwargs.pop("seed", 0))
        panel = make_synthetic_panel(noise=noise, seed=panel_seed, **panel_kwargs)
    n_meetings = len(panel.meetings())
    mt, ts, shrunk = _resolve_window(n_meetings, min_train, test_size)
    saved: dict[str, EvalResult] = {}
    for name, factory in arms.items():
        result = run_walk_forward(
            panel, factory, min_train=mt, test_size=ts, seeds=tuple(seeds), fit_kwargs=fit_kwargs
        )
        params = {
            "data": data,
            "noise": noise,
            "arm": name,
            "seeds": [int(s) for s in seeds],
            "min_train": int(mt),
            "test_size": int(ts),
            "window_shrunk": bool(shrunk),
            "panel_seed": panel_seed,
            "n_firms": int(np.unique(panel.firm_id).size),
            "n_meetings": int(n_meetings),
            "n_rows": int(panel.target.shape[0]),
            "fit_kwargs": fit_kwargs or {},
            "panel_inputs": panel_inputs,
        }
        notes = (
            f"{'Real WRDS' if data == 'real' else 'Synthetic'} Q3 panel. mse_per_meeting is the mean of within-meeting MSE. "
            "rank_ic is null when a meeting has fewer than 3 firms or a constant prediction; "
            "rank_ic_defined_frac is the share of test meetings where it is defined. "
            f"noise={noise}, panel_seed={panel_seed}, min_train={mt}, test_size={ts}, shrunk={shrunk}."
        )
        save(result, store, name, params=params, notes=notes)
        saved[name] = result
    table = pd.DataFrame([{"arm": name, **res.metrics} for name, res in saved.items()])
    if len(saved) > 1:
        for ref in ("pooled", "ridge_interact"):
            if ref not in saved:
                continue
            rows = []
            for name, res in saved.items():
                if name == ref:
                    rows.append(
                        {
                            "arm": name,
                            f"mse_minus_{ref}": 0.0,
                            f"mse_minus_{ref}_ci_low": 0.0,
                            f"mse_minus_{ref}_ci_high": 0.0,
                        }
                    )
                    continue
                boot = compare(res, saved[ref])
                rows.append(
                    {
                        "arm": name,
                        f"mse_minus_{ref}": boot["estimate"],
                        f"mse_minus_{ref}_ci_low": boot["ci_low"],
                        f"mse_minus_{ref}_ci_high": boot["ci_high"],
                    }
                )
            table = table.merge(pd.DataFrame(rows), on="arm", how="left")
    return table


def _resolve_window(n_meetings: int, min_train: int | None, test_size: int | None) -> tuple[int, int, bool]:
    defaulting = min_train is None and test_size is None
    mt = 80 if min_train is None else int(min_train)
    ts = 16 if test_size is None else int(test_size)
    if mt < 1 or ts < 1:
        raise ValueError("min_train and test_size must be positive")
    if mt < n_meetings:
        return mt, ts, False
    if not defaulting:
        raise ValueError(f"min_train={mt} leaves no test meeting out of {n_meetings}")
    ts = max(1, n_meetings // 5)
    mt = n_meetings - ts
    if mt < 2:
        raise ValueError(f"panel of {n_meetings} meetings is too small to score")
    return mt, ts, True


def _meeting_rows(panel: Q3Panel, meetings) -> np.ndarray:
    selected = pd.DatetimeIndex(meetings).to_numpy(dtype="datetime64[ns]")
    own = np.asarray(panel.meeting, dtype="datetime64[ns]")
    return np.flatnonzero(np.isin(own, selected))


def _as_numpy(triple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = []
    for part in triple:
        if hasattr(part, "detach"):
            part = part.detach().cpu().numpy()
        out.append(np.asarray(part, dtype=np.float64).reshape(-1))
    return out[0], out[1], out[2]


def _prediction_frame(panel: Q3Panel, mu, a_hat, b_hat, fold: int, seed: int) -> pd.DataFrame:
    n = len(panel.target)
    for name, arr in (("mu", mu), ("a_hat", a_hat), ("b_hat", b_hat)):
        if arr.shape != (n,):
            raise ValueError(f"{name} has shape {arr.shape}, expected {(n,)}")
    target = np.asarray(panel.target, dtype=np.float64)
    frame = {
        "meeting": panel.meeting,
        "firm_id": panel.firm_id,
        "fold": fold,
        "seed": seed,
        "mu": mu,
        "a_hat": a_hat,
        "b_hat": b_hat,
        "target": target,
        "gap": target - mu,
    }
    if panel.a_true is not None:
        frame["a_true"] = np.asarray(panel.a_true, dtype=np.float64)
        frame["b_true"] = np.asarray(panel.b_true, dtype=np.float64)
    return pd.DataFrame(frame)


def _metrics_by_seed(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, part in predictions.groupby("seed", sort=True):
        row = {"seed": int(seed), **_metrics_one(part)}
        rows.append(row)
    return pd.DataFrame(rows)


def _metrics_one(df: pd.DataFrame) -> dict:
    mses, centered, ics = [], [], []
    for _, group in df.groupby("meeting", sort=True):
        y = group["target"].to_numpy(dtype=float)
        mu = group["mu"].to_numpy(dtype=float)
        mses.append(float(np.mean((y - mu) ** 2)))
        centered.append(float(np.mean(((y - y.mean()) - (mu - mu.mean())) ** 2)))
        ics.append(_rank_ic(mu, y))
    ic = np.asarray(ics, dtype=float)
    defined = np.isfinite(ic)
    out = {
        "mse_per_meeting": float(np.mean(mses)),
        "centered_mse": float(np.mean(centered)),
        "rank_ic": float(np.mean(ic[defined])) if defined.any() else float("nan"),
        "rank_ic_defined_frac": float(defined.mean()) if len(ic) else float("nan"),
        "n_test_meetings": int(len(mses)),
    }
    if "b_true" in df.columns:
        out["b_corr"] = _corr(df["b_hat"], df["b_true"])
        out["b_rmse"] = _rmse(df["b_hat"], df["b_true"])
        out["a_rmse"] = _rmse(df["a_hat"], df["a_true"])
    return out


def _average_metrics(by_seed: pd.DataFrame) -> dict:
    out = {"n_seeds": int(len(by_seed))}
    if "n_test_meetings" in by_seed:
        out["n_test_meetings"] = int(by_seed["n_test_meetings"].iloc[0])
    for col in _METRIC_COLS:
        if col not in by_seed.columns:
            continue
        vals = by_seed[col].to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        out[col] = float(np.mean(finite)) if finite.size else float("nan")
        out[col + "_std"] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    return out


def _paired_meeting_diff(left: pd.DataFrame, right: pd.DataFrame) -> pd.Series:
    a = _meeting_mse(left)
    b = _meeting_mse(right)
    both = a.index.intersection(b.index)
    if len(both) == 0:
        raise ValueError("results have no meetings in common")
    diff = (a.loc[both] - b.loc[both]).astype(float)
    diff.index.name = "meeting"
    return diff


def _meeting_mse(df: pd.DataFrame) -> pd.Series:
    rows = []
    for (meeting, seed), group in df.groupby(["meeting", "seed"], sort=True):
        y = group["target"].to_numpy(dtype=float)
        mu = group["mu"].to_numpy(dtype=float)
        rows.append((meeting, seed, float(np.mean((y - mu) ** 2))))
    table = pd.DataFrame(rows, columns=["meeting", "seed", "mse"])
    return table.groupby("meeting", sort=True)["mse"].mean()


def _rank_ic(mu: np.ndarray, target: np.ndarray) -> float:
    if len(mu) < 3 or np.nanstd(mu) < 1e-12 or np.nanstd(target) < 1e-12:
        return float("nan")
    value = float(pd.Series(mu).corr(pd.Series(target), method="spearman"))
    return value


def _corr(a, b) -> float:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2)))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return _jsonable(value.item())
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)
