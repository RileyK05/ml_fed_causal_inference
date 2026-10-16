"""Experiment: separate the meeting-average move from firm-to-firm differences.

Per meeting, squared error splits exactly into a meeting-average part and a cross-firm part:

    MSE_t = (mean return - mean prediction)^2 + mean[(r - r_bar) - (mu - mu_bar)]^2

Two changes, crossed in a 2 x 2 with the current model as reference:

* ``cross_firm_loss``: train (and early-stop) on ``lambda_mean * mean part + cross-firm part``
  with lambda_mean = 0.05, so the meeting-average move (dominated by 2008) stops drowning the
  firm-to-firm signal. lambda_mean = 1 is the ordinary per-meeting MSE.
* ``centered_a`` (StructModel): a cannot move the meeting average. a is centered within each
  meeting plus one learned constant; context (identical for every firm at a meeting) feeds
  only b. The predicted meeting average is then a0 + mean(b) x shock: shock-driven only.

Predictions are always a + b x shock against the raw Fed-day return. Batches and evaluation
chunks always hold whole meetings, which both changes need.

``TwoPartArm`` is the follow-up: pooled's meeting-wide move plus the networks' firm differences.

    python -m fedenc.decouple run --seeds 0 1 2 --device cuda      (training: needs the go-ahead)
    python -m fedenc.decouple confirm --seeds 3 4 5 --device cuda  (training: needs the go-ahead)
    python -m fedenc.decouple final --device cuda                  (the one 2022-2025 test; run once)
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from fedcore.q3 import ResponseModel
from fedcore.q3.model import _MLP
from fedcore.q3.train import (
    ChannelScaler, _meeting_index_groups, _meeting_rows, _tensors, resolve_device, seed_everything,
    validation_meetings,
)
from fedenc.encoder_transformer import PatchTransformerEncoder

CHUNK_ROWS = 4096
LAMBDA_CROSS_FIRM = 0.05


class StructModel(nn.Module):
    """a from firm traits only, centered within the meeting; b from firm traits and context."""

    def __init__(self, encoder, n_fund, n_ctx, z_dim=64, branch_dim=16, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.fund = _MLP(n_fund * 2, branch_dim, dropout)
        self.context_mlp = _MLP(n_ctx, branch_dim, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(int(encoder.out_dim) + branch_dim, z_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(z_dim, z_dim), nn.ReLU(),
        )
        self.head_a = nn.Linear(z_dim, 1)
        self.head_b = nn.Linear(z_dim + branch_dim, 1)
        self.a0 = nn.Parameter(torch.zeros(1))

    def forward(self, history, history_mask, fundamentals, fundamentals_mask, context, shock):
        """(uncentered a, b). Centering needs meeting codes, so ``outputs`` does it."""
        dtype = self.head_a.weight.dtype
        h = self.encoder(history.to(dtype), history_mask)
        f = self.fund(torch.cat((fundamentals.to(dtype), fundamentals_mask.to(dtype)), dim=-1))
        z = self.fusion(torch.cat((h, f), dim=-1))
        c = self.context_mlp(context.to(dtype))
        return self.head_a(z).squeeze(-1), self.head_b(torch.cat((z, c), dim=-1)).squeeze(-1)


def _segments(codes):
    _, inverse = torch.unique(codes, return_inverse=True)
    width = int(inverse.max().item()) + 1
    count = torch.zeros(width, device=codes.device, dtype=torch.float32).index_add_(
        0, inverse, torch.ones_like(inverse, dtype=torch.float32))
    return inverse, width, count


def _segment_mean(values, inverse, width, count):
    total = torch.zeros(width, device=values.device, dtype=values.dtype).index_add_(0, inverse, values)
    return total / count.to(values.dtype)


def outputs(model, batch, rows):
    """mu, a, b for ``rows`` (whole meetings). Centers a within each meeting for StructModel."""
    args = [batch[k][rows] for k in ("history", "history_mask", "fundamentals", "fundamentals_mask",
                                      "context", "shock")]
    shock = batch["shock"][rows].to(torch.float32)
    if isinstance(model, StructModel):
        a_raw, b = model(*args)
        inverse, width, count = _segments(batch["codes"][rows])
        a = a_raw - _segment_mean(a_raw, inverse, width, count)[inverse] + model.a0
    else:
        _, a, b = model(*args)
    return a + b * shock, a, b


def meeting_losses(mu, target, codes):
    """Per meeting: (mean part, cross-firm part). Their sum is the meeting's MSE."""
    inverse, width, count = _segments(codes)
    y_bar = _segment_mean(target, inverse, width, count)
    m_bar = _segment_mean(mu, inverse, width, count)
    cross = (target - y_bar[inverse]) - (mu - m_bar[inverse])
    return (y_bar - m_bar) ** 2, _segment_mean(cross ** 2, inverse, width, count)


def _chunks(groups, limit=CHUNK_ROWS):
    """Whole-meeting index chunks of about ``limit`` rows."""
    out, current, size = [], [], 0
    for g in groups:
        if current and size + len(g) > limit:
            out.append(np.concatenate(current))
            current, size = [], 0
        current.append(g)
        size += len(g)
    if current:
        out.append(np.concatenate(current))
    return out


def evaluate(model, batch, groups, lambda_mean):
    """(objective, mean part, cross-firm part), each a mean over meetings."""
    model.eval()
    mean_parts, cross_parts = [], []
    with torch.no_grad():
        for idx in _chunks(groups):
            rows = torch.as_tensor(idx, device=batch["target"].device)
            mu, _, _ = outputs(model, batch, rows)
            m, c = meeting_losses(mu.double(), batch["target"][rows].double(), batch["codes"][rows])
            mean_parts.append(m)
            cross_parts.append(c)
    model.train()
    m, c = torch.cat(mean_parts).mean().item(), torch.cat(cross_parts).mean().item()
    return lambda_mean * m + c, m, c


def fit(model, panel, *, lambda_mean=1.0, lr=1e-3, weight_decay=1e-4, max_epochs=40, patience=10,
        val_frac=0.15, seed=0, meetings_per_batch=4):
    """Meeting-batched training on ``lambda_mean * mean part + cross-firm part``; early stopping
    on the same objective over the last ``val_frac`` meetings. Scaler fits on training meetings."""
    val_m = validation_meetings(panel, val_frac)
    train_m = panel.meetings()[:-len(val_m)]
    train_part = panel.subset(_meeting_rows(panel, train_m))
    val_part = panel.subset(_meeting_rows(panel, val_m))
    scaler = ChannelScaler.fit(train_part)
    device = next(model.parameters()).device
    train_batch, val_batch = _tensors(train_part, scaler, device), _tensors(val_part, scaler, device)
    train_groups = _meeting_index_groups(train_part.meeting)
    val_groups = _meeting_index_groups(val_part.meeting)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    best, best_epoch, stale = math.inf, 0, 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    train_curve, val_curve = [], []
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(len(train_groups))
        for start in range(0, len(order), meetings_per_batch):
            chosen = order[start:start + meetings_per_batch]
            rows = torch.as_tensor(np.concatenate([train_groups[j] for j in chosen]), device=device)
            optimizer.zero_grad(set_to_none=True)
            mu, _, _ = outputs(model, train_batch, rows)
            m, c = meeting_losses(mu, train_batch["target"][rows], train_batch["codes"][rows])
            loss = (lambda_mean * m + c).mean() * (len(chosen) / meetings_per_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        train_curve.append(evaluate(model, train_batch, train_groups, lambda_mean)[0])
        val_curve.append(evaluate(model, val_batch, val_groups, lambda_mean)[0])
        if val_curve[-1] < best:
            best, best_epoch, stale = val_curve[-1], epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return argparse.Namespace(scaler=scaler, train_loss=train_curve, val_loss=val_curve, best_epoch=best_epoch,
                              best_val_loss=best, train_meetings=pd.DatetimeIndex(train_m),
                              val_meetings=pd.DatetimeIndex(val_m))


def predict(model, panel, scaler):
    device = next(model.parameters()).device
    batch = _tensors(panel, scaler, device)
    n = len(panel.target)
    out = [np.empty(n) for _ in range(3)]
    model.eval()
    with torch.no_grad():
        for idx in _chunks(_meeting_index_groups(panel.meeting)):
            rows = torch.as_tensor(idx, device=device)
            for k, value in enumerate(outputs(model, batch, rows)):
                out[k][idx] = value.double().cpu().numpy()
    return tuple(out)


def fund_view(panel, fund_cols):
    """``panel`` with only its first ``fund_cols`` fundamentals columns (None: all of them)."""
    import dataclasses

    if fund_cols is None or fund_cols >= panel.fundamentals.shape[1]:
        return panel
    return dataclasses.replace(panel, fundamentals=panel.fundamentals[:, :fund_cols],
                               fundamentals_mask=panel.fundamentals_mask[:, :fund_cols])


class DecoupleArm:
    """``centered_a``: StructModel instead of ResponseModel. ``lambda_mean``: loss weight on the
    meeting-average part (1 = ordinary per-meeting MSE). ``fund_cols``: use only the first
    ``fund_cols`` fundamentals (so arms with and without the sector columns share one panel)."""

    def __init__(self, seed=0, *, centered_a=False, lambda_mean=1.0, device=None, max_epochs=40, patience=10,
                 fund_cols=None):
        self.seed, self.centered_a, self.lambda_mean = int(seed), bool(centered_a), float(lambda_mean)
        self.device, self.max_epochs, self.patience = device, max_epochs, patience
        self.fund_cols = fund_cols
        self.model = self.scaler = self.fit_result = None

    def fit(self, panel, seed=None, **_):
        panel = fund_view(panel, self.fund_cols)
        seed = self.seed if seed is None else int(seed)
        seed_everything(seed)
        make = StructModel if self.centered_a else ResponseModel
        self.model = make(PatchTransformerEncoder(), n_fund=panel.fundamentals.shape[1],
                          n_ctx=panel.context.shape[1]).to(resolve_device(self.device))
        seed_everything(seed)
        self.fit_result = fit(self.model, panel, lambda_mean=self.lambda_mean, max_epochs=self.max_epochs,
                              patience=self.patience, seed=seed)
        self.scaler = self.fit_result.scaler

    def predict(self, panel):
        if self.model is None:
            raise RuntimeError("fit the arm before predict")
        return predict(self.model, fund_view(panel, self.fund_cols), self.scaler)


class TwoPartArm:
    """Meeting-wide move from pooled OLS; firm-to-firm differences from ``members`` networks.

    Each network's (a, b) is demeaned within the meeting and averaged over members, then added
    to pooled's (a, b). So mu's meeting mean is exactly pooled's, and its deviations are the
    members' average deviations. Run-to-run noise lives almost all in the network's meeting
    mean (the 2008 crisis), which this drops."""

    def __init__(self, seed=0, *, members=3, centered_a=True, lambda_mean=LAMBDA_CROSS_FIRM, device=None,
                 max_epochs=40, patience=10, fund_cols=None):
        self.seed, self.members = int(seed), int(members)
        self.net_kwargs = dict(centered_a=centered_a, lambda_mean=lambda_mean, device=device,
                               max_epochs=max_epochs, patience=patience, fund_cols=fund_cols)
        self.pooled = self.nets = self.fit_result = None

    def fit(self, panel, seed=None, **_):
        from fedcore.q3.baselines import PooledOLS

        seed = self.seed if seed is None else int(seed)
        self.pooled = PooledOLS()
        self.pooled.fit(panel)
        self.nets = []
        for j in range(self.members):
            net = DecoupleArm(seed * 100 + j, **self.net_kwargs)
            net.fit(panel)
            self.nets.append(net)
        self.fit_result = self.nets[0].fit_result

    def predict(self, panel):
        if self.nets is None:
            raise RuntimeError("fit the arm before predict")
        codes = pd.factorize(pd.DatetimeIndex(panel.meeting))[0]

        def demean(x):
            return x - pd.Series(x).groupby(codes).transform("mean").to_numpy()

        devs = [tuple(demean(v) for v in net.predict(panel)[1:]) for net in self.nets]
        _, a0, b0 = self.pooled.predict(panel)
        a = a0 + np.mean([d[0] for d in devs], axis=0)
        b = b0 + np.mean([d[1] for d in devs], axis=0)
        return a + b * np.asarray(panel.shock, dtype=np.float64), a, b


def run(through="2009-03-18", seeds=(0, 1, 2), min_train=60, test_size=16, device=None, store=None):
    """2 x 2 (loss x centered a) plus pooled and ridge, then the diagnostics. Training."""
    from fedcore.q3.baselines import BASELINES
    from fedcore.results import ResultStore
    from fedenc import diagnostics
    from fedenc.config import RESULTS
    from fedenc.runs import save_runs
    from fedenc.workflow import bounded_panel

    panel, through = bounded_panel(through)
    store = store or ResultStore(RESULTS)
    budget = dict(device=device)
    lam = LAMBDA_CROSS_FIRM
    makers = {
        "tx_scratch": lambda seed: DecoupleArm(seed, **budget),
        "tx_cross_firm_loss": lambda seed: DecoupleArm(seed, lambda_mean=lam, **budget),
        "tx_centered_a": lambda seed: DecoupleArm(seed, centered_a=True, **budget),
        "tx_centered_a_cross_firm": lambda seed: DecoupleArm(seed, centered_a=True, lambda_mean=lam, **budget),
        "pooled": BASELINES["pooled"],
        "ridge_interact": BASELINES["ridge_interact"],
    }
    summary = save_runs(panel, makers, prefix="decouple-exp",
                                params=dict(through=str(through.date()), lambda_mean=lam),
                                min_train=min_train, test_size=test_size, seeds=seeds, store=store)
    runs = {arm.removeprefix("tx_"): Path(path) for arm, path in zip(summary["arm"], summary["path"])}
    tables = diagnostics.run(runs, ("pooled", "scratch"), store=store, panel=panel)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
        for name, table in tables.items():
            print(f"\n== {name}\n{table.to_string(index=False)}", flush=True)
    return summary, tables


REFERENCES = ("capm_pooled", "pooled", "scratch")  # CAPM first: a Q3 model must beat beta


def confirm(through="2009-03-18", seeds=(3, 4, 5), members=3, min_train=60, test_size=16, device=None,
            store=None, universe="sp500", sector=False, target="raw", scratch=True, shock="STMT"):
    """Re-test the two-part model on fresh seeds, with the plain transformer refit alongside. Training.

    With a broader ``universe`` the arms train on every row; diagnostics run on all rows and
    again on the S&P 500 rows only, which compare with the S&P-only panel's scores. ``sector``
    adds a ``two_part_sector`` arm that also sees GICS sector one-hot columns (``two_part`` keeps
    the original fundamentals). ``target="market_adjusted"`` fits every arm, pooled included, to
    the abnormal return (see ``workflow.market_adjusted``). ``scratch=False`` skips the plain
    transformer. ``shock="ME"`` uses the whole monetary event surprise (statement plus press
    conference) instead of the statement window. The CAPM baseline (``capm_pooled``) is the
    first reference in every table."""
    from fedcore.q3.baselines import BASELINES
    from fedcore.q3.real import FUNDAMENTAL_NAMES, UNIVERSES
    from fedcore.results import ResultStore
    from fedenc import diagnostics
    from fedenc.config import RESULTS
    from fedenc.runs import save_runs
    from fedenc.workflow import bounded_panel

    panel, through, rows = bounded_panel(through, universe=universe, with_rows=True, sector=sector, target=target,
                                         shock=shock)
    store = store or ResultStore(RESULTS)
    base_cols = len(FUNDAMENTAL_NAMES)
    makers = {"tx_scratch": lambda seed: DecoupleArm(seed, device=device, fund_cols=base_cols)} if scratch else {}
    makers["two_part"] = lambda seed: TwoPartArm(seed, members=members, device=device, fund_cols=base_cols)
    if sector:
        makers["two_part_sector"] = lambda seed: TwoPartArm(seed, members=members, device=device)
    makers["capm_pooled"] = BASELINES["capm_pooled"]
    makers["pooled"] = BASELINES["pooled"]
    prefix = "twopart-confirm" if universe == "sp500" else f"twopart-{universe}"
    if pd.Timestamp(through) != pd.Timestamp("2009-03-18"):
        prefix += f"-to{through.year}"
    prefix += "-adj" if target == "market_adjusted" else ""
    prefix += f"-{shock.lower()}" if shock != "STMT" else ""
    summary = save_runs(panel, makers, prefix=prefix,
                                params=dict(through=str(through.date()), lambda_mean=LAMBDA_CROSS_FIRM,
                                            members=members, universe=universe, sector=sector, target=target,
                                            shock=shock),
                                min_train=min_train, test_size=test_size, seeds=seeds, store=store,
                                datasets=UNIVERSES[universe].inputs)
    runs = {arm.removeprefix("tx_"): Path(path) for arm, path in zip(summary["arm"], summary["path"])}
    tables = {"all": diagnostics.run(runs, REFERENCES, store=store, panel=panel)}
    if universe != "sp500":
        sp = rows.loc[rows["in_sp500"], ["meeting", "permno"]].rename(columns={"permno": "firm_id"})
        tables["sp500_rows"] = diagnostics.run(runs, REFERENCES, store=store, panel=panel, keep=sp,
                                               label="diagnostics-sp500-rows")
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
        for scope, group in tables.items():
            for name, table in group.items():
                print(f"\n== [{scope}] {name}\n{table.to_string(index=False)}", flush=True)
    return summary, tables


DEV_END = pd.Timestamp("2021-12-31")    # development period (decided 2026-10-15)
FINAL_END = pd.Timestamp("2025-12-31")  # final test: 2022-2025 meetings, used once


def final(seeds=(6, 7, 8), members=3, device=None, store=None):
    """The one confirmatory test of the locked Q3 model. Training; run once.

    Locked spec: TwoPartArm (pooled meeting-wide move + the average firm differences of
    ``members`` centered-a networks, cross-firm loss), S&P 500 universe, raw returns, the six
    original fundamentals. One fold: fit on every meeting through ``DEV_END``, score the
    2022-2025 meetings against pooled refit on the same data. Saved as role ``primary``.
    """
    from fedcore.protocol import walk_forward
    from fedcore.q3.baselines import BASELINES
    from fedcore.q3.real import UNIVERSES
    from fedcore.results import ResultStore
    from fedenc import diagnostics
    from fedenc.config import RESULTS
    from fedenc.runs import save_runs
    from fedenc.workflow import bounded_panel

    panel, through = bounded_panel(FINAL_END)
    meetings = panel.meetings()
    n_dev = int((meetings <= DEV_END).sum())
    folds = walk_forward(meetings, n_dev, len(meetings) - n_dev)
    if len(folds) != 1 or folds[0].test.min() <= DEV_END or folds[0].train.max() > DEV_END:
        raise RuntimeError("the final test must be one fold: train through 2021, test 2022-2025")
    print(f"final test: train {len(folds[0].train)} meetings to {folds[0].train.max().date()}, "
          f"test {len(folds[0].test)} meetings {folds[0].test.min().date()} to {folds[0].test.max().date()}",
          flush=True)
    store = store or ResultStore(RESULTS)
    makers = {"two_part": lambda seed: TwoPartArm(seed, members=members, device=device),
              "pooled": BASELINES["pooled"]}
    summary = save_runs(panel, makers, prefix="q3-final", role="primary",
                                params=dict(through=str(through.date()), dev_end=str(DEV_END.date()),
                                            lambda_mean=LAMBDA_CROSS_FIRM, members=members, universe="sp500",
                                            target="raw", locked_spec="two_part"),
                                min_train=n_dev, test_size=len(meetings) - n_dev, seeds=seeds, store=store,
                                datasets=UNIVERSES["sp500"].inputs)
    runs = {arm: Path(path) for arm, path in zip(summary["arm"], summary["path"])}
    tables = diagnostics.run(runs, ("pooled",), store=store, panel=panel, label="q3-final-diagnostics")
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
        for name, table in tables.items():
            print(f"\n== {name}\n{table.to_string(index=False)}", flush=True)
    return summary, tables


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name, seeds in (("run", [0, 1, 2]), ("confirm", [3, 4, 5])):
        q = sub.add_parser(name)
        q.add_argument("--through", default="2009-03-18")
        q.add_argument("--seeds", type=int, nargs="+", default=seeds)
        q.add_argument("--device")
        if name == "confirm":
            q.add_argument("--members", type=int, default=3)
            q.add_argument("--universe", default="sp500")
            q.add_argument("--sector", action="store_true", help="also fit a two-part arm that sees GICS sector")
            q.add_argument("--target", default="raw", choices=("raw", "market_adjusted"))
            q.add_argument("--test-size", type=int, default=16, help="meetings per test window")
            q.add_argument("--no-scratch", action="store_true", help="skip the plain transformer arm")
            q.add_argument("--shock", default="STMT", choices=("STMT", "ME"))
    q = sub.add_parser("final", help="the one confirmatory test on 2022-2025 (run once)")
    q.add_argument("--seeds", type=int, nargs="+", default=[6, 7, 8])
    q.add_argument("--members", type=int, default=3)
    q.add_argument("--device")
    args = p.parse_args(argv)
    if args.command == "final":
        final(seeds=tuple(args.seeds), members=args.members, device=args.device)
    elif args.command == "run":
        run(args.through, seeds=tuple(args.seeds), device=args.device)
    else:
        confirm(args.through, seeds=tuple(args.seeds), members=args.members, device=args.device,
                universe=args.universe, sector=args.sector, target=args.target, test_size=args.test_size,
                scratch=not args.no_scratch, shock=args.shock)


if __name__ == "__main__":
    main()
