"""Q3 v6: learn fragility from macro shock days, score on Fed days (docs/encoder/q3-v6-plan.md section 3).

    mu = mu_base + P_m(delta_b) * s,   delta_b = head(market history, fundamentals, context[, filing gate])

Same base, beta-span projection, two-stage step-matched refit and walk-forward folds as v4
(``fedenc.v4.V4Arm``). Two changes:

- **Pretraining** (``pretrain``): before the Fed fit, the model learns beyond-beta responses on
  macro shock days (``fedcore.q3.shock_days``) dated before the fold's last training meeting, with
  its own two-beta base and early stopping on the last 15% of those days. Both Fed stages start
  from the best pretrained weights, so a stage-1 choice of epoch 0 means "use the pretrained
  model as is". One pretraining per fit, shared by the members. Pretrained arms scale inputs
  with statistics from the shock days' inner block throughout.
- **Selection on big surprises** (all arms): stage 1 scores only validation meetings with
  |s| >= 2.5bp, or every validation meeting when the block has none.

Text (``vectors``): the cached frozen EmbeddingGemma 2 filing vector (``fedenc.filing_vectors``)
enters as a bounded gate, ``delta_b * (1 + lam * tanh(g(vector)))``. It can scale a correction
up or down but cannot create one for a firm the market/fundamentals model leaves at zero.
``change=True`` feeds the year-over-year change in the vector instead.

Arms: F0 transformer (v4 B5 with big-surprise selection); F1 F0 pretrained; F2 F1 + filing gate;
F3 F2 with the change vector; F0s/F1s the summary-MLP pair (v4 B4).

    python -m fedenc.v6 run --arms F0 F1 --device cuda      # training: needs the user's go-ahead
    python -m fedenc.v6 score                                # read-only scoring of saved runs
    python -m fedenc.v6 final                                # the one 2022-2025 test: needs the go-ahead
    python -m fedenc.v6 score-final                          # its pre-stated verdict
"""
from __future__ import annotations

import argparse
import copy
import math

import numpy as np
import pandas as pd
import torch
from torch import nn

from fedcore.q3.contracts import Q3Panel
from fedcore.q3.train import _meeting_index_groups, resolve_device, seed_everything
from fedenc.v2 import CAPMBetas
from fedenc.v3 import INPUT_KEYS, _Scaler, basis_rows
from fedenc.v4 import PRIMARY, STANDOUT, V4Arm, _codes, _inner_split

GATE_LAMBDA = 0.5


class GatedCorrection(nn.Module):
    """A v4 structural model whose b correction is scaled by a bounded filing-vector gate."""

    input_keys = INPUT_KEYS + ("filing", "filing_mask")

    def __init__(self, base: nn.Module, width: int, hidden: int = 32, lam: float = GATE_LAMBDA):
        super().__init__()
        self.base, self.lam = base, float(lam)
        self.gate = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)  # gate = 1 at the start: the ungated model
        self.head_a, self.head_b = base.head_a, base.head_b

    def forward(self, history, history_mask, fundamentals, fundamentals_mask, context, filing, filing_mask):
        a, b = self.base(history, history_mask, fundamentals, fundamentals_mask, context)
        g = torch.tanh(self.gate(filing.to(b.dtype))).squeeze(-1) * filing_mask.to(b.dtype)
        return a, b * (1.0 + self.lam * g)[:, None]


class V6Arm(V4Arm):
    """V4Arm with optional shock-day pretraining, a filing gate, and big-surprise model selection."""

    def __init__(self, seed: int = 0, *, pretrain: Q3Panel | None = None, vectors=None, change: bool = False,
                 select_big: bool = True, pretrain_epochs: int = 20, pretrain_patience: int = 3, **kw):
        super().__init__(seed, b_only=True, **kw)
        if change and vectors is None:
            raise ValueError("change=True needs filing vectors")
        self.pretrain, self.vectors, self.change = pretrain, vectors, bool(change)
        self.select_big, self.pretrain_epochs, self.pretrain_patience = bool(select_big), int(pretrain_epochs), int(pretrain_patience)

    def config(self) -> dict:
        return dict(super().config(), model="v6", pretrain=self.pretrain is not None,
                    pretrain_events=None if self.pretrain is None else int(len(np.unique(self.pretrain.meeting))),
                    pretrain_epochs=self.pretrain_epochs, pretrain_patience=self.pretrain_patience,
                    text=None if self.vectors is None else ("filing_change" if self.change else "filing_level"),
                    gate_lambda=GATE_LAMBDA if self.vectors is not None else None,
                    select_big=self.select_big, select_threshold=PRIMARY)

    def _new_model(self, panel: Q3Panel, device):
        model = super()._new_model(panel, device)
        if self.vectors is not None:
            model = GatedCorrection(model, self.vectors.width).to(device)
        return model

    def _data(self, p: Q3Panel, base, scaler, device) -> dict:
        d = V4Arm._data(p, base, scaler, device)
        if self.vectors is not None:
            v, ok = self.vectors.for_panel(p, change=self.change)
            d["x"]["filing"] = torch.as_tensor(v, dtype=torch.float16, device=device)
            d["x"]["filing_mask"] = torch.as_tensor(ok, device=device)
        return d

    def _selection_groups(self, val: Q3Panel, groups):
        if not self.select_big:
            return groups, False
        s = np.asarray(val.shock, dtype=np.float64)
        big = [g for g in groups if abs(s[g[0]]) >= PRIMARY]
        return (big, True) if big else (groups, False)

    def _train_best(self, model, opt, d_tr, g_tr, d_va, g_va, rng, device, max_epochs, patience):
        """Epochs with early stopping; returns (best epoch, best loss, best weights, curve)."""
        best, best_epoch, stale, curve = self._eval(model, d_va, g_va), 0, 0, []
        state = copy.deepcopy(model.state_dict())
        for epoch in range(1, max_epochs + 1):
            for rows in self._batches(g_tr, rng, device):
                self._step(model, opt, d_tr, rows)
            curve.append(self._eval(model, d_va, g_va))
            if curve[-1] < best:
                best, best_epoch, stale, state = curve[-1], epoch, 0, copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= patience:
                    break
        return best_epoch, best, state, curve

    def _pretrain(self, panel: Q3Panel, device, seed: int):
        """Fit on shock days before the window's last meeting; returns (weights, scaler, record)."""
        last = pd.Timestamp(np.asarray(panel.meeting).max())
        sub = self.pretrain.subset(pd.DatetimeIndex(self.pretrain.meeting) < last)
        inner, val = _inner_split(sub, self.val_frac)
        base = CAPMBetas(252, both=True)
        base.fit(inner)
        scaler = _Scaler(inner)
        d_in, d_va = self._data(inner, base, scaler, device), self._data(val, base, scaler, device)
        g_in, g_va = _meeting_index_groups(inner.meeting), _meeting_index_groups(val.meeting)
        seed_everything(seed)
        model = self._new_model(panel, device)
        opt = self._optimizer(model)
        epoch, loss, state, curve = self._train_best(model, opt, d_in, g_in, d_va, g_va, np.random.default_rng(seed),
                                                     device, self.pretrain_epochs, self.pretrain_patience)
        del d_in, d_va, model, opt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return state, scaler, dict(events=len(g_in) + len(g_va), epoch=epoch, val=round(loss, 6),
                                   curve=[round(c, 6) for c in curve])

    def fit(self, panel: Q3Panel, seed: int | None = None, **_) -> None:
        seed = self.seed if seed is None else int(seed)
        device = resolve_device(self.device)
        pre_state, pre_scaler, pre_info = (self._pretrain(panel, device, seed * 100 + 99) if self.pretrain is not None
                                           else (None, None, None))
        inner, val = _inner_split(panel, self.val_frac)
        base_in = CAPMBetas(252, both=True)
        base_in.fit(inner)
        self.base_inner_ = base_in
        sc_in = pre_scaler if pre_scaler is not None else _Scaler(inner)
        d_in, d_va = self._data(inner, base_in, sc_in, device), self._data(val, base_in, sc_in, device)
        g_in = _meeting_index_groups(inner.meeting)
        g_sel, used_big = self._selection_groups(val, _meeting_index_groups(val.meeting))
        steps_per_epoch = math.ceil(len(g_in) / self.meetings_per_batch)
        self.base = CAPMBetas(252, both=True)
        self.base.fit(panel)
        self.scaler = pre_scaler if pre_scaler is not None else _Scaler(panel)
        d_full, g_full = self._data(panel, self.base, self.scaler, device), _meeting_index_groups(panel.meeting)

        def start(member_seed):
            seed_everything(member_seed)
            model = self._new_model(panel, device)
            if pre_state is not None:
                model.load_state_dict(pre_state)
            return model.train()

        self.nets, selection, first = [], [], None
        for j in range(self.members):
            member_seed = seed * 100 + j
            model = start(member_seed)
            epoch, best, _, curve = self._train_best(model, self._optimizer(model), d_in, g_in, d_va, g_sel,
                                                     np.random.default_rng(member_seed), device, self.max_epochs,
                                                     self.patience)
            n_steps = epoch * steps_per_epoch
            model = start(member_seed)
            opt, rng, done = self._optimizer(model), np.random.default_rng(member_seed), 0
            while done < n_steps:
                for rows in self._batches(g_full, rng, device):
                    if done >= n_steps:
                        break
                    self._step(model, opt, d_full, rows)
                    done += 1
            model.eval()
            self.nets.append(model)
            selection.append(dict(epoch=epoch, steps=n_steps, val=round(best, 6)))
            if first is None:
                first = (curve, best, epoch)
        curve, best, epoch = first
        # folds.csv keeps only member_selection, so the selection and pretraining record ride in it
        record = dict(members=selection, selected_on_big=used_big, n_select_meetings=len(g_sel), pretrain=pre_info)
        self.fit_result = argparse.Namespace(train_loss=[float("nan")] * len(curve), val_loss=curve, best_epoch=epoch,
                                             best_val_loss=best, val_meetings=val.meetings(), member_selection=record)


# ----------------------------------------------------------------------------- runner

ARMS = ("F0", "F1", "F2", "F3", "F0s", "F1s")
PAIRS = (("F1", "F0"), ("F2", "F1"), ("F3", "F2"), ("F1s", "F0s"))


def makers(shock=None, vectors=None, device=None, members=3) -> dict:
    common = dict(members=members, device=device)
    return {
        "F0": lambda seed: V6Arm(seed, encoder="transformer", **common),
        "F1": lambda seed: V6Arm(seed, encoder="transformer", pretrain=shock, **common),
        "F2": lambda seed: V6Arm(seed, encoder="transformer", pretrain=shock, vectors=vectors, **common),
        "F3": lambda seed: V6Arm(seed, encoder="transformer", pretrain=shock, vectors=vectors, change=True, **common),
        "F0s": lambda seed: V6Arm(seed, encoder="summary", **common),
        "F1s": lambda seed: V6Arm(seed, encoder="summary", pretrain=shock, **common),
    }


def run(arms, through="2021-12-31", seeds=(0, 1, 2), members=3, device=None, store=None):
    """Development walk-forward on the v4 folds (min_train 60, test_size 16). Training."""
    from fedcore.q3.real import UNIVERSES
    from fedcore.q3.shock_days import load_shock_panel
    from fedcore.results import ResultStore
    from fedenc.config import RESULTS
    from fedenc.runs import save_runs
    from fedenc.v4 import panel_fingerprint
    from fedenc.workflow import bounded_panel

    if pd.Timestamp(through) > pd.Timestamp("2021-12-31"):
        raise ValueError("v6 development runs stop at 2021-12-31")
    panel, through_ts, _ = bounded_panel(through, with_rows=True)
    needs_shock = bool(set(arms) - {"F0", "F0s"})
    needs_text = bool(set(arms) & {"F2", "F3"})
    shock, shock_meta = None, None
    if needs_shock:
        shock, _, shock_meta = load_shock_panel()
    if shock is not None and pd.DatetimeIndex(shock.meeting).max() > through_ts:
        shock = shock.subset(pd.DatetimeIndex(shock.meeting) <= through_ts)
    vectors = None
    if needs_text:
        from fedenc.filing_vectors import FilingVectors
        vectors = FilingVectors.load()
    every = makers(shock, vectors, device=device, members=members)
    chosen = {a: every[a] for a in arms}
    datasets = list(UNIVERSES["sp500"].inputs) + (["gsw_yields"] if needs_shock else []) + \
        (["edgar_market_risk"] if needs_text else [])
    return save_runs(panel, chosen, prefix=f"v6-to{through_ts.year}", role="exploratory",
                     params=dict(through=str(through_ts.date()), panel_fingerprint=panel_fingerprint(panel),
                                 shock_panel=None if shock_meta is None else
                                 {k: shock_meta.get(k) for k in ("threshold_bp", "n_rows", "n_meetings", "inputs")},
                                 configs={a: m(0).config() for a, m in chosen.items()}, plan="docs/encoder/q3-v6-plan.md"),
                     min_train=60, test_size=16, seeds=seeds, store=store or ResultStore(RESULTS), datasets=datasets)


FINAL_ARM = "F0s"
DRIFT_DAYS = 20


def run_final(seeds=(6, 7, 8), members=3, device=None, store=None):
    """The one 2022-2025 test of F0s, with B0 (docs/encoder/q3-v6-final-plan.md). Training."""
    from fedcore.protocol import walk_forward
    from fedcore.q3.real import UNIVERSES
    from fedcore.results import ResultStore
    from fedenc.config import RESULTS
    from fedenc.runs import save_runs
    from fedenc.v4 import panel_fingerprint
    from fedenc.workflow import bounded_panel

    panel, through, _ = bounded_panel("2025-12-31", with_rows=True)
    meetings = panel.meetings()
    n_dev = int((meetings <= pd.Timestamp("2021-12-31")).sum())
    folds = walk_forward(meetings, n_dev, len(meetings) - n_dev)
    if len(folds) != 1 or folds[0].test.min() <= pd.Timestamp("2021-12-31"):
        raise RuntimeError("the final test must be one fold: train through 2021, test 2022-2025")
    chosen = {"B0": lambda seed: CAPMBetas(252, both=True), FINAL_ARM: makers(device=device, members=members)[FINAL_ARM]}
    return save_runs(panel, chosen, prefix="v6-final", role="robustness",
                     params=dict(through=str(through.date()), dev_end="2021-12-31", panel_fingerprint=panel_fingerprint(panel),
                                 configs={FINAL_ARM: chosen[FINAL_ARM](0).config(), "B0": {"model": "capm_beta_both"}},
                                 plan="docs/encoder/q3-v6-final-plan.md"),
                     min_train=n_dev, test_size=len(meetings) - n_dev, seeds=seeds, store=store or ResultStore(RESULTS),
                     datasets=UNIVERSES["sp500"].inputs)


def score_final(save: bool = True, n_boot: int = 5000):
    """The pre-stated final verdict (q3-v6-final-plan.md). Read-only apart from the score run."""
    from fedcore.q3 import beyond_beta as bb
    from fedenc.v4 import GUARDRAIL, _boot, _fmt, _load
    from fedenc.workflow import bounded_panel

    arm, b0 = _load("v6-final", FINAL_ARM), _load("v6-final", "B0")
    if arm is None or b0 is None:
        raise RuntimeError("need the saved v6-final F0s and B0 runs")
    p, run_id = arm
    panel, _, _ = bounded_panel("2025-12-31", with_rows=True)
    s = pd.Series(panel.shock, index=pd.DatetimeIndex(panel.meeting)).groupby(level=0).first()
    s = s[s.index > pd.Timestamp("2021-12-31")]
    prim, five = bb.score(p, s, n_boot=n_boot), bb.score(p, s, min_abs_s=0.5, n_boot=n_boot)
    d = p[["mu", "target"]].join(b0[0][["mu"]].rename(columns={"mu": "mu_b0"}), how="inner")
    raw = ((d["target"] - d["mu"]) ** 2 - (d["target"] - d["mu_b0"]) ** 2).groupby(level=0).mean()
    guard = bool(raw.mean() <= GUARDRAIL)
    dep = bb.departures(p)
    dep = dep[s.reindex(dep.index.get_level_values(0)).abs().to_numpy() >= PRIMARY]
    top = dep[dep.groupby(level=0)["dev"].transform(lambda x: x.abs().rank(pct=True)) > 0.9]
    hit = float((np.sign(top["dev"]) == np.sign(top["miss"])).mean()) if len(top) else np.nan
    if not guard:
        verdict = "fails the guardrail"
    elif prim["spread"] > 0 and prim["spread_lo"] > 0:
        verdict = "replicates"
    elif prim["spread"] > 0:
        verdict = "consistent, not confirmed"
    else:
        verdict = "does not replicate"

    # post-meeting drift (plan: second pre-stated measure): 20-day CAR, |s| >= 2.5bp
    from fedcore.q3 import drift
    from fedcore.q3.channels import abnormal_returns
    from fedcore.q3.real import _grid, load_real_panel, read_inputs

    _, rows, _ = load_real_panel()
    rows = rows[pd.DatetimeIndex(rows["meeting"]) > pd.Timestamp("2021-12-31")]
    base = read_inputs()
    grid = _grid(base.daily, base.market)
    car20 = drift.car_table(rows, abnormal_returns(grid), grid["calendar"], grid["permnos"], DRIFT_DAYS, "2025-12-31")
    dt = drift.drift_table(p, car20, s)
    dr = bb.summarize(dt, n_boot=n_boot)
    flip_p, _ = drift.sign_flip_p(dt)
    if dr["spread"] > 0 and dr["spread_lo"] > 0 and flip_p < 0.05:
        drift_verdict = "drift replicates"
    elif dr["spread"] > 0:
        drift_verdict = "drift consistent, not confirmed"
    else:
        drift_verdict = "drift does not replicate"
    f = lambda r, k, dd=2: f"{r[k]:+.{dd}f} ({r[k + '_lo']:+.{dd}f}, {r[k + '_hi']:+.{dd}f})"
    table = pd.DataFrame([dict(slice="|s| >= 2.5bp (primary)", meetings=prim["meetings"], spread=f(prim, "spread"),
                               claimed=round(prim["predicted_spread"], 2), rank_ic=f(prim, "rank_ic", 3),
                               calibration=f(prim, "calibration")),
                          dict(slice="|s| >= 5bp", meetings=five["meetings"], spread=f(five, "spread"),
                               claimed=round(five["predicted_spread"], 2), rank_ic=f(five, "rank_ic", 3),
                               calibration=f(five, "calibration")),
                          dict(slice=f"{DRIFT_DAYS}-day drift, |s| >= 2.5bp", meetings=dr["meetings"],
                               spread=f(dr, "spread"), claimed=np.nan, rank_ic=f(dr, "rank_ic", 3),
                               calibration=f"sign-flip p {flip_p:.3f}")])
    with pd.option_context("display.width", 300, "display.max_columns", 20, "display.max_colwidth", 40):
        print(table.to_string(index=False))
    print(f"top-decile hit rate {hit:.3f}; all-meeting MSE vs B0 {_fmt(_boot(raw))}; guardrail ok {guard}")
    print(f"VERDICT: {verdict}; {drift_verdict}")
    if save:
        from fedcore.questions import specs
        from fedcore.results import ResultStore
        from fedenc.config import RESULTS

        ResultStore(RESULTS).save(specs()["Q3"], "v6-final-score", role="robustness",
                                  params=dict(run=run_id, b0=b0[1], n_boot=n_boot, plan="docs/encoder/q3-v6-final-plan.md"),
                                  metrics=dict(spread=prim["spread"], spread_lo=prim["spread_lo"], spread_hi=prim["spread_hi"],
                                               guardrail_ok=int(guard), top_decile_hit=hit, drift_spread=dr["spread"],
                                               drift_lo=dr["spread_lo"], drift_hi=dr["spread_hi"], drift_flip_p=flip_p),
                                  tables=dict(final=table), notes=f"Final verdict: {verdict}; {drift_verdict}.")
    return verdict, drift_verdict, table


# ----------------------------------------------------------------------------- scoring

def score(prefix: str = "v6-to2021", arms=ARMS, base_prefix: str = "v4-to2021", save: bool = True, n_boot: int = 5000):
    """The v6 pass bar (plan section 1) from saved predictions. Read-only apart from the score run."""
    from fedcore.q3 import beyond_beta as bb
    from fedenc.v4 import GUARDRAIL, _boot, _fmt, _load
    from fedenc.workflow import bounded_panel

    loaded = {a: _load(prefix, a) for a in arms}
    loaded = {a: v for a, v in loaded.items() if v is not None}
    b0 = _load(base_prefix, "B0")
    if not loaded or b0 is None:
        raise RuntimeError("need saved v6 runs and the v4 B0 run")
    panel, _, _ = bounded_panel("2021-12-31", with_rows=True)
    s = pd.Series(panel.shock, index=pd.DatetimeIndex(panel.meeting)).groupby(level=0).first()
    slices = {"headline": dict(exclude=STANDOUT), "all_big": {}, "ge_5bp": dict(min_abs_s=0.5)}
    f = lambda r, k, d=2: f"{r[k]:+.{d}f} ({r[k + '_lo']:+.{d}f}, {r[k + '_hi']:+.{d}f})"
    rows, pairs = [], []
    for a, (p, run_id) in loaded.items():
        rec = dict(arm=a)
        for name, kw in slices.items():
            r = bb.score(p, s, n_boot=n_boot, **kw)
            rec[f"{name}_spread"] = f(r, "spread")
            if name == "headline":
                rec["headline_lo"] = r["spread_lo"]
                rec["claimed"] = round(r["predicted_spread"], 2)
                rec["rank_ic"], rec["calibration"] = f(r, "rank_ic", 3), f(r, "calibration")
                rec["meetings"] = r["meetings"]
        d = p[["mu", "target"]].join(b0[0][["mu"]].rename(columns={"mu": "mu_b0"}), how="inner")
        raw = ((d["target"] - d["mu"]) ** 2 - (d["target"] - d["mu_b0"]) ** 2).groupby(level=0).mean()
        rec["all_meetings_raw_vs_b0"] = _fmt(_boot(raw))
        rec["guardrail_ok"] = bool(raw.mean() <= GUARDRAIL)
        rec["run"] = run_id
        rows.append(rec)
    for x, y in PAIRS:
        if x in loaded and y in loaded:
            r = bb.paired(loaded[x][0], loaded[y][0], s, exclude=STANDOUT, n_boot=n_boot)
            pairs.append(dict(comparison=f"{x} - {y}", meetings=r["meetings"], spread=f(r, "spread"),
                              rank_ic=f(r, "rank_ic", 3), spread_lo=r["spread_lo"]))
    table, pair_t = pd.DataFrame(rows), pd.DataFrame(pairs)
    control = {x: y for x, y in PAIRS}
    pair_lo = {r["comparison"].split(" - ")[0]: r["spread_lo"] for r in pairs}
    table["passes"] = [bool(r["headline_lo"] > 0 and r["guardrail_ok"] and
                            (r["arm"] not in control or pair_lo.get(r["arm"], -np.inf) > 0)) for _, r in table.iterrows()]
    with pd.option_context("display.width", 300, "display.max_columns", 40, "display.max_colwidth", 40):
        print("== beyond-beta spread, bp per 10bp (headline: |s| >= 2.5bp without the standouts)")
        print(table.drop(columns=["run", "headline_lo"]).to_string(index=False))
        if len(pair_t):
            print("\n== paired, headline slice\n" + pair_t.drop(columns=["spread_lo"]).to_string(index=False))
    if save:
        from fedcore.questions import specs
        from fedcore.results import ResultStore
        from fedenc.config import RESULTS

        ResultStore(RESULTS).save(specs()["Q3"], f"{prefix}-score", role="exploratory",
                                  params=dict(prefix=prefix, base_prefix=base_prefix, runs={a: v[1] for a, v in loaded.items()},
                                              standout=[str(x.date()) for x in STANDOUT], n_boot=n_boot),
                                  metrics={f"{r['arm']}_passes": int(r["passes"]) for _, r in table.iterrows()},
                                  tables=dict(arms=table, pairs=pair_t),
                                  notes="Read-only scoring of saved v6 predictions (docs/encoder/q3-v6-plan.md section 1).")
    return table, pair_t


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arms", nargs="+", required=True, choices=list(ARMS))
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    r.add_argument("--members", type=int, default=3)
    r.add_argument("--device")
    sc = sub.add_parser("score")
    sc.add_argument("--prefix", default="v6-to2021")
    sc.add_argument("--no-save", action="store_true")
    fi = sub.add_parser("final", help="the one 2022-2025 test of F0s (training: needs the user's go-ahead)")
    fi.add_argument("--device")
    fs = sub.add_parser("score-final")
    fs.add_argument("--no-save", action="store_true")
    a = p.parse_args(argv)
    if a.command == "run":
        run(a.arms, seeds=tuple(a.seeds), members=a.members, device=a.device)
    elif a.command == "final":
        run_final(device=a.device)
    elif a.command == "score-final":
        score_final(save=not a.no_save)
    else:
        score(prefix=a.prefix, save=not a.no_save)


if __name__ == "__main__":
    main()
