# Package 00: Q3 shared foundation

**Goal:** build the contract every Q3 model plugs into. It covers the data shape, a synthetic
panel with known answers, the shared response head and loss, a meeting-batched trainer, the
walk-forward scoring harness and the non-neural baselines. Packages 01 (encoder) and 02
(Neural CDE) depend on this exact API. Keep it small, typed and boring.

Read first: `AGENTS.md`, `docs/question.md` (Q3 section), `docs/encoder/transformer.md`
sections 1, 6, 9-12, `core/fedcore/protocol.py`, `core/fedcore/results/store.py`.

## Where it lives and why

This is the one piece of Q3 code shared by the encoder and CDE projects, so it lives in the
shared layer: **`core/fedcore/q3/`**. The projects import it; they never import each other.
`core/tests/test_isolation.py` enforces that, and `fedcore.q3` itself may only import
`fedcore` (data, protocol, results, questions).

## Files you own

```text
core/fedcore/q3/
    __init__.py
    contracts.py      # Q3Panel, HistoryEncoder protocol, channel names
    synthetic.py      # make_synthetic_panel(...), make_pretrain_windows(...)
    features.py       # hand-built summary features from history (for baselines)
    heads.py          # ShockHead: mu = a(z) + b(z) * s
    model.py          # ResponseModel = encoder + fundamentals MLP + context MLP + fusion + head
    train.py          # fit(): meeting batches, per-meeting MSE, early stopping
    evaluate.py       # run_walk_forward(): folds, predictions, metrics, ResultStore
    baselines.py      # pooled OLS, ridge-with-interactions, LightGBM, summary_nn
core/tests/test_q3_foundation.py
core/pyproject.toml   # add a `q3` extra: torch, lightgbm, scikit-learn (only this edit)
```

Torch must stay **optional** for the rest of fedcore: nothing outside `fedcore/q3/` may import
torch, and `test_q3_foundation.py` starts with `pytest.importorskip("torch")`. CPU-only must
work. Install with `pip install -e "core[dev,q3]"`.

Runs are saved under the Q3 definition: `SPEC = fedcore.questions.specs()["Q3"]`. The harness
never chooses a results folder itself. The calling project passes its own `ResultStore`
(`encoder/results/`, `cde/results/`).

## 1. `contracts.py`: the panel

One row = one (firm, meeting). All arrays are aligned on axis 0, length N.

```python
CHANNELS = ("ret", "dlogvol", "rvol21", "ret_rel", "beta63")   # history channel order, fixed
L = 252                                                         # trading days of history

@dataclass(frozen=True)
class Q3Panel:
    meeting: np.ndarray          # (N,) datetime64[ns]  scheduled FOMC announcement date
    firm_id: np.ndarray          # (N,) str or int
    history: np.ndarray          # (N, L, C) float32  right-aligned: history[:, -1] = last
                                 #   trading day BEFORE the meeting (prior close). NaN = unobserved
    history_mask: np.ndarray     # (N, L) bool  True = observed. Padding and missing days are False
    fundamentals: np.ndarray     # (N, F) float32, NaN allowed
    fundamentals_mask: np.ndarray  # (N, F) bool
    context: np.ndarray          # (N, K) float32, identical for all firms in a meeting
    shock: np.ndarray            # (N,) float32  surprise S in units of 10bp, same for all firms in a meeting
    target: np.ndarray           # (N,) float32  return in percent, prior close -> event close
    a_true: np.ndarray | None = None   # synthetic only
    b_true: np.ndarray | None = None   # synthetic only

    def subset(self, rows: np.ndarray) -> "Q3Panel": ...
    def meetings(self) -> pd.DatetimeIndex: ...       # sorted unique
    def validate(self) -> None: ...                   # shapes, dtypes, shock constant within meeting,
                                                      # context constant within meeting, mask/NaN agree
```

`history` holds **unscaled** values. Scaling happens inside the trainer, using statistics
fitted on training rows only (see section 5).

### The encoder interface (the one boundary both arms implement)

```python
class HistoryEncoder(torch.nn.Module):
    out_dim: int
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """x: (B, L, C) float, already scaled, unobserved entries = 0.
        mask: (B, L) bool, True = observed.
        returns h: (B, out_dim). Must not depend on values where mask is False."""
```

Ship one trivial reference implementation, `SummaryEncoder`: masked mean and std per channel
over the last 21 / 63 / 252 days, followed by a linear layer. It is the plug-in sanity check and
the "no sequence model" control.

## 2. `synthetic.py`: a panel with known answers

`make_synthetic_panel(n_firms=200, n_meetings=160, noise="realistic", seed=0) -> Q3Panel`

Requirements, in order of importance:

1. **The true b depends on the history in a way that summary features only partly capture.**
   For example, build b_true from three parts:
   - a fundamentals part (leverage);
   - a level part (trailing volatility, which summary features can see);
   - a **sequence part** (e.g. "a drawdown of more than 15% within the last 60 days followed by
     a partial recovery"; a pattern in time, not a level).

   Document the formula in the docstring. An encoder that beats `SummaryEncoder` is evidence
   that it learned the sequence part.
2. **Firm daily returns:** a market factor, plus firm beta, plus idiosyncratic noise with
   regime-switching volatility. Volume follows volatility. Channels are computed exactly as
   named in `CHANNELS` (beta63 = rolling 63-day beta to the market factor). Firms get random
   listing dates, so some histories are left-padded and masked. Drop about 1% of days at
   random to exercise the mask.
3. **Meetings:** 160 dates about 6 weeks apart. Shock s_t ~ N(0, 0.3), meaning typical surprises
   of about 3bp. Context is about 4 meeting-level series (a rate level, a VIX-like volatility,
   etc.).
4. **Target:** r = a_true + b_true * s + eps. `noise="easy"` gives eps sd 0.2 (for debugging).
   `noise="realistic"` gives eps sd about 1.5, close to real daily returns, where most of the
   variation is noise.
5. Fully deterministic given `seed`. Nothing is written to `data/`; the panel is generated in
   memory.

`make_pretrain_windows(n_windows, seed) -> (history, mask)`: ordinary-day windows from the
same generator, with no meeting and no target. Used by package 01's pre-training.

## 3. `features.py`

`summary_features(panel_or_history, mask) -> np.ndarray`: trailing mean/vol/drawdown per
channel over 21/63/252 days, last beta63, max drawdown, momentum. These feed the baselines.
They are the "engineered characteristics" the deep models must beat.

## 4. `heads.py` and `model.py`

```python
class ShockHead(nn.Module):            # linear readout, per docs/encoder/transformer.md section 6A
    def __init__(self, z_dim): self.a = nn.Linear(z_dim, 1); self.b = nn.Linear(z_dim, 1)
    def forward(self, z, s): a = self.a(z).squeeze(-1); b = self.b(z).squeeze(-1); return a + b * s, a, b

class ResponseModel(nn.Module):
    """encoder(history) -> h ; MLP(fundamentals ++ fund_mask) -> f ; MLP(context) -> c ;
    fusion MLP(concat[h, f, c]) -> z (z_dim=64) ; ShockHead(z, s) -> (mu, a, b)."""
    def __init__(self, encoder: HistoryEncoder, n_fund, n_ctx, z_dim=64, branch_dim=16, dropout=0.1): ...
```

S never enters the encoder or the fusion. Only the head sees it.

## 5. `train.py`

`fit(model, train_panel, *, val_frac=0.15, lr=1e-3, encoder_lr=None, weight_decay=1e-4,
max_epochs=100, patience=10, seed=0, freeze_encoder=False) -> FitResult`

- **Validation** = the chronologically last `val_frac` of the *training* meetings. Never the
  test fold.
- **Scaling:** per-channel median/IQR (or mean/sd) fitted on training rows' observed entries
  only, clipped at ±5. Unobserved entries are then set to 0. The fitted scaler is stored and
  reused for validation and test.
- **Batches are whole meetings** (all firms of 1-4 meetings per batch).
- **Loss = mean over meetings of the per-meeting MSE** (`docs/encoder/transformer.md` section 10). Every
  meeting weighs the same regardless of firm count.
- `encoder_lr` lets packages 01 and 02 use a smaller learning rate for the encoder.
  `freeze_encoder=True` trains only the branches, fusion and head.
- Gradient clipping at 1.0. Early stopping on validation loss; restore the best checkpoint.
- Returns the trained model, the scaler, the loss curves and the best epoch.

## 6. `evaluate.py`: the harness everyone is scored by

`run_walk_forward(panel, make_model, *, min_train=80, test_size=16, seeds=(0,1,2),
fit_kwargs=None) -> EvalResult`

- Folds from `fedcore.protocol.walk_forward(panel.meetings(), min_train, test_size)`.
- For each fold and seed: build a fresh model with `make_model(seed)`, `fit` on the train
  meetings, predict on the test meetings.
- **Prediction table**, one row per test (firm, meeting): `meeting, firm_id, fold, seed, mu,
  a_hat, b_hat, target, gap = target - mu` plus `a_true, b_true` when present.
- **Metrics, per seed and averaged:**
  - `mse_per_meeting` (primary): the mean over test meetings of the within-meeting MSE
  - `centered_mse`: within-meeting demeaned target vs demeaned mu (firm differentiation only)
  - `rank_ic`: mean per-meeting Spearman of mu vs target
  - synthetic only: `b_corr` = corr(b_hat, b_true), `b_rmse`, `a_rmse`
- `compare(result, baseline_result)` → paired per-meeting MSE difference with a
  `meeting_bootstrap` CI.
- `save(result, store, name, params)` → `ResultStore.save(SPEC, name, role="exploratory",
  params=..., metrics=..., tables={"predictions": ..., "folds": ..., "curves": ...})`.

Baselines and neural models go through the **same** function. No model gets its own scoring
code.

## 7. `baselines.py`

Each one exposes the same `fit`/`predict` interface the harness uses (wrap with a small adapter
if non-torch):

| Name | Model |
|---|---|
| `pooled` | r = a + b·s (one a, one b for everything); Q1's analogue |
| `ridge_interact` | ridge on [features, fundamentals, context, s·features, s·fundamentals, s·context, s]; b_hat = the s-coefficient part evaluated per row |
| `lgbm` | LightGBM on the same columns plus s; b_hat by finite difference (predict at s+0.5 and s−0.5) |
| `summary_nn` | `ResponseModel(SummaryEncoder(...))` through the torch trainer |

`summary_nn` is the key control. It's the same network as the deep arms with the sequence model
removed.

## 8. How projects run things

There's no central arm registry (that would make core import the projects). Instead:

- `fedcore.q3.baselines.BASELINES: dict[str, Callable[[seed], model]]` exposes the baselines.
- `fedcore.q3.evaluate.run_arms(arms: dict[str, Callable], store, *, data="synthetic",
  noise="realistic", seeds=(0, 1, 2), **panel_kwargs)` builds the panel, runs every arm
  through `run_walk_forward`, saves each to `store`, and returns a comparison table.
  `data="real"` scores the cached WRDS panel (`fedcore.q3.real`) and requires explicit
  `min_train` and `test_size`; it is training and needs the user's go-ahead.
- Each project calls `run_arms({**BASELINES, **its_own_arms}, ResultStore(its RESULTS))` from
  its own script (no `run.py` is kept: arms live in each project's `arms.py`). Baselines are re-run per project (they're cheap), so each project's
  results folder is self-contained.

## 9. Tests (`core/tests/test_q3_foundation.py`)

- `Q3Panel.validate` catches: a shock varying within a meeting, a mask/NaN mismatch, wrong
  shapes.
- Synthetic determinism: same seed gives identical arrays.
- Scaler leakage: fitting on train rows then changing test-row values leaves the scaler
  unchanged.
- Mask invariance for `SummaryEncoder`: changing values where mask=False leaves h unchanged.
- Walk-forward: no meeting appears in both train and test of a fold; validation meetings
  precede test meetings.
- Recovery smoke test (`noise="easy"`, small panel): `ridge_interact` reaches `b_corr > 0.5`.
- End to end: `run_arms({"pooled": BASELINES["pooled"]}, ResultStore(tmp_path), n_firms=20,
  n_meetings=40)` saves a run.

## Exit criteria

- All of the above implemented, the tests pass, and `fedcore check` + `pytest core` are green
  (`core/tests/test_isolation.py` included).
- Report a table of all baselines on `noise="easy"` and `noise="realistic"`, 3 seeds: the
  numbers packages 01 and 02 have to beat.
- The report is written to `docs/reports/core-report.md`.
