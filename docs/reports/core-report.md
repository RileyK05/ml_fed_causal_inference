# Core report: Q3 foundation

Package 00 from `docs/core/HANDOFF.md`. Synthetic panel, shared model contract, trainer, walk-forward harness, and the four baselines later packages have to beat. Scope stayed inside `core/`. Encoder, CDE, and DML were left alone.

The bar is `ridge_interact`. On easy noise it leads on per-meeting MSE and on recovery of the planted sensitivity `b`. On realistic noise the return MSE sits close to the noise floor for every model that sees the shock, and ridge still leads on `b`.

## 1. What was built

| File | What it is |
|---|---|
| `core/fedcore/q3/__init__.py` | Public API. Importing this package loads torch. `fedcore/__init__.py` does not import it. |
| `core/fedcore/q3/contracts.py` | `Q3Panel`, `CHANNELS`, `L=252`, `HistoryEncoder`, `SummaryEncoder` (default `out_dim=32`). |
| `core/fedcore/q3/synthetic.py` | Deterministic firm-meeting DGP, `sequence_pattern`, `synthetic_components`, `make_pretrain_windows`. Nothing is written under `core/data`. |
| `core/fedcore/q3/features.py` | 48 summary features: per channel, trailing mean / vol / drawdown over 21, 63, and 252 days, plus last `beta63`, compounded max drawdown, and 63-day momentum. |
| `core/fedcore/q3/heads.py` | `ShockHead`: `mu = a(z) + b(z) * s`. Two linear maps. |
| `core/fedcore/q3/model.py` | `ResponseModel`. Shock enters only the head. `z_dim=64`, `branch_dim=16`, `dropout=0.1`. |
| `core/fedcore/q3/train.py` | Training-only scaler, equal-meeting loss, AdamW, grad clip 1.0, early stopping, best-checkpoint restore. |
| `core/fedcore/q3/evaluate.py` | `run_walk_forward`, `compare`, `save`, `run_arms`. One scoring path for every arm. |
| `core/fedcore/q3/baselines.py` | `pooled`, `ridge_interact`, `lgbm`, `summary_nn` in `BASELINES`. |
| `core/tests/test_q3_foundation.py` | 16 tests. File starts with `pytest.importorskip("torch")`, then lightgbm and scikit-learn. |
| `core/pyproject.toml` | Added only the `q3` extra: `torch>=2.2`, `lightgbm>=4.0`, `scikit-learn>=1.4`. |
| `core/results/q3_susceptibility/` | Eight append-only exploratory runs from the scored benchmark. |

`fedcore check`: 16/16 datasets pass.

`pytest core --basetemp C:\ptq3 -q`: 40 passed in 8.13s. That includes `test_isolation.py` and the 16 Q3 tests.

## 2. How to reproduce

Environment that produced the table: Python 3.12.10, torch 2.14.1+cpu, lightgbm 4.7.0, scikit-learn 1.9.1, pandas 3.0.6. CUDA was unavailable. From the repo root, with the root venv:

```powershell
.\.venv\Scripts\python -m pip install -e "core[dev,q3]"
.\.venv\Scripts\fedcore check
.\.venv\Scripts\python -m pytest core --basetemp C:\ptq3 -q
```

The scored table is this call, once per noise, with library defaults (`fit_kwargs` empty). `ResultStore` appends a new run id each time, so a second invocation writes eight more runs beside the ones below.

```python
from fedcore.config import ROOT
from fedcore.q3 import BASELINES, run_arms
from fedcore.results import ResultStore

store = ResultStore(ROOT / "results")
for noise in ("easy", "realistic"):
    run_arms(
        BASELINES,
        store,
        data="synthetic",
        noise=noise,
        seeds=(0, 1, 2),
        min_train=80,
        test_size=16,
        n_firms=200,
        n_meetings=160,
        seed=0,
    )
```

Wall clock on this machine was 872 seconds. Easy noise took 471s and realistic took 397s. `summary_nn` was almost all of that. Saved folds show its best epoch between 10 and 56 on easy noise and between 8 and 31 on realistic noise. The 100-epoch cap was never hit.

Runs, all `role="exploratory"`, question `Q3`, key `q3_susceptibility`. `window_shrunk` is false, `min_train` is 80, `test_size` is 16, panel seed 0, 200 firms, 160 meetings, 32,000 rows. Each run has 5 expanding folds, 80 test meetings, 3 seeds, and 48,000 prediction rows. `datasets` is empty because the panel is synthetic and is not a catalog dataset.

| noise | arm | run id |
|---|---|---|
| easy | pooled | `20261003-173122_pooled` |
| easy | ridge_interact | `20261003-173211_ridge-interact` |
| easy | lgbm | `20261003-173343_lgbm` |
| easy | summary_nn | `20261003-173901_summary-nn` |
| realistic | pooled | `20261003-173913_pooled` |
| realistic | ridge_interact | `20261003-174001_ridge-interact` |
| realistic | lgbm | `20261003-174130_lgbm` |
| realistic | summary_nn | `20261003-174537_summary-nn` |

## 3. Results

Spread is the sample standard deviation across seeds 0, 1, and 2 (`ddof=1`). A reported 0 means the three seeds matched. `pooled` and `ridge_interact` are closed form, so their seed spread is 0. RankIC is the mean Spearman of `mu` vs `target` inside each test meeting. It is undefined when the prediction is constant inside the meeting. Those meetings stay in the MSE. `rank_ic_defined_frac` is 0 for pooled and 1 for the other three arms.

Primary metric is `mse_per_meeting`: the mean, over the 80 test meetings, of within-meeting MSE. With 200 firms in every meeting that equals the row-level MSE. `b_corr` and `b_rmse` compare `b_hat` to the planted `b` on the test rows.

Planted residual of `a + b*s`, same panel seed, measured on all 32,000 rows: mean squared residual 0.0398 on easy noise (eps sd 0.2) and 2.241 on realistic noise (eps sd 1.5). A perfect `a` and `b` would land there.

### noise = easy

| arm | mse_per_meeting | centered_mse | rank_ic | b_corr | b_rmse | a_rmse |
|---|---|---|---|---|---|---|
| pooled | 0.16175 ± 0 | 0.13064 ± 0 | undefined | −0.0553 ± 0 | 0.9122 ± 0 | 0.1861 ± 0 |
| ridge_interact | 0.05534 ± 0 | 0.05474 ± 0 | 0.6241 ± 0 | 0.9027 ± 0 | 0.3934 ± 0 | 0.0145 ± 0 |
| lgbm | 0.05867 ± 0.00014 | 0.05157 ± 0.00014 | 0.6316 ± 0.0011 | 0.8793 ± 0.0006 | 0.4416 ± 0.0010 | 0.2324 ± 0.0009 |
| summary_nn | 0.05965 ± 0.00051 | 0.05711 ± 0.00009 | 0.6183 ± 0.0002 | 0.8827 ± 0.0032 | 0.4286 ± 0.0053 | 0.0421 ± 0.0109 |

Paired per-meeting MSE, arm minus reference, seeds averaged first, then a meeting bootstrap (2,000 draws, 80 meetings). A negative number means the arm has the lower error.

| arm | minus pooled | minus ridge_interact |
|---|---|---|
| pooled | 0 | +0.10641 [+0.08241, +0.13305] |
| ridge_interact | −0.10641 [−0.13305, −0.08241] | 0 |
| lgbm | −0.10308 [−0.12802, −0.07977] | +0.00333 [+0.00045, +0.00699] |
| summary_nn | −0.10210 [−0.12756, −0.07925] | +0.00431 [+0.00323, +0.00553] |

Ridge's MSE advantage over both learned models has a confidence interval that stays positive. The gap (0.003 to 0.004) is larger than either model's seed spread.

### noise = realistic

| arm | mse_per_meeting | centered_mse | rank_ic | b_corr | b_rmse | a_rmse |
|---|---|---|---|---|---|---|
| pooled | 2.36193 ± 0 | 2.31717 ± 0 | undefined | −0.0282 ± 0 | 0.9107 ± 0 | 0.1849 ± 0 |
| ridge_interact | 2.27202 ± 0 | 2.25562 ± 0 | 0.1301 ± 0 | 0.8663 ± 0 | 0.4578 ± 0 | 0.0798 ± 0 |
| lgbm | 2.28705 ± 0.00153 | 2.26526 ± 0.00176 | 0.1222 ± 0.0023 | 0.7870 ± 0.0025 | 0.5769 ± 0.0041 | 0.2917 ± 0.0047 |
| summary_nn | 2.26661 ± 0.00180 | 2.25069 ± 0.00030 | 0.1366 ± 0.0013 | 0.8583 ± 0.0013 | 0.4681 ± 0.0012 | 0.0630 ± 0.0140 |

| arm | minus pooled | minus ridge_interact |
|---|---|---|
| pooled | 0 | +0.08990 [+0.06612, +0.11518] |
| ridge_interact | −0.08990 [−0.11518, −0.06612] | 0 |
| lgbm | −0.07487 [−0.09818, −0.05199] | +0.01503 [+0.00624, +0.02460] |
| summary_nn | −0.09531 [−0.12091, −0.07174] | −0.00541 [−0.01144, +0.00020] |

`summary_nn` has the lowest realistic MSE point estimate. The interval against ridge includes 0, so that edge is not resolved. Ridge remains ahead on `b_corr` (0.8663 vs 0.8583 ± 0.0013) and on `b_rmse` (0.4578 vs 0.4681 ± 0.0012). LightGBM is behind ridge on both MSE and `b`.

### Reading the table

Mean planted `b` is 3.21 with sd 0.91. The shock in this draw has sd 0.35 (the generator's sd is 0.3). A single global loading, which is all `pooled` has, already fits the common piece `b_mean * s`. That is why pooled MSE (2.362) is only about 0.12 above the realistic noise floor (2.241), and why its within-meeting RankIC is undefined: `s` is constant inside a meeting, so `mu` is too. `centered_mse` for pooled is the within-meeting variance of the target.

The cross-section is where the models separate, and return MSE has little room left once the shock is in the regression. Ridge sits 0.015 above the easy floor and 0.031 above the realistic floor. That 0.031 is the whole gap between ridge and the planted conditional mean, which is the most any later model can gain on realistic per-meeting MSE. `b_corr` and `b_rmse` are the metrics with headroom. Ridge's realistic `b_rmse` of 0.46 against a `b` sd of 0.91 is the number a sequence encoder can actually beat.

LightGBM's `a_rmse` is the worst of the four (0.23 easy, 0.29 realistic), worse than pooled. `b_hat` is a central finite difference at `s±0.5`, and `a_hat = mu - b_hat * s`. The tree is not linear in `s`, so that split does not recover a structural intercept. Use LightGBM's `b_hat` and its return MSE. Treat its `a_hat` as a poor attribution.

No hyperparameter was changed after these numbers. Ridge `alpha=1.0` and the LightGBM settings below were fixed in `baselines.py` before this run. The trainer defaults (`lr=1e-3`, `weight_decay=1e-4`, `max_epochs=100`, `patience=10`) were left as specified.

### DGP on the scored panel

Same panel the table uses (`n_firms=200`, `n_meetings=160`, `seed=0`), measured after the runs and not used to change anything.

- Sequence-pattern rate: 0.0936.
- `b` sd: 0.906. Component variance shares: leverage term 0.300, trailing-vol term 0.370, pattern term 0.233. Shares sum to 0.90 because the pieces covary.
- `b = 0.5 + 2.0 * leverage + 0.9 * rvol63 + 1.5 * pattern`.
- `a = 0.30 * (leverage - 0.45) + 0.04 * (rate - 2) + 0.03 * (vix - 18)`.

The pattern is a real minority event and about a quarter of the variance of `b`. Summary features see drawdown and momentum, which correlate with the pattern, and they do not encode the conjunction the DGP uses. That leftover is what a sequence model is for. Ridge already reaches `b_corr` 0.90 on easy noise from the linear pieces, so the sequence arm has to clear a high bar.

## 4. Deviations from the brief

1. **Small-panel window.** `run_arms` with both `min_train` and `test_size` left at their defaults shrinks the 80/16 window when the panel has fewer than 80 meetings, and records `window_shrunk` in the params. The mandated end-to-end test uses 40 meetings and needs a fold. The scored runs pass 80 and 16 explicitly. `window_shrunk` is false on all eight manifests.
2. **Partial recovery.** The brief says a drawdown followed by a partial recovery. The code uses a compounded peak-to-trough drawdown above 15% in the last 60 days, the trough before the last day, and a recovery of at least one third of the loss (`RECOVERY_FRACTION = 1/3`).
3. **Finite observed days.** `beta63` falls back to the firm's true beta when fewer than 10 days are observed or the market is flat. `rvol21`, and the `rvol63` inside `b`, fall back to 1.5 when fewer than two days are observed. An observed day would otherwise be NaN and fail the mask check. A fully observed trailing window is a real estimate. The unit test locks `beta63` and `rvol21` on that case.
4. **RankIC.** Undefined RankIC is stored as JSON null, with `rank_ic_defined_frac` beside it. Meetings are kept in the MSE.
5. **Scaler.** Median and IQR are fit on the early-stopping training meetings only. The last 15% of the training fold is the validation tail and is excluded. Fundamentals and context are scaled the same way. Shock and target are not scaled. Unobserved history entries become 0 after scaling and are clipped at ±5. The encoder must still ignore masked positions when those entries are nonzero. The mask-invariance test covers that.
6. **Summary encoder width.** `out_dim` defaults to 32. The brief requires an `out_dim` and a mask-safe forward. 32 is the choice used by `summary_nn`.
7. **Ridge.** `alpha=1.0`, training-fold median impute, then z-score of the summary-plus-fundamentals-plus-context block only. `s` itself is not standardized. `sklearn` `Ridge` keeps its default intercept. `a_hat` is rebuilt so `a_hat + b_hat * s` equals `mu`.
8. **LightGBM, fixed before any scored metric.** `n_estimators=300`, `learning_rate=0.05`, `num_leaves=15`, `min_child_samples=20`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`, `n_jobs=1`, `deterministic=True`, `verbosity=-1`. `b_hat = predict(s+0.5) - predict(s-0.5)`.
9. **Batching.** Each step is a batch of 4 meetings. A shorter tail batch scales the loss by `n_meetings / 4`, so each meeting's gradient has the same weight. The trainer steps every batch.
10. **Checkpoint.** Curves are eval-mode meeting MSE. The weights restored for prediction are the epoch with the lowest validation loss. There is no second fit on train+validation. `docs/encoder/transformer.md` §11 describes that refit after architecture selection. This package follows the handoff, which says to restore the best checkpoint. Selection of an architecture is not this package's job.
11. **`by_seed` table.** Saved next to `predictions`, `folds`, and `curves`.
12. **Pattern access.** `sequence_pattern` and `synthetic_components` are public so later packages can split rows without re-deriving the formula. `Q3Panel` has no pattern column.
13. **CPU.** The scored `summary_nn` ran on CPU. The trainer uses whatever device the module parameters are on.
14. **`compare` sign.** First result minus the baseline. The harness passes `meeting_col="meeting"` because the protocol default column is `announcement_date`.
15. **Branch.** The agent guide names `q3/foundation`. These files are uncommitted on `restructure/four-projects`. That branch already carried an unrelated dirty tree, so this work was not moved onto a new branch.

## 5. Requested contract changes

None. The encoder and CDE handoffs can import `HistoryEncoder`, `ResponseModel`, `fit` (`freeze_encoder`, `encoder_lr`), `make_pretrain_windows`, `BASELINES`, and `run_arms` as written.

Two choices are easy to mistake for gaps:

- Row-level pattern labels come from `synthetic_components` or `sequence_pattern`. They are not columns of `Q3Panel`.
- The scored network is the early-stopped checkpoint. A train+validation refit would be a change to `fit`, and it should be asked for explicitly.

## 6. Open questions and weaknesses

- **The real panel does not exist.** `run_arms(..., data="real")` raises `NotImplementedError("real Q3 panel not built yet")`. Nothing here is a claim about markets.
- **Return MSE will barely move.** On realistic noise the oracle residual is 2.241 and ridge is at 2.272. Beating ridge by a tenth of an MSE point would mean a model better than the planted conditional mean. Judge sequence models on `b_corr` and `b_rmse`, and require the MSE gap over ridge to clear the meeting-bootstrap interval before calling it a win. The realistic `summary_nn` interval does not.
- **`summary_nn` is a weak control on this DGP.** It loses to ridge on easy MSE, easy `b_corr`, and realistic `b_corr`. Matching the summary net is a low bar. The published bar is ridge: easy `b_corr` 0.9027 and `b_rmse` 0.3934, realistic `b_corr` 0.8663 and `b_rmse` 0.4578, with the MSE numbers in the tables above.
- **The pattern rate is 9.4%.** That is 23% of `b` variance, concentrated in a minority of rows. A sequence encoder can look fine on average `b_corr` while missing the only term it was built to get. Split `b_corr` by `synthetic_components(panel)["pattern"]` when those models are scored. This package does not do that split in the saved metrics.
- **LightGBM `a_hat` is not comparable to ridge's `a_hat`.** The finite difference is the specified `b_hat`. The level soaked up by `a_hat` is not a clean intercept.
- **Pooled RankIC is null by construction.** A report that drops undefined meetings, or that fills them with zero, will invent a RankIC for pooled. The manifests keep it null.
- **Same seed, two noise levels.** History, shock, fundamentals, context, `a`, and `b` match. Only the target noise changes. `eps` is drawn last as a standard normal times the noise sd.
- **No tuning grid was run.** There is nothing to declare beyond the constants already in `baselines.py` and `train.py`. Leave them there. The easy-noise ridge recovery test (`b_corr > 0.5` on a small panel) passed without strengthening the DGP.
