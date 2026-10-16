# Plan: grade b directly, then rerun the pretraining comparison

> Archived 2026-10-16: the code this describes was removed; see git tag `q3-pretraining-archive` and [encoder-report.md](../../reports/encoder-report.md).

Written 2026-10-14 for an implementation session. Code and unit tests only: **no training,
no `run_arms`, no pretraining, nothing that saves a run.** The user launches the GPU run.
Read `AGENTS.md`, `docs/encoder/next-return-pretraining.md` and
`docs/encoder/pretraining-status.md` first. Code is the source of truth over docs.

## Why (facts from the saved first-trial runs, computed without fitting)

Runs: `encoder/results/q3_susceptibility/20261014-153152_tx-scratch`, `..-153526_tx-pre-full`,
`..-154006_tx-masked-full` (development period, 60 held-out meetings, 3 seeds).

- Shock `s` (10bp units): std 0.39. Pooled b is -1.0 %/10bp; it explains 9% of the
  meeting-level move and ~2% of a firm's Fed-day return.
- Firm-to-firm differences in b_hat times s explain **0.05%** of the within-meeting return
  variance. Scratch's cross-firm b is worth 0.01 MSE out of 9.1. The primary metric
  (mean per-meeting MSE) cannot reward learning b; cheap test R1 said the same.
- Graded on b directly (slope of the held-out return on centered b_hat × s, within meeting):
  scratch +2.3 (t 1.5), next-day pretrained -4.1 (t -1.2), masked +1.2 (t 0.6). Nothing
  significant; scratch is the only one pointing the right way.
- MSE gain over predicting zero is ~0.6 for scratch, of which ~0.58 is the meeting level
  (pooled b·s plus the context-driven level). Cross-firm a adds 0.03, cross-firm b 0.01.
- Full-day change in the 1-year yield as an alternative, larger shock: std 0.64 but
  correlation with the Fed-day market move is 0.00. Rejected; the 30-minute surprise stays.
- The Gaussian next-day objective learned volatility (val loss 3.346 vs 3.401 for "mean 0,
  trailing variance"). The next-day mean is unpredictable from price history; volatility is
  what carries over to Fed days (cheap tests R11, R6). So the Gaussian objective is the
  candidate again; `next_return_scaled` (mean only) is a check, expected to land at scratch.

## Task 1: b-calibration metric in the shared evaluator (core)

Files: `core/fedcore/q3/evaluate.py`, `core/tests/` (find the existing evaluate tests).
Shared because every Q3 arm (baselines, encoder, CDE) must report it.

1. `_prediction_frame`: add a `shock` column (`panel.shock`, float64). Keep all existing columns.
2. New helper `b_calibration(df) -> dict`, given one seed's predictions:
   - `yc = target - within-meeting mean`, `bc = b_hat - within-meeting mean`, `x = bc * shock`.
   - `b_calib_slope = sum(x*yc) / sum(x*x)`. 1 = b_hat calibrated, 0 = noise.
   - `b_calib_se`: meeting-clustered (sum over meetings of `sum(x*resid)`, squared, summed,
     sqrt, divided by `sum(x*x)`), `b_calib_t = slope / se`.
   - `b_quintile_slope`: per meeting, mean return of the top minus bottom b_hat quintile
     (rank with `method="first"`, `pd.qcut` into 5), regressed through the origin on
     `shock × (top-minus-bottom mean b_hat)` across meetings.
   - `b_var_share = var(x) / var(yc)`.
   - `b_hat_within_std = bc.std()`.
   - Guard: if `sum(x*x) == 0` (constant b_hat or all-zero shocks) return NaN for the slopes.
3. New helper `mse_decomposition(df) -> dict` (per meeting then averaged, like `_meeting_mse`):
   - `mse_zero` (predict 0), `mse_meeting_level` (predict the within-meeting mean of `mu`),
     `mse_a_only` (`a_hat + mean_b·shock`, mean_b within meeting),
     `mse_b_only` (`mean_a + b_hat·shock`), `mse_full` (= `mse_per_meeting`).
   - Gains: `gain_meeting_level = mse_zero - mse_meeting_level`;
     `gain_cross_firm_a = mse_meeting_level - mse_a_cross`, where `mse_a_cross` predicts
     `meeting mean of mu + (a_hat - meeting mean of a_hat)`;
     `gain_cross_firm_b = mse_meeting_level - mse_b_cross`, where `mse_b_cross` predicts
     `meeting mean of mu + (b_hat - meeting mean of b_hat)·shock`.
     The three gains add up to roughly `mse_zero - mse_full` (cross terms are small).
4. `_metrics_one`: merge both dicts in. `_average_metrics` averages them over seeds like the
   rest (`_METRIC_COLS` may need the new names; check how it is built).
5. `compare(result, baseline_result)` is MSE-only; leave it. Add
   `compare_b(result, baseline_result)`: meeting bootstrap of the difference in
   `b_calib_slope`, same `meeting_bootstrap` helper. Not required by the runner; nice to have.
6. Tests (synthetic panel, `make_synthetic_panel`, which has `b_true`):
   - Perfect oracle predictions (`b_hat = b_true`, `a_hat = a_true`) give slope ≈ 1 and
     `b_var_share > 0`.
   - `b_hat` = permuted `b_true` gives |t| small and slope near 0 (use a fixed seed, loose bound).
   - Constant `b_hat` returns NaN slopes without raising.
   - The decomposition gains sum to `mse_zero - mse_full` within tolerance.
7. Also run the metric on the three saved runs once by hand (reading `predictions.csv`; the
   `shock` column is missing there, so map it from `load_real_panel()`), and check it
   reproduces the numbers above (scratch slope ≈ +2.3 with the 3-seed ensemble, or report
   per-seed values). Put the per-run numbers in the status doc. Reading saved runs is not training.

## Task 2: encoder runner reports the b metrics and runs three pretraining objectives

Files: `encoder/fedenc/workflow.py`, `encoder/tests/test_return_pretrain.py`, `vast_train.sh`.

1. Comparison table: it already carries `result.metrics`, so the new metrics appear once
   Task 1 lands. Add `b_calib_slope` and `b_calib_t` to the printed summary explicitly and
   keep the existing paired MSE columns.
2. Generalize the comparison sets. Today: `--checkpoint` (next-return set) and
   `--masked-checkpoint`. Replace the hard-coded masked handling with
   `--compare NAME=RUN [RUN ...]` (repeatable), e.g.
   `--compare masked=M0 M1 M2 M3 --compare scaled=S0 S1 S2 S3`. For each named set build
   arms `tx_<name>_{full,staged,frozen,top}`. Keep the checks: same architecture, same
   history scaler per fold, distinct cutoffs per set, fingerprints unchanged. Keep
   `--masked-checkpoint` working as an alias for `--compare masked=...` so old commands and
   tests still pass, or update the tests. The `checkpoint-by-fold` table gets one column per set.
3. `_checkpoint_set`: the main set must be `next_return_gaussian` (the candidate). A compare
   set may be any objective.
4. `vast_train.sh`: pretrain-folds for `next_return_gaussian` (main), `masked_reconstruction`
   and `next_return_scaled` (compares), all `--through 2009-03-18`, three processes in
   parallel; then
   `evaluate --checkpoint $GAUSS --compare masked=$MASKED --compare scaled=$SCALED
   --arms tx_scratch tx_pre_full tx_pre_staged tx_pre_frozen tx_masked_staged tx_scaled_staged
   --baselines pooled ridge_interact --seeds 0 1 2 --device cuda`.
   The script is untracked (it hardcodes a host); edit it in place, do not commit it.
5. Tests: the existing fake-walk-forward runner test, extended to two compare sets; the
   printed summary contains the b columns.

## Task 3 (optional, do only if 1 and 2 are done and tested): a shrunk b arm

Rationale: real cross-firm b dispersion is small, so the estimator should shrink. Design,
encoder-local (`encoder/fedenc/arms.py`, `encoder/fedenc/train.py`):

- `adapt="staged"` plus `b_shrink=λ`: an extra penalty `λ · mean((b_hat - mean_b_in_batch)²)`
  added to the meeting loss, pulling each firm's b toward the meeting's pooled b. `a` is
  untouched. Arm name `tx_pre_shrunk`, λ fixed (e.g. 1.0) and recorded in params; no λ sweep.
- Test: with a huge λ the within-meeting std of b_hat collapses toward 0 on a synthetic smoke fit.

Skip if it would take more than a couple of hours; the metric and the rerun matter more.

## Task 4: docs

- `docs/encoder/pretraining-status.md`: add a section "Graded on b (2026-10-14)" with the
  facts above, the reversal on the objective (Gaussian is the candidate again), the rejected
  full-day-yield shock, and what the next run contains. Keep the "rejected" list current.
- `docs/encoder/next-return-pretraining.md`: document the metric, `--compare`, and the
  three-objective command.
- `docs/methods.md` or `docs/question.md`: if the b-calibration slope is added as a reported
  (not primary) metric, state it in one sentence where the Q3 metrics are listed. Mean
  per-meeting MSE stays primary; do not change the selection rule.

## Rules that stay in force

- No training. Unit-test smoke fits on tiny synthetic panels are fine; nothing saves a run.
- Never touch the 2026 holdout or any meeting after `--through` for selection.
- Walk-forward by whole meetings; checkpoints must precede each fold's inner validation.
- Don't commit WRDS data, `vast_*.sh`, or anything under `*/results/`.
- Windows: run pytest with `--basetemp=C:/tmp/pt`. Verify order: `pytest core`, `pytest encoder`.
- Commit when green, one commit per task, messages in the style of `b634f2d`.

## Done when

- `pytest core encoder` green; new tests cover the metric, the decomposition, `--compare`.
- Running the new metric on the three saved runs reproduces the numbers at the top.
- `vast_train.sh` is the three-objective pipeline and the status doc says what it will produce.
