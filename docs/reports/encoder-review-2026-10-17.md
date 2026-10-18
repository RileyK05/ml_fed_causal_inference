# Q3 review after the v2 final check

Reviewed 2026-10-17 at commit `9fad797`, including the untracked
`docs/encoder/q3-v3-roadmap_g.md`. This is a review, not approval to train.
No model training, saved experiment, or changes to implementation were made.

## Current evidence

The v2 regime, macro-history, two-shock, shock-weighting, and b-only experiments have now
run. They should not be presented as untested ideas. The two-shock baselines and the
out-of-sample checks of the gate passes also exist.

I recomputed the latest scores directly from saved final prediction tables. Predictions
were averaged across seeds within each arm, then equally across the six predeclared
CAPM-base network arms. Scores give each of the 32 meetings equal weight.

| Predictor | 2022–2025 MSE |
|---|---:|
| Six-arm network consensus | 5.722499 |
| Original CAPM | 5.725065 |
| CAPM with last and smoothed beta | 5.707116 |

An independent-meeting percentile bootstrap with 10,000 resamples gave these 95% intervals:

- Consensus minus original CAPM: -0.002566, interval [-0.023833, +0.017536].
- Consensus minus two-beta CAPM: +0.015383, interval [-0.032266, +0.066098].

These are descriptive re-scores of an already used evaluation period, not new confirmation.
They do not account for serial dependence or model-selection uncertainty. The report's
predeclared directional tests remain separate; this review does not replace them.

The evidence supports using the two-beta CAPM as a strong reference for subsequent
development. It does not prove that it outperforms the original CAPM in the population,
that all firm sensitivity is beta, or that no useful nonlinear signal exists.

Sources: [v2 results](../encoder/v2-results.md), the `20261017-23*_v2-final-*` saved runs,
and [CAPMBetas](../../encoder/fedenc/v2.py).

## Implementation improvements before another model comparison

### 1. Isolate the inner validation block from the fitted base

`V2TwoPartArm.fit` fits its CAPM/pooled base on all supplied meetings and constructs residuals
before `fit_v2` separates the final 15% of those meetings for early stopping. Thus inner
validation outcomes affect the fitted base and the residuals used for checkpoint selection.
This is inner-validation contamination, not evidence that outer-test labels entered fitting.

Split first. During checkpoint selection, fit the base and preprocessing on the inner
training block only. If refitting on the complete outer-training panel afterward, specify
the procedure in advance and refit both stages consistently at the selected budget.

Regression check: changing only inner-validation labels must leave the base coefficients
and inner-training residuals unchanged.

Code: `encoder/fedenc/v2.py`, `V2TwoPartArm.fit` and `fit_v2`.

### 2. Choose checkpoints using the prediction actually deployed

Training and early stopping penalize a neural meeting-average error with `lambda_mean=0.05`.
Prediction subsequently removes each member's meeting-average `a` and each shock loading,
then adds the fixed base. Consequently, checkpoint selection partly rewards a component
that is discarded at prediction time.

A forward-only synthetic check confirmed this: adding 2 to the network's common b bias
changed its validation objective from 2.663590 to 2.632367, while its deployed centered
prediction changed by less than 9e-8. No optimizer steps were taken.

Apply the same centering in training, validation and prediction, and validate the complete
base-plus-correction prediction. If the uncentered mean objective is deliberately retained
as an auxiliary regularizer, separate it from the deployment score used for early stopping.

This architecture also fixes the predicted meeting mean to the base's mean. Network-only
changes cannot fix aggregate-market prediction errors. Improvements to that component need
a separate base-model experiment.

### 3. Save each shock contribution separately

For `target_slope`, the second shock's response is added to `a_hat` to maintain the legacy
identity `mu = a_hat + b_hat*s`. The identity is correct, but `a_hat` is no longer entirely
a pre-event baseline: it includes the realized announcement-window slope response.

The claim that almost all neural adjustment is "drift, not sensitivity" must therefore be
qualified for any consensus containing this arm. Save `base_mu`, `delta_a`, each `b_k`, each
`s_k`, and each `b_k*s_k`. Keep compatibility columns if needed, but perform scientific
decompositions on the separate components.

The policy/information split likewise conditions on realized announcement-window equity
movement. Its results answer a broader conditioning question than the original spec's
pre-event state plus rate surprise. Label that explicitly and match comparator information.

### 4. Preserve missing macro information and improve the experiment record

`load_macro_tables` still replaces missing yields, factor returns and regime calculations
with zero. This makes unavailable data appear to be measured zero changes or neutral
states, including in trailing-exposure calculations. Preserve NaNs, calculate exposures
from jointly observed pairs with a minimum count, and add availability indicators or a
declared exclusion policy. This remains distinct from masked zero storage inside a model.

The ablation and final runners save arm names but omit their full resolved `V2Config`.
Only the first ensemble member's fit result is retained. Save resolved configuration,
code revision, all member seeds and selected epochs, and the exact consensus run mapping.
The decomposition and final-claim calculations currently described as scratchpad scripts
should become reproducible read-only commands. These changes matter in a rapidly changing
repo even if they do not improve MSE.

## Corrections needed in the draft v3 roadmap

The draft contains useful hypotheses, but several explanations are presented more strongly
than the saved evidence supports.

1. **The G2 horizon example is misread.** The saved gates contain only `day 0` and `days 0-2`.
   For the within-sector rate-beta × shock × drawdown test, p=0.000876 is the 1994–2008
   result and p=0.000366 is the 2009–2021 result, both for days 0–2. They are not successive
   day-1 and day-2 results. The reported out-of-sample follow-up did not improve prediction.
2. **Gate significance does not establish predictable non-beta signal.** Hundreds of tests
   were examined; the sector and drawdown follow-ups failed out of sample. Treat the patterns
   as motivation, not proof that the objective is concealing an established forecasting edge.
3. **Several proposed fixes revisit negative evidence.** Sector inputs were tested in v1;
   sector two-shock models were tested in v2. Ordinary and macro-release-day rate loading
   transfer was tested in R4/R7/R10. Longer-window diagnostics and the G2 follow-up exist.
   Revisit only with a specific changed assumption and a matched control.
4. **The advertised data tensor is inaccurate.** Current macro augmentation adds `dy1`, HML
   and SMB; it does not already contain daily `dy2` and `dy10` channels. Verify availability
   before claiming the proposed trajectory task needs no data work.
5. **Hard projection is a restriction, not a signal generator.** It removes candidate
   predictions in the span of measured betas, potentially including useful signal correlated
   with those exposures. The written ridge projection is not exactly orthogonal when
   epsilon is positive; use rank-aware QR/SVD for an exact test or call it a soft penalty.
6. **The proposed target purification changes the training target.** Its equivalence to raw
   conditional-mean prediction needs assumptions about the conditional mean of the removed
   market residual. No measured 80% noise reduction is established here.
7. **Cosine loss is not rank loss or outlier-invariant.** A single large target can change its
   direction substantially. Huber, clipping and cosine losses also do not automatically
   preserve the raw conditional-mean estimand. Treat them as distinct objectives.

Claims about the fraction of trading dominated by ETF arbitrage, exact causes of failure,
and runtime promises are not established by the saved runs. They should be sourced,
measured or explicitly labeled hypotheses.

Source for the G2 correction:
[saved gate table](../../encoder/results/q3_susceptibility/20261017-120033_v2-gates/tables/tests.csv).

## A smaller, informative next experiment set

These are proposed future fits and require explicit training authorization.

### First: stronger base, unchanged correction

After correcting validation and deployment alignment, compare the two-beta CAPM alone
against that same base plus one small residual model. Compare a summary-feature MLP with
the transformer on identical inputs and folds. Start without projection or a new loss so
the effect of the stronger base can be measured. A later projected arm can specifically
test the narrower hypothesis of a correction outside the measured-beta span.

`CAPMBetas(252)` is an average of trailing 63-day beta estimates across a 252-day history,
not a fresh beta regression estimated over 252 raw returns. Preserve that distinction.

### Second: recency before a regime-similarity kernel

The era changes motivate testing whether historical weighting matters. Compare expanding
history against one predeclared recency decay, with the same strongly regularized model,
same two-beta reference and same inputs. Select any decay parameter in earlier inner
validation, and weight the reference consistently.

This is a cleaner first test than simultaneously adding a covariance-distance kernel,
many new ratios, signed shocks and a new loss. Report concentration and the descriptive
weighted meeting count `(sum(w)**2)/sum(w**2)`; it is not a serial-dependence-adjusted sample
size. A localized model has less information and cannot be assumed to improve.

### Third, if needed: relative firm state before set attention

Add within-meeting ranks and a few cross-sectional summaries alongside the existing absolute
features, using only pre-event data. Test these with a small model before adding attention
across firms. This asks whether missing relative context matters without introducing a large
new architecture. Keep absolute inputs: ranks alone discard economically meaningful levels.

## Scope and verification

Reviewed commits, current model/scoring code, tests, reports and saved runs; recomputed the
final consensus MSE and paired intervals from stored predictions; ran one synthetic
forward-only objective check. No test suite or fitted experiment was launched. No saved run,
raw data, model implementation or existing roadmap was changed. Both v1 and v2 have now used
2022–2025; future uses of those years remain reused evaluation, not a fresh holdout.
