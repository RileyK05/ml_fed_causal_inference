# Q3 pretraining: cheap test menu

Written: 2026-10-13. Status: candidate checks and experiments, not an adopted primary specification.
This is the single combined list: the original menu plus the real-data linear diagnostics in section 3a.

This records the tests proposed while discussing [the pretraining debate](pretraining-debate.md).
The aim is to resolve one question at a time before committing to expensive experiments.
The deliverable remains the full expected firm return on a Fed day, conditional on pre-event
state and the realized surprise:

```text
mu(z, s) = a(z) + b(z) * s
```

The final target is the absolute meeting-day return, with mean per-meeting MSE as the primary
selection metric. Ordinary-day tasks are candidates for learning transferable state; an
ordinary-day rate loading is not assumed to equal a Fed-surprise loading.

**No training without the user's explicit go-ahead in chat.** Writing this menu does not
authorize any fits, pretraining, benchmarks, ablations, or saved runs. The repo's existing
allowance for tiny smoke fits inside unit tests remains separate from experiment approval.

## 1. Data checks: no training

These are descriptive calculations and timing/coverage audits. They should be relatively
cheap once the relevant arrays are loaded; loading the full daily panel may dominate cost.

| ID | Check | Question and readout |
|---|---|---|
| A1 | Shock distribution by era and direction | Are quiet, hawkish, and dovish meetings represented in each training period? Report counts, signed and absolute quantiles, and ranges for Fed surprises and ordinary daily yield changes. |
| A2 | Shock concentration by training fold | How much slope information comes from the largest 1, 5, and 10 meetings? Report shares of centered shock sum of squares. This measures leverage concentration, not fitted influence or an effective inference sample size. |
| A3 | Market-rate correlations on ordinary days | Does a rate move mostly accompany a market move, and does the relationship change by era? Inspect correlations and the factor design's conditioning to assess the case for multifactor pretraining. |
| A4 | Common versus within-meeting return variation | How much observed return variation is the meeting-wide move versus differences between firms? Decompose realized returns around the meeting mean. This does not identify signal versus noise. |
| A5 | Feature coverage and distribution shifts | Does the ordinary-day corpus resemble the Fed-day population in volatility, beta, fundamentals, and missingness? Compare development-period distributions and coverage; document universe selection. |
| A6 | Timing and label audit | Verify previous-close history and context, same-day response labels, point-in-time fundamentals, shock/return units, and label end dates. Future-beta and future-volatility labels must finish before their applicable cutoff. |
| A7 | Window and target overlap audit | Count duplicate windows and shared reconstruction/future-label target dates across pretraining train/validation blocks. Use explicit calendar dates rather than inferred ordering. Record historical context overlap separately from target overlap. |

### Descriptive checks already performed during the discussion

The cached real panel contains 253 scheduled meetings from 1994-05-17 through 2025-12-10.
The ordinary-day comparison below uses that date span and excludes all FOMC statement
dates, including unscheduled statements. Yield differences preserve missing values.

| Absolute move | Fed surprise | Ordinary daily 1-year yield change |
|---|---:|---:|
| Median | 1.42 bp | 1.63 bp |
| 90th percentile | 4.78 bp | 6.11 bp |
| Observations | 253 meetings | 7,620 days |

The five largest contributions account for 31.5% of centered shock variation across all
253 meetings and 40.5% in the first 120 meetings. These are descriptive leverage checks;
no response model was fitted. They do not establish the influence of those meetings on a
model or justify excluding them. The full-span figures are context, not model-selection evidence.

## 2. Architecture and gradient checks: no optimizer steps

Use fixed batches and forward/backward passes. Inspect initialization first, then repeat
at checkpoints only if training is subsequently authorized. Initialization alone cannot
establish the gradient balance of a trained model.

| ID | Check | Question and readout |
|---|---|---|
| B1 | Separate encoder gradients through a and b | Measure each path's contribution to the shared encoder gradient, using the same residual and batch. Report norms by shock-size bin. Test the claimed imbalance directly. |
| B2 | Gradient agreement | Measure cosine similarity between the two path contributions. Large a gradients may also help b if the paths push the representation in similar directions. |
| B3 | Shock-unit rescaling | Express shocks in bp versus 10 bp and adjust b to preserve predictions. Compare gradients to see how much apparent imbalance depends on parameterization. Equivalent predictions do not imply equivalent optimizer dynamics or penalties. |
| B4 | Dial and variance-weighting audit | On the same batch, inspect the effects of a mild backward-only dial and NLL weighting. Verify that a backward-only dial preserves predictions and that a's weights still receive gradients. Variance weighting scales both mean-head paths; it does not inherently favor b. |

Masking, padding, input timing, meeting weighting, and freezing already have unit-test
coverage. Extend that coverage where a change requires it rather than duplicating existing
tests. Relevant files: `encoder/tests/test_encoder.py`, `core/tests/test_q3_foundation.py`,
and `core/tests/test_q3_real.py`.

## 3. Small fitted diagnostics: training approval required

Start with summary features and linear models. These should cost much less compute than
transformer pretraining, although data preparation still matters.

| ID | Experiment | Question and readout |
|---|---|---|
| C1 | a(z) versus a(z) + b0*s versus a(z) + b(z)*s | Does knowing the shock improve prediction? Does state-dependent sensitivity add value beyond a common slope? Compare later validation per-meeting MSE with comparable state features and regularization. Highest-priority fitted comparison. |
| C2 | Ordinary-day rate model versus market-plus-rate model | Does the apparent rate exposure survive accounting for market moves? Compare chronological validation predictions and loading stability. These remain ordinary-day associations. |
| C3 | Future-beta label reliability | Estimate loadings over 1-, 3-, and 6-month windows. Examine uncertainty, factor-design conditioning, and stability to judge whether future betas are sufficiently reliable pretraining labels. Account for overlapping observations. |
| C4 | Leave-one-large-meeting-out fits | Exclude each selected influential training meeting individually, refit a simple response model, and score the same later validation meetings. Measure changes in slopes and predictions. Select the audit meetings using training data only. |
| C5 | Global shock centering | Compare s with s - mean_train(s), converting reported a and b back to the original units. Does reduced intercept-slope coupling help optimization? Preserve comparable regularization and the same prediction function class. |

## 3a. Real-data linear diagnostics: training approval required

These use the real panel and real residuals, with OLS or ridge only. Each fit should take seconds to
minutes on CPU. They test the premises behind the pretraining candidates directly, before any of
those candidates is built. Use whole-meeting splits and the same development-period boundary as
section 10. Fitted results are exploratory runs.

**Scope rule:** every fitted test here and in section 3 runs **inside the development period** (the
first 120 meetings, through 2009-03-18), with chronological validation inside it. "Walk-forward" below
means inner folds, not the outer 2009–2025 test folds. Comparing models on outer-fold performance now
would spend the test period on model selection. The one exception is R1: it may use the outer design,
because its readout is whether a *planted* effect is detectable, not which model wins. Say so
explicitly when reporting it. Expect less power inside the development period (about 120 meetings); R1
quantifies how much.

| ID | Test | Question and readout |
|---|---|---|
| R1 | Power on real noise | Plant a known, realistic b(z) heterogeneity on top of the real Fed-day returns (keeping the real residuals, including meeting-common noise), then run the planned walk-forward split and paired meeting bootstrap with a linear interaction model. Can the planned evaluation detect it? Repeat at a few planted effect sizes to find the detectable minimum. **If a realistic effect is undetectable, fix the evaluation design before any training.** Highest priority, together with C1. |
| R2 | Firm-level b dispersion vs noise | Fit return ~ s per firm (firms with enough meetings, e.g. 100+). Compare the cross-firm spread of b to the spread that sampling noise alone would produce (for example, a test of equal slopes, or permuting s across meetings). Is there heterogeneity in sensitivity to find at all? Firm-level, static, and coarse; complements C1. Caveat: firms with 100+ meetings are long-lived index members, a survivor sample; report how many firms qualify. |
| R3 | Which features carry b | Meeting fixed effects plus s × each pre-event feature (rvol21, beta63, leverage, log size, book-to-market, sector), one at a time and jointly, walk-forward. Which state variables predict sensitivity? Sets what the encoder must at least recover. Note: meeting fixed effects absorb the common slope (s is constant within a meeting), so only the interactions are identified, and test meetings have no fitted effect. Score R3 with centered (within-meeting) MSE; the absolute-return version of this question is C1. |
| R4 | Ordinary-day rate beta → Fed-day response | For each firm-meeting, compute the 63-day ordinary-day rate loading (market-plus-rate regression, history ending the day before the meeting, FOMC days excluded). Add s × rate_beta to the R3 model. Does ordinary-day rate sensitivity predict Fed-day sensitivity? Direct real-data test of the P3 premise (S4 is its synthetic counterpart). |
| R5 | Future-loading persistence | Do loadings estimated before date t predict loadings estimated over the next 1/3/6 months? Report persistence by factor (market, rate, SMB, HML) and by era. If future loadings are mostly noise relative to past ones, P5 has little to learn. Complements C3. |
| R6 | Earnings-day vs Fed-day reaction size | Using Compustat report dates (rdq), compute each firm's typical absolute abnormal return on earnings days before t, and test whether it predicts the absolute Fed-day residual. Is "how violently a firm reacts to news" one transferable trait? Premise test for an earnings-reaction auxiliary task. |
| R7 | Which ordinary days transfer | Repeat R4 with loadings estimated on (a) all ordinary days, (b) macro-release days only (CPI, payrolls), (c) the remaining days. Which day set's rate loading best predicts the Fed-day response? Settles the day-selection question for P3 with evidence. Release dates need sourcing: the repo has CPI dates from 2023 only and no payroll dates. Run (a) first; (b) and (c) wait for the dates. |
| R8 | Stability of sensitivity over time | Compare R2/R3 estimates in the first vs second half of the development meetings (1994–2001 vs 2001–2009), and across positive vs negative stock–bond correlation regimes (sign of the trailing 63-day correlation of market and 1-year yield change). Kept inside the development period so 2009–2025 stays untouched. If b is stable, a static beta may suffice; if it moves with state, a state-dependent model is justified. |
| R9 | Placebo date | Re-run R3 with the target replaced by the return on the trading day **before** each meeting (history shifted back accordingly). The s-interaction terms should be about zero. A non-zero result signals look-ahead or a timing error in the panel build. The known pre-FOMC drift can make the baseline non-zero on that day; only the s terms are the test. |
| R10 | Placebo shock in linear fits | Re-run R3/R4 with s shuffled across whole meetings (within era). Shock-dependent gains should vanish. A linear-model counterpart to E7, available before any neural predictions exist. |
| R11 | Residual dispersion vs shock and state | Two specs: squared absolute residuals (no meeting fixed effects) on \|s\|, rvol21 and \|s\| × rvol21; and squared within-meeting residuals from the R3 model on rvol21 and \|s\| × rvol21 with meeting fixed effects (which absorb \|s\| itself). Does dispersion rise with surprise size or with firm state? A cheap premise check for the volatility outputs in section 7 before fitting V1. |

## 4. Synthetic stress tests: experiment approval required

Use linear models or tiny networks first. Benchmarks and recovery sweeps require approval
even when the data is synthetic. Tiny unit-test smoke fits remain governed by the existing
repo allowance and must not become disguised experiment sweeps.

| ID | Synthetic case | Question and readout |
|---|---|---|
| S1 | State-dependent a, constant b | Does the method invent heterogeneous sensitivity when only the baseline varies? Assess false heterogeneity as well as mean prediction. |
| S2 | State-dependent b, strong baseline, mostly quiet shocks | Can it recover sensitivity without sacrificing baseline prediction? Compare mu, a, and b errors against known truth. Direct test of the debate's central concern. |
| S3 | Shared meeting noise | Add a common residual to every firm in a meeting. Does performance collapse when more firms cannot average away all residual noise? Preserve whole-meeting evaluation. |
| S4 | Helpful versus misleading ordinary-day exposure | Make ordinary and Fed sensitivities related, unrelated, and opposed. Measure when pretraining helps and when it causes negative transfer. |
| S5 | Shock-dependent noise and saturating response | Separate heteroskedasticity from misspecification of the linear mean. Assess uncertainty calibration and overswing against known truth, using separate cases before combining them. |

The current simulator has independent row noise and one particular planted response
relationship. These cases broaden the checks without treating synthetic success as evidence
that the real-data relationship has the same form.

## 5. Pretraining-target pilots: training approval required

Use one reduced, shared corpus, the same encoder, comparable label eligibility, and a
comparable compute budget. Preserve date coverage when reducing firms. For response tasks,
align yesterday's state plus today's realized factors with today's return; tomorrow's
return is a different target.

| ID | Candidate | Comparison it resolves |
|---|---|---|
| P1 | No pretraining | Essential scratch reference. Include a frozen random encoder as an additional representation control. |
| P2 | Masked reconstruction | Does reconstructing history provide transferable state? Existing implementation; retain as the comparator. |
| P3 | Ordinary-day a + b*DeltaYield | Does practicing the response-shaped task improve transfer despite the mixed meaning of ordinary rate moves? |
| P4 | Ordinary-day multifactor response | Does explaining market, rate, and style-factor responses jointly teach more useful state than rates alone? Retain an intercept and absolute-return labels. |
| P5 | Future betas plus future volatility | Does learning estimated exposures transfer better than learning conditional returns directly? Proceed only if C3 supports label reliability. |
| P6 | Response task plus auxiliary beta/volatility target | Does combining tasks add useful information? Try only if the individual objectives show complementary strengths; prechoose the auxiliary weight. |

The primary readout is later Fed-response prediction. Better reconstruction or beta-label
prediction alone does not establish transfer. First freeze each representation and fit
the same small response head. Then permit adaptation for promising candidates: a poor
frozen result does not prove the representation cannot be adapted.

The real history-window corpus exists, but streaming pretraining and real-checkpoint
loading in the transformer arms still need implementation. Ordinary-day response tasks
also need dated factor/outcome labels aligned to the windows. History-only pilots can
reuse the history corpus; adding daily fundamentals/context requires additional
point-in-time alignment and should be recorded as extra build scope.

## 6. Narrow architecture and loss comparisons: training approval required

Change one feature at a time around a promising objective. Do not multiply all these
choices into a full grid.

| ID | Comparison | Question and readout |
|---|---|---|
| M1 | Frozen versus top-block adaptation | Does the representation transfer directly or need modest Fed-specific adjustment? Full adaptation follows if evidence warrants it. |
| M2 | Short versus full history | Compare, for example, 63 and 252 days. Does older history contribute enough to justify its cost? |
| M3 | One versus three transformer layers | Does attention depth help? Consider a larger model only after an objective shows useful transfer; change depth and width separately. |
| M4 | Mean versus last-token pooling | Does averaging the window dilute recent state? Already configurable. |
| M5 | Transformer plus explicit summary features | Does direct access to volatility, beta, and momentum let the transformer focus on additional sequence information? |
| M6 | Normal joint learning versus a mild gradient dial | Revisit the dial only if B1/B2 and S2 expose a learning problem. Keep a learning in both cases, and distinguish a backward-only dial from changing the forward mean. |
| M7 | MSE versus one Huber alternative | Does residual robustness improve held-out conditional-mean prediction? Check extreme-shock leverage separately; Huber does not remove the shock multiplier from the slope gradient. |

Select capacity and adaptation using later Fed-response validation, not auxiliary loss alone.

## 7. Volatility-output tests: training approval required

Separate the value of predicting dispersion from its effect on mean-model training.

| ID | Experiment | Question and readout |
|---|---|---|
| V1 | Fixed mean; constant variance versus state variance versus state-and-surprise-size variance | Can dispersion be predicted? Compare held-out likelihood and interval coverage, including quiet and large-shock meetings. Use positive variance parameterization and document any imposed monotonicity. |
| V2 | MSE versus Gaussian NLL versus one prechosen beta-NLL setting | Does joint uncertainty training improve mean prediction? Track mean per-meeting MSE separately from uncertainty quality; do not rank models by beta-NLL's training-loss value. |

Beta-NLL mitigates an optimization problem caused by learned variance downweighting
hard-to-fit observations. It is not a guaranteed fix, and at beta = 1 the mean-output
gradients are MSE-like. See [Seitzer et al.](https://arxiv.org/html/2203.09168).
An estimated surprise-dependent dispersion term is a predictive association; it does
not by itself establish a causal volatility response.

## 8. Diagnostics after predictions exist

Most reuse saved models and predictions. Feature-removal comparisons that refit models
and shuffled-shock refits are additional fitted experiments, not evaluation-only checks.

| ID | Diagnostic | Question and readout |
|---|---|---|
| E1 | Error by shock size, direction, and era | Does an average gain hide failures on quiet, dovish, or particular-regime meetings? Use predeclared slices. |
| E2 | Sensitivity calibration | Does the predicted shock contribution correspond to realized responses across meetings? Examine realized returns net of predicted a, while acknowledging contamination from baseline errors. This is not true-b recovery on real data. |
| E3 | Seed stability of b | Do seeds yield similar mu predictions but very different sensitivities? Report both prediction stability and b stability. |
| E4 | Feature removal | Remove history, fundamentals, or context one branch at a time to locate the gain. Distinguish perturbing a fitted model from refitting without a branch; the latter needs training approval. |
| E5 | History-order perturbation | Does scrambling patches hurt? Treat this as a stress test because scrambled histories can be unrealistic. |
| E6 | Shock sweep at fixed state | Plot predictions across the observed training shock range and beyond it to expose extrapolation. For the current head, linearity is structural; the plot does not establish that real responses are linear. |
| E7 | Whole-meeting shock shuffling within eras | Negative control for shock-dependent gains. Distinguish evaluation perturbation from shuffled-data refitting. This is not automatically a valid significance test without the necessary exchangeability assumptions. |

## 9. First batch and decision sequence

1. Complete the remaining data audits A1-A7, extending the descriptive checks already done.
2. Inspect B1-B4 without optimizer steps.
3. Once training is authorized, run the go/no-go fits first: R1 (can the evaluation detect a
   realistic effect?), C1 and R2 (is there state-dependent sensitivity at all?), and R9 (placebo
   date). If R1 fails, fix the evaluation design before anything else.
4. Run the premise tests: R3-R8 and R10-R11, plus C3. R4/R7 gate P3, R5/C3 gate P5, R6 gates an
   earnings auxiliary task, R11 gates section 7, and R8 tells whether state dependence is
   worth modelling.
5. Run S1-S5 at small scale, emphasizing common meeting noise and misleading transfer.
6. Use those findings to select two or three pretraining objectives for pilots, with
   scratch as the reference. P2, P3, and P4 are the initial candidates; P5 depends on C3 and R5.
7. For promising representations, compare limited adaptation and inspect E1-E3 before
   adding capacity, a gradient dial, or a new loss.

### Proposed overnight batch (needs the user's go-ahead in chat)

Everything here is descriptive, gradient inspection without optimizer steps, or linear on CPU:
A1-A7, B1-B4, C1-C5, R1-R11, and S1-S5 using linear models (the S checks use linear models only). The
transformer pilots (P, M, V) and post-prediction diagnostics (E) wait until these results are read.

**How to run** (code: `core/fedcore/q3/cheap/`, B checks in `encoder/fedenc/cheap_gradients.py`):

```text
python -m fedenc.cheap list                                  # every check, kind, approval gate
python -m fedenc.cheap run --batch overnight --dry-run       # print the plan, load nothing
python -m fedenc.cheap run --batch no_fit                    # A + B only, a few minutes: validates the data load
python -m fedenc.cheap run --batch overnight                 # all 32 checks
python -m fedenc.cheap run --ids R1,C1,R2,R9                 # or --batch go_no_go
```

Each check is saved as an exploratory Q3 run in `encoder/results/q3_susceptibility/` (tables, metrics,
data fingerprints), plus a `cheap-batch` run with status and timing. A failing check is logged and the
batch continues. `--quick` cuts bootstrap and permutation budgets; `--synthetic` swaps in a small
simulated stand-in (smoke run). Loading the real data needs about 3 GB of free memory: close heavy apps first.

## 10. Evaluation and budget controls

- Screen models inside an early development period. If using the first 120 meetings
  before 2009-04-29, create chronological validation inside that period. Pretraining
  and future labels must finish before the inner validation boundary, not merely
  before the first outer test meeting.
- Reduce firms before sacrificing date coverage. Holding out firms alone does not
  establish prospective performance on new meetings.
- Keep a common development split, target, preprocessing procedure, and comparison
  budget. Fit scalers and any calibration using permitted training data only.
- Use one seed to detect gross failures in initial screening. Small gains need
  replicated seeds and paired meeting-level uncertainty before promotion.
- Preserve whole meetings in splitting and inference, with appropriate treatment
  of serial dependence and overlapping labels. Firm counts do not create new shocks.
- Write the candidate comparison and readout before each experiment. Log every fitted
  pilot as exploratory through ResultStore; retain append-only runs and count alternatives.
  The permitted tiny unit-test smoke fits remain unsaved tests.
- Keep outer evaluation folds outside model selection and the 2026 holdout untouched
  until the final evaluation. No downstream trading metric selects Q3 models.

Reference specifications: [question.md](../question.md), [methods.md](../methods.md),
[transformer.md](transformer.md), and [training-plan.md](../training-plan.md).
Verify implementation claims against code before execution.
