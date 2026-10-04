# Methods - Active Model Plan

**Companion to:** [question.md](question.md), Q-spec v1.1 (2026-09-25).
**Status:** Candidate models for discussion; only the project structure is settled.
**Historical reference:** [Original broad catalog](archive/methods-v1.md), superseded.
**Q3 design note:** [Transformer architecture, pretraining and fine-tuning options](encoder/transformer.md).

The active chain is Q1 -> Q3 -> Q4 -> Q5. Q3 is the main architecture research task.
Q2 is parked. More elaborate models must earn their place on earlier validation data.

## Q1 - Keep the causal estimator focused

**Primary candidate: partially linear DML (PLR).** Continuous surprise s, aggregate return
Y and pre-event controls X. Fit l(X) = E[Y|X] and m(X) = E[s|X] outside the scored block,
then estimate theta from residualized Y and s. The partially linear restriction is a
substantive assumption; a slope under misspecification needs a qualified interpretation.

Start nuisance estimation with ridge/elastic net; compare shallow boosting or random
forests on historical validation data. Use an OLS event-study coefficient as the baseline.
There is no requirement for complicated nuisances to outperform simple ones at this sample
size. Inspect residual treatment variation and sensitivity to influential meetings.

Use explicit temporal sample splits and appropriate dependent-data inference. Forward-only
nuisance fits discard an initial training period from scoring; document the resulting
estimation population. Standard random-fold DML standard errors do not automatically justify
an arbitrary temporal procedure. Audit library support and the moment/inference implementation.

Use surprise as the treatment. A PLIV model instrumenting a separate policy action answers
a different estimand and requires exclusion/relevance assumptions. Binary-treatment AIPW/TMLE,
staggered-adoption DiD and meeting fixed effects are not drop-in alternatives for this target.
Shock decomposition is a small predeclared robustness exercise, not an architecture search.

Reference: [DoubleML model definitions](https://docs.doubleml.org/stable/guide/models.html).

## Q3 - Architecture comparison

All candidates predict the same firm event-day return using pre-event state plus the
realized shock. Score conditional-mean predictions with per-meeting MSE; also evaluate
cross-sectional RankIC and relative-return performance so common market prediction does
not conceal weak firm differentiation. Q1 can supply an optional historically fitted offset;
Q3 need not mechanically reuse Q1's nuisance learners.

| Candidate | How it represents susceptibility | Role |
|---|---|---|
| Ridge interactions | Shock multiplied by observed characteristics, with regularization | Essential interpretable baseline |
| Hierarchical random slopes | Firm and sector sensitivities partially pooled | Strong small-event-sample comparison |
| Gradient boosting | Nonlinear interactions among state, regime and shock | Main tabular challenger |
| Small MLP | Shared nonlinear state encoder and response head | Cheap neural architecture control |
| TCN | Convolutions over pre-event return/volume histories | First sequence baseline |
| Masked-patch transformer | Pretrained temporal representation, adapted to event response | Main ambitious candidate |
| DeepSets / cross-sectional attention | Pre-event states of other firms | Later incremental context test |
| Economic graph network | Point-in-time supply-chain, ownership or other edges | Deferred until reliable edges and baseline gains exist |

### Proposed sequence architecture

    Pre-event return/volume history -> temporal encoder ----+
                                                          |
    Point-in-time fundamentals + sector + regime ----------+-> firm state h
                                                                  |
    Realized shock -> conditioning parameters --------------------+-> response head

Two conditioning choices deserve a direct comparison:

1. **Structured sensitivity head:** mu(z,s) = a(z) + b(z)'s. A nonlinear encoder
   learns state-dependent loadings while the response stays linear in the shock.
   This is interpretable and limits how much must be learned from sparse events.
2. **FiLM-conditioned head:** h_s = gamma(s) * h + beta(s), followed by a small
   response head. This allows a more flexible response to shock size/direction.
   Compare against simply concatenating s and h to show whether FiLM itself helps.

Pretrain on masked historical ordinary-day sequences, then freeze the encoder and train
an event head. Compare light fine-tuning only after this baseline. Never pretrain on dates
later than the held-out event, even without labels. Use training-only normalization.

PatchTST provides a precedent for patching and masked temporal pretraining; FiLM provides
feature-wise affine conditioning. Their combination for firm responses is our proposed
adaptation, not an established result from those papers.

References: [PatchTST](https://arxiv.org/abs/2211.14730),
[FiLM](https://arxiv.org/abs/1709.07871).

### Ablations and promotion

- State: tabular only, history only, combined.
- Shock: state-only predictor versus shock-conditioned predictor; then shock definitions.
- Encoder: simple lag summaries, TCN, masked-patch transformer.
- Adaptation: frozen encoder versus limited fine-tuning.
- Head: state-dependent linear shock loading, concatenation, FiLM.
- Context: add cross-sectional or graph information only after the above comparisons.

Use equal meeting weights, shared outer folds and a bounded tuning budget. Fix a practical
gain threshold before final evaluation and report paired per-meeting loss differences.
No architecture is selected using Q4 reversal results. Q3 remains worthwhile if Q4 is null.
Auxiliary events, regime models and distributional heads are optional later experiments.

### Uncertainty

A mean head is enough to define the primary gap. Quantile heads can later estimate
conditional response tails; enforce monotonicity and state interpolation/tail assumptions
before computing PIT scores. Quantiles are not conditional means. Ensembles measure model
disagreement, not guaranteed coverage. Ordinary-day conformal calibration does not guarantee
FOMC coverage, and prediction intervals are not confidence intervals for the conditional mean.

## Q4 - Simple tests of a difficult hypothesis

**Signal:** gap = actual response - saved out-of-sample Q3 mean prediction.

Start with per-meeting regressions of subsequent abnormal return on the raw signed gap,
then aggregate slopes with inference allowing serial dependence. Report signed RankIC,
portfolio sorts and cumulative-return paths as secondary views. Negative relation means
reversal; positive means continuation. Preselect the primary horizon (proposed t+1 close
to t+20 close) and count every extra horizon/score specification.

Compare against ordinary-day benchmark residuals, raw-return reversal, sector and volatility
exposures. Match placebo days using pre-event information. Compare the ordinary-day signal
on both event and placebo days; do not assume a zero-shock ordinary day is in the support
of an event-trained Q3 model. Use simulations with no true reversal to assess mechanical
false positives. Retain raw magnitudes alongside ranks; ranks always generate extreme buckets.

Freeze Q3 selection before Q4 evaluation. If resampling the full fitting pipeline, preserve
chronological information boundaries. Separately label inference conditional on a fixed fitted
model versus inference that includes model-training variability. Permutation tests require a
valid exchangeability argument; arbitrary shock/date shuffling across regimes is insufficient.

## Q5 - A simple portfolio before policy learning

Only after Q4 validation: fixed gap-rank or threshold portfolio, exposure constraints,
next-day-close execution, fixed holding period, and costs including borrowing. Compare
against no-skill and simple raw-return signals on untouched chronological data. Select
reversal versus continuation direction using earlier validation only.

Learned sizing and regime gates are later challengers. Contextual bandits, reinforcement
learning and logged-policy OPE are deferred: historical prices alone do not supply the
behavior-policy setup those estimators require. Selection-aware backtest reporting is still
needed regardless of the portfolio rule.

## Q2 - Parked

No channel model is in the active plan. Sector/characteristic heterogeneity belongs in Q3.
Formal mediation, channel IV and structural VAR work would need a separate design review.

## Suggested first comparison

Q1: OLS plus PLR-DML with simple nuisance learners.
Q3: ridge/hierarchical interactions versus boosting, then a small temporal encoder with a
state-dependent linear shock head. Compare a pretrained transformer plus FiLM once the
baselines and evaluation are established. Q4/Q5 remain downstream tests, not architecture
selection criteria.
