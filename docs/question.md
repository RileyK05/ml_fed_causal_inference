# Question Spec - FOMC Causal Inference Project

**Version:** Q-spec v1.1 (2026-09-25)
**Status:** Agreed project direction; detailed specifications remain draft until final evaluation.
**Model discussion:** [methods.md](methods.md); detailed Q3 architecture options in [transformer.md](transformer.md).

Q3 is the central contribution: predict how firms respond to monetary-policy news.
Q1 supplies the treatment definition and aggregate baseline. Q4 tests whether Q3's
prediction errors forecast subsequent returns; Q5 evaluates trading usefulness.
Q2 is parked. Existing identifiers are retained for code and saved-run continuity.

## Design principles

- The complexity belongs in representation learning, not in the causal claim.
- Q1's identifying assumptions do not automatically make Q3's heterogeneity causal.
- Q3 uses pre-event firm state and the realized shock. This is conditional response
  prediction, not a forecast of the announcement before it happens.
- The primary gap is actual response minus Q3's out-of-sample expected response.
  It is model-dependent. An ordinary-day benchmark remains an independent comparison.
- Select Q3 models using response-prediction performance, not downstream trading profits.
- Q3 can succeed even if Q4 finds no reversal. Q5 is conditional on Q4 evidence.
- Mispricing is a possible interpretation to investigate, never the definition of a residual.

Firms add information about cross-sectional structure; they do not create additional
independent policy shocks. Keep whole meetings together in evaluation and inference.

## Active questions

| ID | Role | Target | Starting method |
|---|---|---|---|
| Q1 | Causal foundation | Aggregate return response per unit of surprise | Partially linear DML, with OLS baseline |
| Q3 | Main ML contribution | Expected firm response given pre-event state and realized surprise | Shrunk interactions, boosting, then sequence architectures |
| Q4 | Predictive extension | Relationship between Q3 prediction errors and subsequent abnormal returns | Per-meeting regressions, ranks, sorts, placebos |
| Q5 | Optional application | Executable gap-strategy returns after costs | Simple constrained portfolios and chronological backtests |
| Q2 | Parked | Transmission through rates, dollar, oil, credit, volatility | No active build or additional data collection |

## Q1 - Aggregate shock response

**Question:** How much do equity values respond to an unanticipated Fed shock?

**Treatment:** A continuous high-frequency monetary-policy surprise, not the announced
rate change. Select and document the surprise's units, normalization and event window.

**Starting model:** Partially linear DML: Y_t = theta * s_t + g(X_t) + error_t.
X contains information available before the event. Estimate E[Y|X] and E[s|X]
out of sample and use the orthogonal residual moment to estimate theta. Start with
regularized linear or shallow tree nuisance models; report OLS beside DML.
Causal interpretation requires the treatment assumptions and model restrictions;
orthogonalization alone does not remove unobserved confounding.

**Data and timing:** The current merged daily table has 30 meetings. The raw USMPD
statement and monetary-event tables each have 262 nonmissing SP500 entries before
scheduled-event and other exclusions. Audit units and coverage for an announcement-window
aggregate analysis. Daily ETF outcomes are a separate window/robustness analysis.
For daily outcomes, account explicitly for press-conference news as well as statements.
Extended coverage of roughly 230 meetings is a planning target, not the current matched sample.

**Inference:** Independent treatment variation lives across meetings. With one outcome
per meeting use suitable heteroskedasticity/serial-dependence robust inference; with
multiple outcomes per meeting account for within-meeting dependence as well.
Meeting fixed effects absorb a meeting-common shock and cannot identify its level coefficient.
Contemporaneous channel variables are post-treatment and are not Q1 total-effect controls.

**Robustness:** Information effects, shock construction, influential meetings, regime
changes and same-day news. Separate unscheduled actions. Preselect a small robustness
set rather than automatically running every shock decomposition.

**Output to Q3:** Treatment definition, alignment, baseline estimates and evaluation
conventions. A fitted Q1 component may be used as an optional Q3 offset only if estimated
using the same historical training cutoff. It is not a required architecture.

## Q2 - Parked mechanism question

No active mechanism decomposition or channel-specific causal claim. Some channel data
already exists, but separating pathways requires additional identifying assumptions
and potentially additional data. Sector and exposure comparisons remain part of Q3;
they do not establish mediation through rates, dollar, oil or credit.

## Q3 - Firm susceptibility (main contribution)

**Question:** Given pre-event firm state z and realized shock s, what response should
we expect from this firm?

**Primary target:** Conditional expected adjusted-price daily return on the FOMC day:
mu(z_it, s_t) = E[r_it | z_it, s_t]. The response runs from the previous trading-day
close to the event-day close. Use the same target for every architecture. Benchmark
relative-return performance separately; do not silently replace the target with a
percentile or factor residual. A future change of target requires a spec revision.

**Inputs:** Firm information available by the previous close (return/volume history,
volatility, beta, sector, and point-in-time fundamentals) plus the realized event shock.
No event-day firm returns or realized event-day market returns enter the predictor.
With daily labels, statement-only versus full-event/statement-plus-press-conference
conditioning must be settled before final evaluation.

**Model progression:**

| Stage | Model | Comparison it resolves |
|---|---|---|
| V0 | Ridge interactions and hierarchical sector/firm shrinkage | Does firm state improve on pooled/sector response baselines? |
| V1 | Gradient-boosted trees on state and surprise | Do nonlinear tabular interactions improve prediction? |
| V2 | Small TCN or masked-patch transformer, fused with fundamentals and shock | Does sequence representation add information beyond tabular state? |
| V3 | Pre-event cross-sectional context (DeepSets/attention) | Does the state of other firms improve prediction? |
| V4 | Graph model with point-in-time economic edges | Do observed economic relationships add value? |

These are comparisons, not a requirement to implement every stage. Establish baselines
first; promote complexity only for meaningful validation gains. A null linear interaction
result does not logically exclude nonlinear signal. Q4 reversal is not a gate for Q3 work.

**Training:** Pretrain sequence encoders on earlier ordinary-day histories, then freeze
or lightly adapt them on training meetings. Ordinary-day data teaches state representations,
not the policy response itself. Auxiliary event tasks are optional; all their observations
and labels must precede the evaluation cutoff. No same-meeting press conference can enter
training while its statement is held out. Scheduled FOMC meetings remain the test population.

**Metrics:** Primary mean per-meeting MSE in return units; secondary per-meeting RankIC
between predicted and realized firm returns, relative-return accuracy and regime stability.
Report gains over pooled, sector and historical-exposure baselines. Quantile heads add
pinball loss and coverage diagnostics without replacing the conditional-mean target.

**Data:** Liquid large caps, initially a manageable universe with a long history.
Audit point-in-time fundamentals and historical membership before choosing the final start
date. Daily bars back to 1998 do not guarantee fundamentals that far back. A present-day
survivor universe limits generalization; document survivorship and missing delisting returns.

## Q4 - Does Q3's prediction error forecast subsequent returns?

**Definition:** For each held-out firm-meeting observation,

    gap_it = r_it - mu_hat_<t(z_it, s_t)

A positive gap means the firm outperformed its predicted response; a negative gap means
it underperformed. Preserve the raw gap in return units. Within-meeting signed ranks and
historically volatility-scaled gaps are secondary representations. Every meeting has rank
extremes, so a rank alone does not establish absolute abnormality.

**Model selection:** Choose Q3 architectures and hyperparameters on earlier response
prediction validation data, freeze them for the outer test block, and save predictions
before calculating future-return outcomes. Do not select Q3 by reversal or strategy P&L.
Q4 model/threshold selection needs its own earlier validation and untouched evaluation.
Prediction errors include omitted information and estimation error, not just overreaction.

**Primary test:** Regress future factor-adjusted returns on the signed raw gap within each
meeting, then aggregate slopes across meetings. Negative slopes indicate reversal; positive
slopes indicate continuation. The proposed primary outcome is next trading-day close to
trading-day t+20 close. This deliberately excludes the first post-event day's return.
Other horizons, RankIC and portfolio sorts are secondary, counted specifications.

**Independent benchmark:** Fit an ordinary-day response model using only historical
non-event data at each cutoff. Its residual is a competing signal. Hold its procedure fixed
across Q3 comparisons. Future-return factor loadings are also estimated before the event;
realized future factors can be used for ex-post adjustment, never as predictor inputs.

**Falsification:** Match ordinary days on pre-event volatility, sector conditions and regime.
Run the ordinary-day benchmark signal on both FOMC and matched ordinary days, alongside
Q3 gaps on FOMC days. An event-trained Q3 model is not automatically valid on an ordinary
day with an invented zero shock. Any such additional placebo requires a stated transport
assumption and separate diagnostics. Compare against raw-return reversal and sector/volatility
controls; account for bid-ask bounce, stale prices, overlapping outcomes and date reuse.

**Interpretation:** A stable gap-return relationship is predictive evidence. Stronger reversal
than matched benchmarks is consistent with overreaction, but does not prove mispricing.
A null is reportable and does not invalidate Q3's predictive contribution.

## Q5 - Trading usefulness (optional)

Proceed only if Q4 has a stable held-out relationship that survives benchmark/placebo
comparisons. Start with a simple long-short portfolio: reverse the gap if reversal is
validated, or follow it if continuation is validated. Fix direction on validation data.

The gap is available only after the event close. The default test enters at next trading-day
close, consistent with Q4, and exits at t+20 close. Model turnover, spreads, slippage,
borrow costs and exposure constraints. An earlier entry needs separate executable-price data.
Use chronological held-out returns and selection-aware reporting. Logged-policy OPE is not
a default solution for price-history backtests; revisit only with an appropriate action/reward
logging setup. Learned sizing or regime gates are later comparisons to simple rules.

## Shared evaluation protocol

1. Split by whole meetings in chronological order. Tune within earlier data and preserve
   an untouched outer test. Purge by actual label end dates, including auxiliary-event labels.
2. Fit preprocessing, feature selection, pretraining and calibration only on data available
   at each cutoff. Save models, cutoffs, data fingerprints and out-of-sample predictions.
3. Account for within-meeting dependence and serial dependence/overlapping outcomes.
   Independent meeting resampling alone does not handle dependence across meetings.
4. Predeclare one primary specification per active question, model promotion thresholds,
   shock definition, units, horizons and metric aggregation. Count alternative specifications.
5. Predeclare exclusions (unscheduled events, same-day macro news, halts and earnings).
   Do not select the primary sample using subsequently realized earnings or 8-K events;
   post-event news exclusions belong in clearly labeled sensitivity analyses.
6. Simulate known heterogeneous responses, noise and optional reversal to validate both
   Q3 recovery and Q4 false positives. Include a case where Q3 works but reversal is absent.

## Dependency chain and next decisions

    Q1: treatment + aggregate baseline
                 |
                 v
    Q3: expected firm response (main contribution)
                 |
                 v
    Q4: held-out prediction gap -> future-return test
                 |
                 v
    Q5: executable strategy, conditional on Q4 evidence

    Ordinary-day benchmark -> independent Q3/Q4 comparison
    Q2: parked

Next: finalize event/shock alignment, audit firm-feature history, specify chronological
folds and practical model-gain thresholds, then compare V0/V1 and the first sequence model.
Architectures in methods.md are candidates for discussion, not finalized commitments.
