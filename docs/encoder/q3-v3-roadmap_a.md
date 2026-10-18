# Q3 v3 roadmap — Astra

Written 2026-10-17 against `9fad797` and the completed v2 runs. This is an alternative
research proposal for the encoder project, within the existing Q-spec. No new training
was performed for this document.

## My position

**Give market exposure a strong, explicit model. Give the transformer a separate job whose
success cannot be explained by reproducing that model. Then improve the information that
the separate job receives.**

I would pursue three bets:

1. **A specialist residual branch:** test whether a representation can predict firm
   differences outside the measured-beta span, while the base retains the useful beta signal.
2. **A different pretraining task:** make that specialist insensitive to an added market
   component, rather than rewarding it for reconstructing or forecasting ordinary returns.
3. **More direct exposure data:** measure how a firm's cash flows actually reprice with rates,
   especially floating-rate debt after hedging. Broad leverage and debt maturity are imperfect
   substitutes for this information.

Two supporting experiments are worth doing cheaply: relative firm-state features and a
shorter memory in the response readout. I would not combine all five changes in one run.

The goal is still a better conditional mean of the **raw firm FOMC-day return**. Making a
network mathematically different from CAPM is not enough; it must predict something useful.

## 1. What the existing work tells us

The latest saved 2022–2025 predictions give these mean per-meeting MSEs:

| Model | MSE |
|---|---:|
| Original CAPM | 5.7251 |
| Six-arm v2 consensus | 5.7225 |
| CAPM with last and smoothed beta | 5.7071 |

The consensus has not established an advantage over either reference. The stronger-beta
model is a promising reference, not a statistically established population winner. The
2022–2025 period has been examined for both v1 and v2; it is not a fresh test for v3.

Important negative evidence:

- Gaussian and masked pretraining did not improve transfer; ordinary-day `s=0` training tied
  scratch. The old machinery was archived.
- Sector and regime gate passes did not turn into reliable out-of-sample improvements.
- Ordinary-day and CPI/jobs-day rate loadings did not transfer convincingly to Fed sensitivity.
- Sector inputs, larger branches, more firms and longer-window diagnostics have already been
  tried. They are not untouched opportunities.
- The v2 network mostly adjusts the baseline component. However, the slope-shock arm stores
  an additional shock response in `a_hat`, so not all reported `a` adjustment is pure drift.

Sources: [v2 results](v2-results.md), [v1 report](../reports/encoder-report.md), and
[code/results review](../reports/encoder-review-2026-10-17.md).

These results weaken the case for more generic pretraining. They do not establish that
firm-specific monetary sensitivity is absent, or that suppressing beta will reveal it.

## 2. Proposed architecture: useful beta plus a restricted correction

### 2.1 Reference model

Use the existing `CAPMBetas(window=252, both=True)` as the main development reference:

```text
mu_base = c + d1*beta63 + d2*beta_smooth
            + (k1*beta63 + k2*beta_smooth)*s
```

`beta_smooth` is the average of observed rolling 63-day beta estimates in the 252-day history.
It is **not** a beta regression estimated on 252 raw returns. Keep `capm_pooled` for continuity.

The initial experiment uses the original statement surprise. Extra shock components are a
separate experiment with equally informed baselines, not a hidden benefit of the network.

### 2.2 An unconstrained control and a projected specialist

For meeting `m`, define the pre-event cross-sectional matrix:

```text
B_m = [ones, beta63, beta_smooth]
Q_m = orthonormal basis for the column space of B_m
P_m = I - Q_m @ Q_m.T
```

Use rank-aware QR/SVD; the two beta columns can be nearly collinear. Do not form an explicit
500-by-500 projection matrix: apply `P_m*v = v - Q_m*(Q_m.T*v)`. A ridge inverse is a soft
projection and should not be called exactly orthogonal.

The specialist predicts two unconstrained vectors from pre-event inputs, then projects them:

```text
h_i = HistoryEncoder(history_i, mask_i)
z_i = Fuse(h_i, fundamentals_i, pre_event_context_m)
delta_a_m = P_m * head_a(z_m)
delta_b_m = P_m * head_b(z_m)
mu_m = mu_base_m + delta_a_m + delta_b_m*s_m
```

Both heads learn. The base carries baseline drift and beta-related shock response; the
specialist can add baseline differences and shock-related differences outside that span.
The projection uses the intended firm universe known before the event, never a universe
selected by realized returns. Missing outcome rows are omitted from scoring, not used to
redefine the prediction universe. Use the same eligible universe and missingness rules for
all arms.

**Why I want this experiment:** it makes the proposed contribution falsifiable. An improvement
cannot be an arbitrary meeting-by-meeting linear combination of these two measured betas.
It can still be nonlinear beta exposure, beta measurement error, another factor, or noise;
orthogonality is not causal identification.

### 2.3 Keep raw-return MSE; no target purification is necessary

For a frozen base, residual `e = y - mu_base`, and correction `d` in the range of `P`:

```text
||e - d||² = ||P*e - d||² + ||(I-P)*e||²
```

The second term is constant with respect to the specialist. Thus optimizing raw-return MSE
already gives that branch the projected residual gradient. There is no need to subtract the
realized event-day market move from the target or pretend it was available as an input.
The identity assumes a fully observed, equally weighted cross-section. With missing targets
or unequal firm weights, it does not generally hold for the same `P`. Always optimize the
actual masked raw-return loss; do not substitute a projected-target loss in that case.
Keep the prediction universe fixed by pre-event eligibility even when scoring omits labels.

This preserves the raw evaluation target within a deliberately restricted model class. It
does not preserve every possible conditional-mean function: any useful correction inside
the removed span is prohibited. That is why an unprojected correction is an essential control.

### 2.4 First comparison

| Arm | Base | Correction | Question |
|---|---|---|---|
| A0 | Two-beta CAPM | None | Can we improve on the stronger reference? |
| A1 | Same | Small MLP on summary features, projected | Is sequence modeling necessary? |
| A2 | Same | Existing transformer, centered only | Does upgrading the base suffice? |
| A3 | Same | Same transformer, projected | Does specialization help beyond centering? |

A1 receives the same fundamentals/context and explicit history summaries: recent returns,
drawdown, volatility, volume changes and the beta levels. A2/A3 share width 64, three blocks,
five-day patches, seeds, chronology, optimizer budget and raw-return loss. No enlargement.

**Success:** A3 improves held-out raw-return MSE over A0 and adds something over A1; its gain
is not entirely one crisis block. **Failure:** projection removes the gain or every correction
ties the base. Report that; do not redefine success as "the output is no longer beta."

## 3. Pretraining bet: invariance to added market exposure

This is my most speculative proposal. It is a different hypothesis from the previous
next-return and reconstruction tasks, not a reason to restore their entire pipeline.

**Hypothesis:** a representation trained to preserve firm-specific history while ignoring a
controlled added market component will be a better initialization for the projected branch.

### 3.1 Paired historical views

Use ordinary historical windows strictly before the applicable inner validation cutoff.
For each firm-window, make two views with different small market-loading perturbations:

```text
return_day(view 1) = observed_return_day + delta1 * observed_market_return_day
return_day(view 2) = observed_return_day + delta2 * observed_market_return_day
```

The entire market series is historical relative to the window endpoint. Start with
`delta1, delta2` independently uniform in [-0.25, 0.25] beta units as a proposed fixed setting;
audit the resulting feature distributions before fitting. Reject mechanically invalid views
and preserve missing days. This is a controlled numerical augmentation, not a claim that
the views are economically valid counterfactual firms.

Recompute **all affected channels** from the perturbed returns: rolling volatility, relative
returns and rolling beta. Fetch sufficient warm-up history to do so. Keep volume and its
mask unchanged. Never perturb returns while leaving a contradictory original beta channel.
Apply a scaler fitted only on the pretraining training corpus and carry it into transfer.
Give the scratch transfer control that same scaler, so a comparison of initialization does
not also change preprocessing.

Under a linear market-component description, changing the added loading preserves the
firm-specific residual path. This encourages the specialist to retain features beyond the
amount of market exposure. It may also suppress useful interactions; the unaugmented control
and the separate beta base are essential.

### 3.2 One self-supervised objective

Map each view through the shared history encoder and a temporary projection MLP. Make their
embeddings agree, while explicitly preventing constant or redundant embeddings:

```text
L_pre = 25*view_agreement + 25*variance_floor + covariance_penalty
```

Use the definitions and coefficient convention in the [VICReg paper](https://arxiv.org/abs/2105.04906)
and [authors' implementation](https://github.com/facebookresearch/vicreg). VICReg supplies a
collapse-control mechanism; its published results do not establish financial transfer.
The market-loading augmentation here is our proposed adaptation.

Sample multiple dates and firms per batch; do not let a batch consist of one firm's adjacent
windows. Reserve a calendar validation block and purge overlapping windows for this first
comparison. Record exact dates rather than inferring them from values. Use the current small
encoder, a temporary two-layer projection head, and a bounded pretraining budget. Discard only
the projection head and transfer the history scaler with the encoder.

No ordinary yield change is labeled as a Fed shock; no ordinary-day b is treated as a Fed b.
The self-supervised objective teaches an invariance, not monetary sensitivity.

### 3.3 Controls and rejection criteria

Compare A3 from scratch against the same projected model initialized by this objective.
Include a matched mild-corruption/VICReg control that uses ordinary missing-patch views
without market-loading perturbation, if the candidate earns a larger test. Otherwise a gain
could reflect generic pretraining rather than the proposed invariance.

Before transfer, check embedding variance, effective dimension and sensitivity to changes
in genuinely firm-specific history. Low response to added market exposure alone is insufficient:
a constant representation also passes that test. A model can preserve idiosyncratic noise or
firm identity without learning useful state.

**Promotion depends on Fed-day transfer**, against A0, A1 and A3 on matched development folds.
Good pretraining loss or weaker beta probes do not count as Q3 success. Do not add earnings,
volatility and return forecasting losses simultaneously if this task fails.

## 4. Data bet: observed net rate exposure, not another leverage proxy

The existing tests constrain the value of broad ratios. They do not exhaust the value of
actual debt repricing and hedging information.

The highest-value new feature I would investigate is:

```text
net_floating_exposure = net floating-rate liability exposure after disclosed hedges / assets
```

Keep floating-rate assets, floating-rate liabilities, hedge direction and hedge coverage
separate when the disclosure supports them. Define netting rules explicitly; a swap can
increase or reduce floating exposure. Treat financial firms separately initially because
asset-side repricing is central to their business.

Why this is more than adding `dlcq / debt`: debt due soon and debt that reprices soon are
different. The current WRDS pulls contain broad accounting quantities, not a verified
historical series of net floating-rate exposure. Research on unhedged bank debt provides a
mechanism worth testing, including its dependence on the rate environment; it does not
guarantee an incremental daily-return forecasting edge in our sample.
[Ippolito, Ozdagli and Perez-Orive](https://www.federalreserve.gov/econres/feds/files/2017026pap.pdf)
and [Federal Reserve discussion of swap exposure](https://www.federalreserve.gov/econresdata/notes/feds-notes/2016/corporate-bond-issuers-swap-exposure-to-rising-interest-rates-20160526.html).

### Feasibility pilot before a collection project

Audit 50 development-period filings selected across sectors, firm sizes and reporting years
without looking at subsequent return outcomes. Extract:

- Fixed versus floating debt amounts, including explicit unknowns.
- Hedge amounts, direction and any limits on interpretation.
- Disclosed annual cash-flow/interest-expense effect of a standard rate change, when present.
- Source passage, units, as-of date, filing acceptance date and amendment status.

Do not combine a disclosed debt fair-value sensitivity with an earnings/cash-flow sensitivity
as if they were the same quantity. Normalize units before building features. A small hand-audited
sample should establish whether the measurement is reproducible and historically available.

Use original filings available before the cutoff. EDGAR's submissions and company-facts APIs
are useful discovery tools, but the frames endpoint selects last-filed facts and is not itself
a point-in-time panel. Later restatements must not be assigned to earlier meetings.
[SEC API documentation](https://www.sec.gov/search-filings/edgar-application-programming-interfaces).

Pilot decision: proceed only if usable exposure can be measured consistently for a meaningful
cross-section across multiple development eras. If coverage supports only a narrow later
sample, state that and compare every model on exactly that sample. Do not turn 2022–2025
into the training-design sample because those filings are easier to obtain.

### First predictive test

Add exposure and exposure × surprise to a small regularized model with the two-beta base.
One prespecified interaction with the pre-event short-rate level is a reasonable second
comparison. Fit all scaling, clipping and penalties on earlier data.

Compare actual exposure against the existing leverage/maturity proxies. Only if it adds
held-out information should it enter the transformer. Document extraction can use a model,
but every retained value needs a source and uncertainty flag; do not turn a narrative guess
into a numeric fact.

## 5. Fine-tuning and adaptation: make sparse meetings count

### 5.1 Start from zero correction and adapt gradually

Zero-initialize the final `delta_a` and `delta_b` readouts, so the model initially predicts
the strong base exactly. This makes early performance and degradation interpretable.
It does not guarantee the model will stay near the base.

For the pretrained specialist, compare frozen-encoder readout training with one staged
schedule: train both readouts, then unfreeze the top transformer block at a lower learning
rate. Keep the total optimizer-step budget comparable and select on the complete deployed
raw-return prediction. Lower layers can remain frozen in the first transfer experiment.

Do not freeze a's learning or train b alone. The v2 b-only result argues against making that
the default. Separate diagnostic contributions from training objectives.

### 5.2 Shorter memory in the small readout before another regime network

Test one five-year exponential recency half-life against expanding equal weights:

```text
w(meeting) = 2 ** (-years_before_training_cutoff / 5)
```

This is a proposed fixed comparison, not a searched optimum. Weight the base and the
readout consistently, and normalize weights so comparisons do not silently change the
effective penalty or step size. Record weighted meeting concentration and shock support.
The descriptive count `(sum(w)**2)/sum(w**2)` is not an inference sample size.

Start with a regularized linear/small-MLP correction. Then, if useful, keep a shared history
encoder and adapt only a small recent readout. This tests changing response mappings without
asking a new deep network to relearn state from a handful of recent meetings.

Avoid retrospective era labels or kernels centered on future test-block conditions.
For each outer block, the reference cutoff and weighting rule are fixed using information
available at its start. Regime FiLM's previous failure does not settle this weighting test,
but era changes do not guarantee it will help.

### 5.3 Post-fit shrinkage is optional calibration, not new signal

A single correction multiplier `gamma` in `[0,1]` is a cheap way to test whether the network
adds useful but overlarge adjustments:

```text
mu = mu_base + gamma * delta_mu
```

Estimate it only from chronological out-of-fold predictions inside the outer-training period.
Do not fit it on the outer test or on the same observations that trained the underlying model.
Use one gamma initially; per-sector or per-shock calibrators create another sparse estimation
problem. Gamma near zero is evidence to prefer the base, not a reason to tune more calibrators.
Keep this optional until there is a candidate worth calibrating.

## 6. Relative firm state before cross-sectional attention

Retain absolute characteristics and add their pre-event within-meeting ranks, plus a few
summaries such as the cross-sectional dispersion of beta and leverage. A firm's relative
position may be more stable across decades than its raw size or accounting ratio.

Use a small MLP first. Compute ranks on the pre-event eligible universe, preserve missingness,
and keep denominators/coverage as metadata. Do not rank using event returns or a universe
filtered by realized jumps.

If this helps, a small DeepSets aggregate is the next test:

```text
market_state = mean_j(phi(pre_event_firm_state_j))
correction_i = rho(firm_state_i, market_state, pre_event_context)
```

Compare against the explicit cross-sectional summaries before implementing all-to-all
attention. Sharing information across 500 firms does not create 500 independent policy shocks.

## 7. Prerequisites that affect interpretation

Complete these in code/tests before a new authorized fitted comparison:

1. **Inner validation:** fit the base and preprocessing before the inner validation boundary.
   Define any later refit explicitly; do not train residuals against a base that has already
   used validation outcomes.
2. **Deployment alignment:** apply centering/projection identically during training, validation
   and prediction. Choose checkpoints using the deployed base-plus-correction score.
3. **Missing data:** preserve absent macro values and availability. Current all-channel day
   masks are insufficient for independently missing appended macro channels; use an
   encoder-local per-feature mask or documented jointly observed subset.
4. **Fundamental availability:** the existing `rdq < meeting` merge is necessary but does not
   prove the downloaded values are unrevised historical vintages. Audit availability and
   amendments before claiming precise historical accounting inputs.
5. **Outputs:** save base prediction, each correction component, every shock input and loading,
   full config, code revision, member seeds, selected epochs and source run IDs. Never hide an
   extra shock contribution inside a column interpreted as pre-event drift.
6. **Information contract:** the initial primary comparison uses no event-day equity move as
   an input or classifier. A policy/info sign split based on that move is a separately labeled
   conditioning experiment with the same information supplied to its baselines.

The detailed findings and a forward-only demonstration of the validation mismatch are in
[the review](../reports/encoder-review-2026-10-17.md).

## 8. Evaluation and stopping rules

### What counts as evidence

- Primary: equal-meeting raw-return MSE versus the two-beta reference; retain the original
  CAPM comparison. Also compare against the summary-feature correction.
- Secondary: cross-firm MSE, RankIC, calendar-block stability, large-surprise performance and
  the size/concentration of the correction. Thresholds are fixed from earlier development.
- Show a-only, b-only and full correction predictions. Their improvements need not add up
  because of cross terms. A useful drift correction is not evidence of monetary sensitivity.
- For projected arms, measure projection orthogonality numerically and report its sensitivity
  to beta definitions. A beta-neutral output is not an independently identified causal effect.
- Use paired meeting-level differences and a prespecified dependence-aware sensitivity check,
  such as resampling short consecutive meeting blocks. Seed spread is a separate quantity.
- Assess power on the available meeting counts before a confirmatory claim. Do not replace
  weak precision with hundreds of in-sample p-value gates.

Model selection remains on development data through 2021. That data has already been explored
extensively, so new results there are exploratory even with chronological splits. Any further
2022–2025 check is reused evaluation. Reserve the still-unused 2026-and-forward observations
for a separately authorized, fixed-specification check and be candid about its small size.

### Bounded sequence of work

| Stage | Deliverable | Stop or advance based on |
|---|---|---|
| 0: code/tests | Correct validation, deployed loss, masks and saved components | Invariance/chronology tests pass; no fitted research run |
| 1: base/correction | A0–A3, fixed architecture and matched folds | Does anything add beyond the stronger beta reference and summaries? |
| 2a: pretraining pilot | Market-loading-invariant initialization vs scratch A3 | Actual transfer; no promotion on pretraining loss |
| 2b: data feasibility | Audited filing-based exposure sample | Reliable historical measurement, coverage and provenance |
| 3: one targeted extension | Either exposure, recency or relative state | One hypothesis at a time; no Cartesian-product grid |
| 4: locked candidate | Full record of selected and rejected configurations | Practical gain, uncertainty and independent evaluation budget |

Stage 2a may be worth one bounded attempt even if scratch A3 fails: lack of a suitable
representation is its hypothesis. Do not repeatedly enlarge it after negative transfer.
Stage 2b is a data-measurement project and can be useful even if all neural arms fail.

Before the first fitted run, record exact folds, seeds, step budgets, regularization choices,
pretraining dates and the practical gain threshold. Use three fixed evaluation seeds initially;
do not add seeds only to the apparent winner. Avoid training a six-arm consensus merely
because the previous analysis used one. No runtime or GPU-price promise is made here.

## 9. Proposed implementation boundaries

These are suggested new files/APIs, not components that already exist.

- `encoder/fedenc/v3.py`: base-plus-correction arms, encoder choice and controlled experiment
  factories. Preserve v1/v2 code and saved runs.
- `encoder/fedenc/projection.py`: whole-meeting rank-aware projection, identity and permutation
  checks. Keep it encoder-local until another project needs it.
- `encoder/fedenc/pretrain_invariance.py`: one optional paired-view objective, explicit corpus
  dates and scaler/checkpoint transfer. All trained artifacts go through `ResultStore`.
- Shared as-of exposure ingestion/catalog entries belong in `core`; encoder-only augmentation
  stays in `encoder`. New shared baselines belong in `fedcore.q3.baselines`.
- `encoder/tests/test_v3.py`: unit tests for chronology, projection under rank deficiency,
  missing labels/features, full-meeting versus chunked inference, zero correction, and
  consistency of training/validation/prediction. Tiny synthetic smoke fits only.

Synthetic checks should include both a beta-only world and a known additional firm response.
Passing the latter proves implementation capability; failing to improve real data remains
an admissible research outcome.

## 10. What I would defer

- Another broad ratio sweep, ordinary rate-day pretraining, or larger model without a changed
  hypothesis that addresses the existing negative tests.
- Unsupervised loss improvement as evidence of Fed sensitivity.
- Huber/clipped-target/rank losses as an automatic fix for conditional-mean prediction.
- Training on only the biggest events, pooling unscheduled events, or silently changing the
  outcome horizon. Each changes the experiment and needs its own specification.
- Claiming that forcing a network outside beta guarantees a useful result. It may correctly
  learn zero. That is a cleaner conclusion than rediscovering a weak benchmark.

**My first implementation would be A0–A3 with correct validation. My most interesting
pretraining experiment is the market-loading invariance task. My highest-value data bet is
net floating-rate exposure after hedging. Those are the three contributions I would put
forward for v3.**
