# Q3 v3 debrief — Astra

Written 2026-10-18 against `f83275e`; updated for the big-movers evidence at `e07deb3`.
Reviewed the v3 code, manifests, saved predictions,
and [results report](q3-v3-results.md). This updates the priorities in
[my roadmap](q3-v3-roadmap_a.md). All new calculations below use existing predictions;
no model was fitted, no research run was saved, and no holdout was opened.

## My assessment

**Keep the two-beta CAPM as the reference. The tested corrections have not earned a claim
of better overall prediction. There is a small, concentrated large-surprise lead that mostly
comes from the shock-response head. The big-movers follow-up moves a small pre-event peer-stress
test ahead of pretraining. Better exposure data remains my next substantial data investment.**

The [big-movers follow-up](big-movers-followup_a.md) reproduces the earnings concentration
and rescores the saved models. Excluding earnings-flagged rows does not uncover an overall
winner, but most of the large-surprise improvement survives. It updates the next-step order
below and distinguishes retrospective news diagnostics from valid pre-event model inputs.

I would revise my original enthusiasm for projection: it succeeded as a diagnostic restriction,
but it has not delivered a useful overall forecasting improvement. It is worth retaining as
a way to test a contribution beyond measured beta, not as evidence that such a contribution exists.

The strongest negative result is the agreement between simple and sequence models. A1, A2 and
A3 all tie their own base. That lowers the priority of more depth, cross-firm attention,
recency variants and another broad feature sweep on the current inputs.

## 1. What was actually tested

| Roadmap proposal | Status | What we learned |
|---|---|---|
| Strong two-beta base, A0 | Run | Best reported raw-return MSE among the fitted v3 arms |
| Summary MLP and transformer corrections, A1–A3 | Run | No established overall gain; no sequence-model advantage |
| Exact projection off `[1, beta63, beta_smooth]` | Run | Works as a restriction; no established overall benefit over centering |
| Five-year recency weighting, R1 | Run | No improvement over R0 |
| Within-meeting ranks, R2 | Run | No improvement; worse after 2004 in the reported comparison |
| Signed-shock sector response, G4 | Run | In-sample gate pass; reported follow-up fails out of sample |
| Market-loading-invariant pretraining | **Not run** | This proposal remains untested |
| Direct net floating-rate/hedging exposure | **Not collected/tested** | This proposal remains untested |
| Frozen versus staged adaptation of the proposed pretrained encoder | **Not run** | Scratch v3 does not answer this question |

A1 uses `SummaryEncoder`: channel means and standard deviations over 21, 63 and 252 days,
followed by the shared nonlinear response model. It is a useful no-sequence control, but
does not include every explicit drawdown/momentum feature proposed in the roadmap. The ridge
arms use the richer tabular summaries. This distinction does not justify another broad sweep.

## 2. The headline loss gap mostly comes from the base

I reproduced the reported ensemble scores from 80,756 firm-meeting rows across 161 development
test meetings. Average the saved predictions across seeds first, then compute each meeting's
MSE and give meetings equal weight. The manifest's mean of seed-specific MSEs is a different
quantity and is slightly higher for the networks.

| Fitted arm | Raw-return MSE |
|---|---:|
| A0, full-window two-beta base | 5.902417 |
| R0, ridge correction | 5.906841 |
| R2, ranks | 5.911110 |
| R1, recency | 5.916721 |
| A3, projected transformer | 5.932155 |
| A2, centered transformer | 5.932673 |
| A1, projected summary MLP | 5.932986 |

The neural arms retain a base estimated on the inner 85% training block; A0 uses the full
outer-training window. Their own base scores **5.932972**. Thus the approximately 0.030 gap
to A0 is not evidence that the neural correction itself costs that much. Against their own
base, the corrections are approximately zero, as the existing report correctly explains.

### A useful consequence of exact projection

For A1 and A3, the correction `d` is orthogonal within each meeting to the two-beta base's
entire span. If `mu_old` and `mu_new` are any two predictions in that same span, then:

```text
MSE(y, mu_old + d) - MSE(y, mu_old)
  = MSE(y, mu_new + d) - MSE(y, mu_new)
```

This holds for the same fully observed, equally weighted firm cross-section. It does not
generally hold after changing the universe, firm weights or projection, or for A2's merely
centered correction.

Consequently, we can combine saved A0 predictions with the saved A1/A3 corrections without
training anything. This is a descriptive arithmetic check, not another independently
evaluated fitted model:

| Saved-prediction combination | MSE | Difference versus A0 | Paired 95% meeting-bootstrap interval |
|---|---:|---:|---|
| A0 + existing A1 correction | 5.902430 | +0.000014 | [-0.004079, +0.003830] |
| A0 + existing A3 correction | 5.901600 | -0.000817 | [-0.003138, +0.001495] |

A3's point improvement is **0.014%** of A0's loss. A four-consecutive-meeting circular block
bootstrap also includes zero: [-0.002984, +0.001390]. Correcting the comparison does not turn
the model into a convincing winner.

**Implementation implication:** select the projected correction on a valid inner split, then
fit the deployment base on all outer-training meetings. Re-estimating that base alone does
not require retraining a strictly projected correction. A full refit of the correction and
its scaler is a separate choice; the current correction still trained on only the inner block.
Keep validation chronological and the outer test untouched throughout selection.

## 3. The large-surprise lead deserves a precise reading

The post-hoc threshold `|s| >= 5bp` leaves ten meetings. I reproduced the reported improvement
and decomposed it using the saved `a_hat` and `b_hat` outputs.

These are **prediction ablations of jointly trained models**, not new b-only fits. Negative
numbers mean lower MSE than the arm's own base. Because the corrections are centered, these
within-meeting MSE differences are also the raw-return MSE differences.

| Arm | Full correction | a-only correction | b × s correction only | Full-correction 95% interval |
|---|---:|---:|---:|---|
| A1, projected MLP | -0.018367 | -0.005068 | -0.013729 | [-0.043243, -0.000845] |
| A2, centered transformer | +0.012629 | +0.006720 | +0.004308 | [-0.006815, +0.039897] |
| A3, projected transformer | -0.013811 | -0.001182 | -0.011943 | [-0.030030, -0.001289] |

The a-only and b-only gains do not add to the full gain because the squared loss includes
their cross term. Still, this is a better lead than a gain attributable solely to drift:
**most of the projected arms' tail improvement comes from the shock-response correction.**

Its limits are substantial:

- **Nine of the ten meetings are in 2001–2008.** The only later one is 2011-09-21. The results
  report originally counted eight; the listed dates imply nine. This slice provides little
  evidence about later policy environments.
- **Two meetings account for 87% of A3's net improvement and 93% of A1's:** 2004-01-28 and
  2011-09-21. A3's gains on those dates are 0.07010 and 0.05015. This is concentration,
  not proof that either date is an error or should be excluded.
- It is not literally a one-meeting result: every leave-one-meeting-out mean remains negative
  for A1 and A3. Excluding both standout dates shrinks their mean gains to about 0.00161 and
  0.00223. A3's exploratory iid-bootstrap interval then only barely excludes zero. None of
  these post-hoc exclusions supplies independent confirmation.
- A3 improves on seven of the ten dates; the mean gain has the same sign in all three saved
  seeds. Seeds share the events, so this addresses optimization variability, not event uncertainty.
- The MLP is at least as competitive as the transformer. This is not evidence for sequence
  modeling, and being outside two measured betas does not exclude nonlinear beta or other factors.
- Thresholds and multiple arms have been inspected. These intervals are unadjusted exploratory
  summaries; their just-below-zero endpoints should not be treated as a discovery.

### Do not infer that we should train only on large surprises

For the 161 evaluation meetings, the ten 5bp+ events already account for **59.6% of the sum
of squared surprises**, although they contribute about 15.2% of A0's summed meeting MSE.
The concentration statistic `(sum(s²))² / sum(s⁴)` is only **17.4**. This is a descriptive
shock-concentration measure, not an effective inferential sample size.

The curvature of a simple squared-error response model with respect to a loading scales with
`s²`; actual network information also depends on features, noise, projection and training
folds. The accounting above concerns evaluation meetings. It does not establish gradient
starvation during training. Removing quiet meetings could further concentrate the fit in a
few old events and weaken estimation of `a`.

V2 already tested shock weighting and b-only training. The latter was worse. These new
prediction ablations are not a reason to repeat b-only training or automatically change the loss.

## 4. What I would do next

### First: make the current comparison reproducible and keep A0 as the reference

This is a small engineering step, not another architecture experiment:

1. Make the full-training deployment base explicit for projected arms. Retain the valid
   inner selection procedure; do not compare models using different base-training windows
   without also showing the matched-base comparison.
2. Save `mu_base`, `a_base`, `b_base`, `delta_a`, `delta_b`, shock, every member's selected
   epoch, checkpoint/scaler lineage, code revision and a fingerprint of the actual panel.
3. Commit a scoring utility for the existing predictions: overall raw MSE, fixed shock bins,
   a/b prediction ablations, paired meeting uncertainty and event influence. Keep all new
   tail analyses explicitly exploratory.

There is no need to spend another GPU run merely to discover whether replacing the base
recovers the 0.030 loss gap: the saved-prediction calculation already answers that.

### Next small experiment: pre-event peer stress

The big-movers narratives suggest a more specific hypothesis than static sector labels:
the group's condition before the meeting may modify its response. Compare the two-beta base,
that base with own-firm state and static group controls, and the same model with one trailing
63-day leave-one-out peer-state measure plus its surprise interaction. Keep both baseline and
shock-response terms. Use as-of groups and matched samples, with overall raw-return MSE primary.

The [follow-up](big-movers-followup_a.md) gives the controls, timing requirements and the
earnings-exclusion results. This is the next small fitted proposal; no new fitting was done
for either document. A failed simple comparison would not justify cross-firm attention.

### Larger data bet: direct exposure measurement

Move the net floating-rate/hedging proposal ahead of generic pretraining in priority. The
current results weaken the hypothesis that the existing price histories and broad ratios
contain enough additional signal for a different architecture to extract.

**Feasibility deliverable:** 50 development-period nonfinancial firm filings, stratified across
sectors, size and earlier/later years without selecting on return outcomes. Recover, where
explicitly disclosed:

- Floating-rate liabilities and assets, separately.
- Hedge amounts and direction; distinguish converting fixed to floating from the reverse.
- Net floating-rate exposure divided by assets.
- A disclosed cash-flow/interest-expense sensitivity to a rate change, when comparable.
- Original filing availability, source passage, units, missingness and uncertainty.

Audit measurement and coverage before attempting a predictive fit. Do not treat missing
disclosure as zero exposure, debt maturity as floating exposure, or a debt fair-value
sensitivity as an earnings sensitivity. If consistent historical coverage is too narrow,
stop the collection expansion and report that limitation.

**First authorized fitted comparison:** two-beta base versus that base plus exposure and
exposure × surprise in a small regularized model. Use the same covered sample for every arm,
compare against existing leverage/maturity proxies, and allow both baseline and shock-response
terms. Add the feature to the transformer only after it provides useful held-out information.

This is a mechanism-motivated hypothesis. Research connects unhedged bank debt to monetary
policy sensitivity, but does not establish forecasting gains in our sample.
[Ippolito, Ozdagli and Perez-Orive](https://www.federalreserve.gov/econres/feds/files/2017026pap.pdf).
For historical availability, use original filings rather than treating latest-filed facts as
point-in-time observations. [SEC API documentation](https://www.sec.gov/search-filings/edgar-application-programming-interfaces).

### Optional neural bet: one controlled invariance-pretraining experiment

The v3 results did **not** test the proposed market-loading augmentation. It remains a
reasonable bounded hypothesis, but I would lower its priority after the scratch-model results
and the more specific peer-stress hypothesis in the big-movers follow-up.

If we fund another neural experiment, compare exactly these three initializations of the
same projected A3 architecture:

| Initialization | Purpose |
|---|---|
| Scratch | Current reference with identical preprocessing |
| Mild-corruption VICReg | Control for generic self-supervised pretraining |
| Market-loading-perturbation VICReg | Test the proposed invariance specifically |

Use the same historical corpus, scaler, small encoder, transfer schedule and evaluation seeds.
Recompute all return-dependent channels after the augmentation, preserve missingness, and
keep corpus dates before inner validation. Keep both a and b trainable. Use one transfer
schedule initially; adding frozen/full/top-block variants immediately would multiply the search.

The [VICReg paper](https://arxiv.org/abs/2105.04906) motivates collapse control, not financial
transfer. Promotion requires an overall raw-return gain over A0 and an improvement over the
matched scratch/generic-pretraining controls, with evidence beyond the two standout events.
Choose the practical gain threshold and compute cap before fitting. A tiny improvement on
the same ten events alone would not justify a larger campaign.

If it fails, stop this pretraining branch. Do not add several auxiliary tasks until something
looks positive. Better reconstruction, weaker beta probes or lower pretraining loss are not
substitutes for Q3 performance.

## 5. What I would stop pursuing for now

- More depth, all-to-all cross-firm attention, or a larger training universe on the same inputs.
- Another recency/rank/sector/sign grid after this batch's negative comparisons.
- Treating every gate pass or narrow shock-threshold interval as a reason to launch a new run.
- Increasing the tail weight or dropping quiet meetings merely because the tail slice looks better.
- Reopening 2022–2025 to choose between v3 variants. It is reused evaluation; new development
  results are exploratory too. Preserve 2026-and-forward for a fixed, separately authorized check.

A credible research outcome remains that a five-parameter beta model captures the currently
demonstrable predictive structure. The next attempt to improve on it should add information
or test one specific representation hypothesis, with an explicit stopping point.

## 6. Verification details and remaining limits

Sources: `encoder/fedenc/v3.py`, `core/fedcore/q3/evaluate.py`, `encoder/fedenc/runs.py`,
`core/fedcore/q3/contracts.py`, and these append-only runs under
`encoder/results/q3_susceptibility/`:

```text
20261017-232247_v3-to2021-a0
20261017-232514_v3-to2021-r0
20261017-232751_v3-to2021-r1
20261017-233056_v3-to2021-r2
20261017-233059_v3-to2021-capm-pooled
20261017-233233_v3-signed-gate
20261018-033103_v3-to2021-a1
20261018-042938_v3-to2021-a3
20261018-043016_v3-to2021-a2
```

The scores above use aligned `(meeting, firm_id)` rows with identical saved targets. The
three-seed neural predictions are averaged before scoring. Shock values are recovered from
`mu = a_hat + b_hat*s`; the panel's `s` unit is 10bp. Bootstrap calculations use 20,000
replicates with NumPy seed 0. The block sensitivity resamples circular blocks of four
consecutive meetings and truncates each draw to 161 meetings. Neither bootstrap corrects
for the accumulated model and slice search.

**Recovering the unsaved base without fitting:** for each meeting, form an orthonormal basis
`Q` from `[ones, A0.a_hat, A0.b_hat]`. Those columns span the two-beta space when its two
coefficient vectors are independent. The smallest/largest singular-value ratio is at least
0.00477 in these predictions. Project A1/A3's saved `a_hat` and `b_hat` onto `Q` to recover
their base, then subtract it to obtain the corrections. The independently recovered A1/A3
bases agree within 2e-15. No regression on outcomes is involved.

I also checked against the two beta vectors computed from the existing cached panel. The
alternative base reconstructions agree within 7e-14; saved targets match the cache within
8e-15. The base-replacement MSE identity holds numerically within 3e-15 per meeting.

Two recording issues remain:

- CPU v3 manifests record CRSP hash `56a316e2b455`; GPU neural manifests record `76f7cd27e0c5`.
  The local cached panel records the former. Byte hashes can differ for logically equivalent
  data, so this alone does not establish a changed training sample. The matching outcomes and
  beta space support the calculations here, but do not prove every training input was identical.
  Resolve the provenance difference and save actual panel-array fingerprints before the next batch.
- `V3Arm.fit_result` holds every member's selected epoch, but the shared evaluator serializes
  only the first member's epoch/loss curves. The saved prediction tables also omit base and
  correction components. The in-memory fields do not yet satisfy the roadmap's recording requirement.

The signed-shock gate table is present. Its out-of-sample follow-up is described in the results
report via a scratchpad script; I did not independently reproduce that fitted follow-up.
