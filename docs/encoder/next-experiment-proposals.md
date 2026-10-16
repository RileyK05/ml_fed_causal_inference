# Q3: Two proposals for the next experiments

Date: 2026-10-14. Status: proposals for discussion, not an adopted specification or authorization to train.

## What the latest results show

The latest experiment pretrained the full response model on ordinary days with `s = 0`, then adapted it to Fed days. Ordinary-day training supplies no direct response-loss signal to the `b` head. Evaluation covered 60 held-out development meetings through 2009-03-18, with three seeds.

| Model | Mean per-meeting MSE (lower is better) | RankIC (higher is better) |
|---|---:|---:|
| Ordinary-day pretrained transformer | 8.538 | 0.087 |
| Transformer from scratch | 8.505 | 0.079 |
| Pooled baseline | 8.587 | — |

The ordinary-day pretrained model has slightly worse MSE and slightly better RankIC than scratch. This does not establish a pretraining improvement. The experiment sampled every tenth ordinary trading day, yielding 362 days across the full development span; each fold used only eligible earlier days. It does not settle the value of training on all ordinary days.

The saved comparison omits the paired interval for daily pretraining versus scratch. The pooled-minus-scratch MSE interval includes zero. Treat these results as exploratory, without claiming equivalence or a clear transformer advantage over pooled prediction.

Sources: [comparison table](../../encoder/results/q3_susceptibility/20261015-031300_daily-exp-comparison/tables/comparison.csv), [daily-pretraining manifest](../../encoder/results/q3_susceptibility/20261015-030832_daily-exp-tx-daily/manifest.json), and [implementation](../../encoder/fedenc/daily.py).

## Proposal 1: Regularize firm-specific shock sensitivity

**Question:** Can partial pooling make differences in firm sensitivity more reliable with relatively few independent policy events?

Keep the transformer and the raw Fed-day return target. Express the prediction as:

```text
predicted_return(i,t) = a(z_i,t) + [b0 + delta_b(z_i,t)] * shock_t
```

Here `b0` is a learned common sensitivity and `delta_b` is the firm's state-dependent deviation. Add a penalty on squared deviations, averaged equally across meetings and then firms within each meeting:

```text
loss = mean_per_meeting_MSE + lambda * mean_per_meeting(delta_b ** 2)
```

The penalty is what changes the model; writing `b(z) = b0 + delta_b(z)` alone does not restrict it. Both the baseline branch `a` and the sensitivity branch continue learning. Keep the deviation head's constant component out of the parameterization, or otherwise identify it separately from `b0`, so the common loading and the deviations are interpretable.

### Evidence

- In the latest experiment, scratch and daily pretraining are close to the pooled baseline on MSE.
- The earlier C1 ridge comparison scored 8.910 MSE with a common shock loading versus 9.688 with state-dependent loadings, across 48 development test meetings. Its paired interval for the increase was [0.099, 1.607].
- These results make estimation noise a plausible bottleneck. They do not prove that nonlinear sensitivity differences are absent.

Source: [C1 comparison](../../encoder/results/q3_susceptibility/20261013-223729_cheap-c1/tables/comparisons.csv).

### Experiment

Compare three versions using the same scratch transformer, inputs, chronological folds, seeds, and adaptation budget:

1. Common sensitivity: `a(z) + b0 * s`.
2. Current flexible sensitivity: `a(z) + b(z) * s`.
3. Penalized sensitivity deviations: `a(z) + [b0 + delta_b(z)] * s`.

Predeclare a small penalty-selection budget and choose its value using earlier inner validation only. Hold pretraining out of this comparison to isolate the sensitivity restriction.

### Decision

Promote the penalized model only if held-out development response prediction improves, supported by paired per-meeting comparisons and seed stability. Report RankIC, the dispersion of predicted sensitivities, and sensitivity diagnostics as secondary outcomes. A smaller spread of `b` is not itself success.

**Presentation sentence:** “We will test whether partial pooling lets the transformer learn reliable differences between firms from a limited number of policy events.”

## Proposal 2: Add past earnings reactions to firm state

**Question:** Does a firm's history of reacting to company news add information beyond ordinary price history and volatility?

Add the historical earnings-reaction statistic already used in R6 to the firm-state inputs, with an availability mask. Keep the raw Fed-day return as the prediction target.

### Evidence

R6 found an association between past earnings-reaction magnitude and the magnitude of a firm's Fed-day return relative to its meeting mean:

- Mean within-meeting RankIC: approximately 0.143.
- RankIC after residualizing the earnings statistic on trailing volatility: approximately 0.058.
- Coverage in that diagnostic: approximately 99.5%.

This is exploratory association within the development sample, not a demonstrated out-of-sample gain or evidence about the direction of Fed sensitivity.

Source: [R6 manifest](../../encoder/results/q3_susceptibility/20261013-224003_cheap-r6/manifest.json).

### Feature definition

Reuse the diagnostic's definition for the first comparison:

```text
earnings_reaction = mean absolute two-day market-adjusted return
                   over the firm's last four completed earnings events
```

Require at least two usable past events. Both days of each event window must finish before the prediction cutoff. Preserve missing observations and include an availability mask. Audit report-date alignment and point-in-time availability before fitting. This is a historical predictor; it does not change the response target or imply that earnings shocks and policy shocks are interchangeable.

### Experiment

Compare existing inputs against existing inputs plus this feature for:

1. A simple ridge interaction baseline.
2. The scratch transformer with the feature added to its firm-state branch.

Use identical chronological folds and seeds where applicable. Fit preprocessing only on each training block. Keep the architecture and other training choices fixed.

### Decision

Require improvement in held-out raw-return prediction to support this feature for the current Q3 task. If it predicts reaction magnitude but does not improve the conditional mean, report that distinction; it may motivate a separate uncertainty-modeling experiment, but it does not establish better `a` or `b` prediction.

**Presentation sentence:** “We will test whether a firm's history of reacting to company news helps predict its response to monetary-policy news.”

## Recommended order and evaluation rules

Run proposal 1 first, then proposal 2. Test them separately before combining them. They address two distinct explanations for current performance: noisy sensitivity estimates and missing event information.

- Keep mean per-meeting raw-return MSE as the primary selection metric.
- Use the existing development period for selection; preserve later outer evaluation and the separate 2026 holdout.
- Split by whole meetings and keep all fitting and preprocessing before the applicable validation or test boundary.
- Report paired meeting-level differences and seed variation. The existing independent-meeting bootstrap does not itself handle serial dependence across meetings.
- Do not treat the existing `b` slope diagnostic as direct identification of true sensitivity: baseline-return differences can affect it, and its current across-seed summary is not a combined inferential test.
- Record each fitted comparison as exploratory. Training requires explicit authorization in chat.

No training or new result generation was performed to prepare this brief.
