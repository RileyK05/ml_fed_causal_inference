# Big movers: implications and saved-prediction follow-up

2026-10-18. Follow-up to [big-movers-news.md](big-movers-news.md) at `e07deb3` and
[the v3 debrief](q3-v3-debrief_a.md). No fitting, new saved run or holdout evaluation.

**This changes the next-step order: audit the earnings contribution first, then test a small
pre-event peer-stress interaction before another pretraining experiment. The audit below
does not reveal a hidden overall model advantage, but the small large-surprise gain survives.**

## 1. The earnings concentration is reproducible

I matched the saved development predictions to cached firm identifiers and Compustat `rdq`.
The flag is an earnings report dated on the meeting or the preceding trading day, using the
Fama–French trading calendar. These reproduce the news document's rounded figures:

| Scored meetings | Flagged firm-meetings | Share of squared cross-firm CAPM error | Flagged share of top ten absolute cross-firm misses |
|---|---:|---:|---:|
| All 161 | 6.18% | 21.36% | 33.54% |
| 42 with surprise magnitude ≥ 2.5bp | 4.06% | 15.29% | 23.10% |
| 10 with surprise magnitude ≥ 5bp | 4.06% | 8.73% | 19.00% |

Here the error share is the flagged sum of squared within-meeting-centered residuals divided
by that sum for all firms and meetings in the slice. It is not the average of meeting-level
shares: that alternative is 22.65% over all meetings. Neither quantity is the primary raw-return
MSE. Keeping those definitions separate matters.

This is evidence of substantial error concentration around earnings. It does **not** show
that earnings cause every flagged residual, that all this loss is irreducible, or that no
pre-event feature could predict any of it. A calendar flag might help predict uncertainty;
it does not supply the direction of the eventual earnings surprise.

## 2. Excluding flagged rows does not uncover a hidden overall winner

I rescored the existing v3 predictions without changing or refitting them. For the neural
corrections, the comparison uses their own inner-training base, recovered as described in
the v3 debrief. Negative differences mean better prediction.

| All 161 meetings | Raw MSE difference versus own base, all rows | Difference after excluding flagged rows | 95% meeting-bootstrap interval after exclusion |
|---|---:|---:|---|
| A1 projected MLP | +0.000014 | +0.000092 | [-0.004007, +0.003850] |
| A2 centered transformer | -0.000299 | -0.000781 | [-0.006492, +0.005139] |
| A3 projected transformer | -0.000817 | -0.000775 | [-0.003056, +0.001476] |

A0's raw MSE falls from 5.9024 to 5.2333 when those rows are omitted. That is a different
evaluation population, not an improvement in the fitted model. R0, R1 and R2 also remain
worse than A0 in point estimates on the unflagged rows. The saved combination of A0 plus
A3's correction remains a tie: difference -0.000780, interval [-0.003061, +0.001471].

Thus earnings concentration is important, but it does not explain the overall null as a
predictive advantage that was merely hidden in the aggregate score. Rescoring does not settle
whether a different, prospectively specified training treatment would change the learned model.

## 3. The large-surprise gain largely survives the earnings diagnostic

For the same ten 5bp+ meetings, exclusion leaves 4,797 of 5,000 firm-meetings. The following
secondary scores recenter residuals within each meeting's retained firms:

| Arm | Cross-firm MSE difference, all rows | Difference after exclusion | 95% interval after exclusion | b × s-only difference after exclusion |
|---|---:|---:|---|---:|
| A1 projected MLP | -0.018367 | -0.017545 | [-0.040021, -0.000898] | -0.012352 |
| A3 projected transformer | -0.013811 | -0.012809 | [-0.027848, -0.001225] | -0.010939 |

The raw-return differences after exclusion are -0.018467 for A1 and -0.013352 for A3.
Raw and centered differences need not agree after filtering: projection was defined on the
original full cross-section, and no longer guarantees a zero correction mean on the subset.
I did not recompute the projection or use the full-universe orthogonality identity on that subset.

The two standout dates still account for most of the gain. After exclusion, A3's cross-firm
improvements are 0.06539 on 2004-01-28 and 0.04582 on 2011-09-21. Earnings-rich dates therefore
do not by themselves explain away the improvement. Other corporate news remains unflagged,
and neither this exclusion nor the b-head ablation isolates a causal monetary response.

All intervals use 20,000 iid meeting-bootstrap draws, seed 0. The thresholds, arms and
exclusions have been inspected retrospectively; these are exploratory, unadjusted intervals.
Ten meetings and a good sign across model seeds do not supply independent confirmation.

## 4. How the news stories change the model hypothesis

The original document's idea of **industry stress before the meeting** is more specific than
another static sector indicator. It proposes that the response changes with the group's
recent condition. A failure of fixed sector loadings or a market-wide regime input does not
fully test this hypothesis.

However, the narrative evidence has limits. Firm news was found for 33 of the 100 selected
extreme firm-events, while many searches were incomplete. That does not establish that most
misses are company news, or that an unfound story means a Fed-driven move. The markdown does
not include the per-event source links/timestamps needed to independently audit its news
attributions. Treat the industry narratives as candidate explanations, not training labels.

### The next small fitted comparison I would propose

Use one fixed, measurable state: a group's **trailing 63-trading-day market-relative return**,
ending at the previous close. Calculate the peer aggregate leave-one-firm-out, so adding the
group statistic is not merely reintroducing the same firm's own return. Keep availability
and coverage explicit; do not fill gaps or infer pre-event stress from the realized event return.

Compare three small regularized specifications on the same covered sample:

1. Two-beta base.
2. Base plus own-firm trailing relative return and static group controls, with both baseline
   terms and surprise interactions.
3. The same model plus peer trailing relative return and its surprise interaction.

This asks whether **peer state adds information beyond own state and group identity**. Keep
all-meeting raw-return MSE primary, with the fixed 2.5bp+ slice secondary. Report whether any
incremental gain survives outside the two standout dates; do not select those dates away.
Choose penalties using earlier inner validation and use chronological outer folds.

The present cached metadata has GICS **sector** and a source flag, not a fully audited fine
industry history. Start with genuinely as-of sectors, explicitly calling the test sector stress,
or obtain point-in-time industry classifications before making the finer industry claim.
Retrospectively backfilled classifications are not valid pre-event inputs. Use the same
eligible rows for all comparators and distinguish unknown groups from unstressed groups.

If this controlled comparison adds nothing, do not automatically escalate to cross-firm
attention. If it helps across development blocks, it supplies a concrete reason to investigate
cross-firm representations. Further rate-sensitivity, drawdown and credit inputs are separate
extensions, not an immediate grid.

## 5. Earnings flags: diagnostic now, prospective information only in a model

The existing `rdq` flag is retrospective and date-only. A previous-day release before that
day's close may already be reflected in the starting price; a meeting-day release after the
close may fall outside the target. It is an imperfect overlap proxy without release times.
There are also 428 development rows without a linked `gvkey`; they are retained as unflagged
by this diagnostic, not certified to have no earnings news.

Keep the full population as the primary score, consistent with
[the Q-spec](../question.md). An exclusion based on subsequently realized news is a labeled
secondary sensitivity analysis. An earnings-calendar input would need the schedule as it was
known by the previous close, including revisions, rather than today's actual report date.

Do not use "news found after the event" as a training-time clean/noisy label and then claim
the primary predictor could have made that distinction prospectively. Also do not switch
automatically to Huber, clipped returns or a ranking target: those alter the fitted objective
without establishing the conditional-mean improvement we need.

## Revised priority

The saved-prediction earnings check is now done. Next comes one small, controlled pre-event
sector/industry-stress comparison. Net floating-rate debt after hedging remains the larger
data investment. Market-loading-invariant pretraining remains an optional later experiment,
behind these more specific checks of what information the current inputs are missing.
