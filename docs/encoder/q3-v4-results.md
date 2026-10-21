# Q3 v4 results

Run 2026-10-18 under [q3-v4-plan.md](q3-v4-plan.md) (revised the same day), user go-ahead in
chat. Code: `fedenc.v4` (uncommitted at the time of the run; GPU boxes received the working tree,
not a commit). Scoring: `python -m fedenc.v4 score`, saved as `*_v4-to2021-score`. Development
only: 161 walk-forward test meetings, S&P 500, raw day-0 returns. 2022-2025 was not used.

All runs share panel fingerprint `2f84a4acdf8968b9` (local and vast). The differing CRSP dataset
hashes in v3 manifests come from the vast boxes holding price files through 2021 only, not from
different training data.

## Verdict

**No arm meets the pre-stated primary.** On the 42 meetings with |s| >= 2.5bp, every correction
ties the two-beta CAPM (B0); with 2004-01-28 and 2011-09-21 removed, every point estimate turns
slightly worse. Per the stop rule there is no 2022-2025 confirmation run, and model work on Q3
day-0 b stops here.

## Primary and guardrail (firm-vs-firm MSE vs B0; negative = better)

| Arm | \|s\| >= 2.5bp (42) | Without the two standout dates (40) | All meetings, raw MSE | Supported |
|---|---|---|---|---|
| B1 ridge, all features | +0.0081 (-0.0016, +0.0199) | +0.0086 | +0.0016 | no |
| B2 ridge, six named variables | -0.0010 (-0.0102, +0.0078) | +0.0029 | -0.0004 | no |
| B3c B2 + own 63-day return + sector | -0.0030 (-0.0156, +0.0089) | +0.0016 | -0.0003 | no |
| B3 B3c + peer stress | -0.0004 (-0.0114, +0.0105) | +0.0028 | +0.0000 | no |
| B4 MLP, b only | -0.0001 (-0.0114, +0.0108) | +0.0053 | -0.0009 | no |
| B4j MLP, joint | -0.0002 (-0.0070, +0.0057) | +0.0029 | +0.0006 | no |
| B5 transformer, b only | -0.0018 (-0.0133, +0.0087) | +0.0027 | -0.0011 | no |
| B5j transformer, joint | -0.0024 (-0.0132, +0.0067) | +0.0031 | -0.0009 | no |

Every arm passes the guardrail (all-meeting raw MSE within 0.005 of B0); B0 scores 5.9024 and the
best arm, B5, 5.9013.

## Pre-stated comparisons (\|s\| >= 2.5bp)

| Hypothesis | Comparison | Result |
|---|---|---|
| H2: the named distress variables carry the gain | B2 - B1 | -0.0091 (-0.0196, -0.0004): the six named variables beat the full feature set |
| | B2 - B4, B2 - B5 | -0.0009, +0.0008: ridge on six variables matches both networks |
| H3: peer stress adds to own state and sector | B3 - B3c | +0.0026 (-0.0011, +0.0070): no |
| H4: b-only training beats joint training | B4 - B4j, B5 - B5j | +0.0001, +0.0006: no |
| Sequence over no-sequence | B5 - B4 | -0.0017 (-0.0071, +0.0038): no |

## Secondary

- **10 meetings at 5bp or more:** B3c -0.040 (-0.073, -0.009), B3 -0.031 (-0.055, -0.007),
  B4 -0.023 (-0.054, -0.001), B4j -0.014 (-0.033, -0.001); B5 and B5j -0.022 and -0.021 with
  ranges just including zero. These ten include both standout dates, which the primary shows the
  gains depend on.
- **Era (2.5bp slice):** only the transformers lean the right way in both 2001-2008 and
  2009-2021 (B5 -0.000 and -0.006, B5j -0.002 and -0.004); the rest are mixed. All are small.
- **Earnings-day firms excluded:** no change in conclusions (all ranges include zero).
- **What b points at is the same everywhere:** within-meeting correlations of every arm's b
  correction with cash +0.23 to +0.51, size +0.15 to +0.29, past relative return +0.17 to +0.35,
  book-to-market -0.18 to -0.33, leverage -0.10 to -0.22, drawdown -0.09 to -0.18. Ridge and
  networks agree on the direction of the tilt; it does not predict better on the meetings tested.

## Reading

- With the v3 problems fixed (base refit on the full window, drift switched off, projection off
  both betas, matched controls), firm shock sensitivity beyond beta does not improve prediction
  on large-surprise meetings in 1994-2021. The v3 5bp lead was the two standout meetings.
- The tilt the models agree on (cash-rich, large, low-debt, strong firms do better on hawkish
  surprises) is a description of the fitted b, not a demonstrated effect.
- Six named balance-sheet variables in a ridge model do as well as either network, and better
  than the full feature set: if this tilt is ever confirmed, it does not need a transformer.

## What remains open (plan section 8)

- An intraday target (30-60 minutes around the statement, needs TAQ) would remove most of the
  firm news that dominates the day-0 error.
- Firm-level shrinkage of b on each firm's own history of surprises (empirical Bayes) has not
  been tried.
- Q4 reversal of the big misses; direct floating-rate debt exposure (Capital IQ); unscheduled
  meetings (D1).

## Cost

Four RTX 3090s on vast (B4, B4j about 25-35 minutes; B5, B5j about 2 hours), about $2; all
destroyed after their runs were downloaded.

## Conditional view: where the corrections disagree with beta (added 2026-10-18)

Averages bury the cases that matter, so this asks where each model departs from the two-beta
base, on the 42 meetings with |s| >= 2.5bp. Scratchpad `conditional.py`, `vuln_check.py`.
Exploratory: the "vulnerable" group was defined from the distress tilt seen earlier on the same
data.

- **By size of departure.** In the top decile of departures, the actual move goes the model's way
  by more than the model says (0.13-0.18 points realized vs 0.10-0.12 predicted), but the
  direction is right only 51-52% of the time: when the models are right, it is on the big moves.
  The squared-error change in that decile is not distinguishable from zero for any arm.
- **Beta's worst misses.** On the worst 10% and worst 1% of beta's misses, no model cuts the
  error measurably (all ranges include zero; helps on 49-57% of cases).
- **Vulnerable firms (bottom third by cash and top third by leverage within the meeting).** When
  a model departs from beta by more than 0.1 point on these firms, it is almost always predicting
  underperformance on hawkish surprises (76-98% of departures). Meeting-clustered hit rates:

| Arm | All 42 meetings | Without 2004-01-28 and 2011-09-21 |
|---|---|---|
| B5 transformer, b only | **59% (52%, 67%)**, 310 firm-meetings in 26 meetings | **54% (51%, 57%)** |
| B5j transformer, joint | 61% (51%, 70%) | 53% (48%, 59%) |
| B4 MLP, b only | 60% (49%, 69%) | 54% (43%, 61%) |
| B3c ridge, distress + sector | 56% (49%, 62%) | 54% (47%, 61%) |

  The squared-error cut on these firms is positive with all meetings and turns slightly negative
  without the two standout dates (all ranges include zero).

**Reading.** The beyond-beta corrections carry a directional signal concentrated where the
distress story says it should be: on levered, low-cash firms, they lean the right way more often
than chance, including (for B5, narrowly) without the two standout meetings. The size of what
they predict is not right, so squared error does not improve. A test that scores direction or
ranking on flagged firms (a long-short on the model's departures on big meetings, a Q5-style
check) is the natural next step for this signal; squared-error scoring cannot see it.
