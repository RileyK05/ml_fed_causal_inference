# Q3 v3 results: two-beta base plus restricted corrections

Plan sources: [q3-v3-roadmap_a.md](q3-v3-roadmap_a.md) (A0-A3, recency, ranks),
[q3-v3-roadmap_g.md](q3-v3-roadmap_g.md) (signed shocks), and the
[2026-10-17 review](../reports/encoder-review-2026-10-17.md) (code fixes). Code: `fedenc.v3`,
`fedenc.gates --signed`. Training authorized by the user in chat on 2026-10-17.

## Pre-stated criteria

Written and committed before any v3 result exists. Development meetings only (through
2021-12-31). The same walk-forward as v2 (`min_train=60, test_size=16`, 161 test meetings,
S&P 500, raw day-0 returns, the panel's STMT surprise). 2022-2025 is not used. Development data
has been explored heavily, so every result here is exploratory.

**Arms.**

| Arm | Base | Correction |
|---|---|---|
| A0 | `capm_beta_both` | none |
| A1 | same (fit on the inner-training block) | summary-feature MLP, projected off span[1, beta63, beta_smooth] |
| A2 | same | transformer, centered within meeting only |
| A3 | same | transformer, projected off span[1, beta63, beta_smooth] |
| R0 | `capm_beta_both` | ridge on firm features and features x s, centered; penalty from inner validation |
| R1 | same, recency-weighted | R0 with a 5-year half-life on meetings (base and ridge) |
| R2 | same | R0 plus within-meeting percentile ranks of every feature |
| `capm_pooled` | | continuity |

Networks: width 64, 3 blocks, 5-day patches, 3 seeds x 3 members, lr 1e-3, weight decay 1e-4,
up to 40 epochs, patience 10, zero-initialized readouts, loss and early stopping on the deployed
base-plus-correction prediction.

**Questions and what counts.** Per-meeting MSE differences with meeting-bootstrap 95% ranges.

1. Does a correction beyond the measured-beta span help? A3 vs A0. Supported if the range
   excludes zero in A3's favour.
2. Is sequence modeling needed? A3 vs A1, same projection.
3. Does the projection matter? A3 vs A2.
4. Does recency weighting help? R1 vs R0. Do ranks help? R2 vs R0.
5. Is any gain one crisis block? Every gain is also reported for 2001-2003 and 2004-2021
   separately; a gain only in 2001-2003 is not counted as support.
6. G4: do firm groups respond differently to hawkish (s > 0) and dovish (s < 0) surprises?
   Same bar as G1-G3 (within sector, p < 0.01 in both 1994-2008 and 2009-2021). A pass would
   still need an out-of-sample check.

Also reported: cross-firm MSE, the disagreement slope (CAPM's miss on each arm's deviation from
A0), and the members' selected epochs. A tie with A0 is a reportable outcome, not a failure to
fix. No arm is added, retuned or rerun on the apparent winner.

## Results (run 2026-10-17/18)

161 development test meetings; per-meeting MSE differences with meeting-bootstrap 95% ranges
(negative = first arm better). A1-A3: 3 seeds x 3 members on vast; the rest on CPU. Scoring
script: scratchpad `score_v3.py`.

| Arm | MSE | Cross-firm MSE |
|---|---|---|
| A0 two-beta CAPM | **5.9024** | 4.1211 |
| R0 ridge correction | 5.9068 | 4.1255 |
| R2 + within-meeting ranks | 5.9111 | 4.1298 |
| `capm_pooled` | 5.9139 | 4.1259 |
| R1 + 5-year recency | 5.9167 | 4.1426 |
| A3 transformer, projected | 5.9322 | 4.1170 |
| A2 transformer, centered | 5.9327 | 4.1176 |
| A1 summary MLP, projected | 5.9330 | 4.1179 |

**Pre-stated questions.**

| Question | Comparison | All meetings | 2004-2021 | Answer |
|---|---|---|---|---|
| Q1 beyond the beta span | A3 - A0 | +0.030 (-0.053, +0.135) | +0.044 | No |
| Q2 sequence needed | A3 - A1 | -0.001 (-0.005, +0.004) | -0.001 | No difference |
| Q3 projection matters | A3 - A2 | -0.001 (-0.006, +0.005) | -0.000 | No difference |
| Q4 recency | R1 - R0 | +0.010 (-0.103, +0.098) | +0.003 | No |
| Q4 ranks | R2 - R0 | +0.004 (-0.003, +0.011) | **+0.008 (+0.002, +0.013)** | No; slightly worse after 2004 |
| Q6 G4 signed shocks | gate | 2 of 36 candidates pass (sector raw x s_hawk and x s_dove, days 0-2) | | Pass in-sample, **fails out of sample** |

**The A-arms' gap to A0 is the base, not the correction.** Review fix 1 fits the deployed base on
the inner-training block (85% of each training window), declared with no refit. Refitting
`capm_beta_both` that way alone (`A0_inner`, diagnostic, no network) scores 5.9330: the 85% base
costs +0.031 by itself. Against their own base the corrections add nothing:

| Correction | vs own base | Correction sd | Slope of own base's miss on it |
|---|---|---|---|
| A1 | +0.0000 (-0.0041, +0.0038) | 0.044 | +0.50 (t +0.5 per meeting) |
| A2 | -0.0003 (-0.0058, +0.0054) | 0.052 | +0.56 (t -0.2) |
| A3 | -0.0008 (-0.0031, +0.0015) | 0.036 | +0.82 (t -0.5) |

Selected epochs (first member per fold and seed): A1 mean 3.9 (often 0, the zero correction),
A2 8.1, A3 8.2.

**G4 out of sample** (top 1,500, sector intercepts and sector x s_hawk / x s_dove fit on
training folds; null = nothing beyond beta; positive = worse): days 0-2 +0.043
(-0.012, +0.111), day 0 +0.008 (-0.005, +0.022). Scratchpad `g4_oos.py`.

## Reading

- **With the two-beta base, the networks have nothing left to add.** v2's CAPM-base networks
  deviated from CAPM by 0.085 per stock, with slope 1.5. Here, on top of the two-beta base, the
  corrections are half that size (0.036-0.052) and have no skill (slopes 0.5-0.8, t < 1). The
  v2 edge was the beta smoothing; once the base has it, the transformer, the MLP and the
  projection all learn the same nothing.
- **Recency weighting, within-meeting ranks and signed shocks do not help.** Ranks are slightly
  but measurably worse after 2004. The signed-shock sector pattern is one more in-sample sector
  pattern that does not predict.
- **A cost of the review's fix 1.** Keeping the base on 85% of the window throws away the most
  recent 15% of meetings, and on these data that costs more (0.031) than any correction gains.
  Future arms should select epochs with the inner split, then refit the base (and, at the
  selected epoch count, the correction) on the full training window, as the review allowed.
- **Best model in v3: A0, the five-number two-beta CAPM.** It is the reference to beat.
