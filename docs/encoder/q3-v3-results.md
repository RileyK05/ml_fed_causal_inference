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

## Results

_Pending._
