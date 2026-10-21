# Q3 v4 plan: firm shock sensitivity beyond beta, on the meetings that matter

Written 2026-10-18; revised the same day after review (evidence wording, a clean peer-stress
control, a matched joint-training control, the narrower primary claim). Merges [q3-v4-plan_g.md](q3-v4-plan_g.md), Astra's
[v3 debrief](q3-v3-debrief_a.md) and [big-movers follow-up](big-movers-followup_a.md),
[big-movers-news.md](big-movers-news.md), the [v3 results](q3-v3-results.md) and the
[2026-10-17 review](../reports/encoder-review-2026-10-17.md). Numbers below were re-verified
from saved predictions. Plan only: no run is authorized by this document.

## 1. The question

On a Fed day, most of a stock's move is beta times the market's move. That part is solved: a
five-number CAPM using both the 63-day and the one-year average beta (`capm_beta_both`) is the
best model we have. **v4 asks only: beyond beta, does a firm's sensitivity to the surprise (b)
differ in a predictable way, and does it show where it should, on large surprises?** The drift
(a) is not the target; on small-surprise meetings b x s is near zero and those meetings cannot
answer the question.

## 2. What we know going in

| Finding | Evidence |
|---|---|
| Beta is most of it | Every model through v3 ties or loses to `capm_beta_both` on all meetings |
| The networks' v2 "edge" was beta smoothing | It tracked the one-year beta; a 5-number CAPM reproduces about 70% |
| **The shock-sensitivity part of the v3 corrections gains as the surprise grows; the drift part shows no gain** | Prediction ablation of v3 A3 vs its own base, b x s part: -0.0012 (-0.0030, +0.0004) on all 161 meetings, -0.0025 on \|s\| >= 2.5bp, -0.0119 (-0.0237, -0.0027) on the 10 meetings at \|s\| >= 5bp; drift part +0.0003. A1 (MLP) is similar. Only the 10-meeting slice excludes zero |
| What the b correction predicts (a description, not a mechanism) | Within meetings, A3's b correction correlates with cash +0.39, size +0.23, past relative return +0.23, book-to-market -0.21, leverage -0.19, drawdown -0.18 to -0.20. If real, hawkish surprises favour cash-rich, low-debt, strong firms and cuts favour levered, beaten-down ones: a credit/distress reading, tested as H2 |
| The news agrees | Distressed power producers rallied on the Nov 2002 cut; homebuilders fell on the hawkish Jan 2004 statement; coal and levered financials fell in Sep 2011; insurers jumped on the Dec 2008 zero-rate cut |
| But it is concentrated | Two meetings (2004-01-28, 2011-09-21) carry about 90% of the 5bp gain; 9 of the 10 5bp meetings are 2001-2008 |
| Firm news is a large noise source | Earnings-day firms are 6% of rows but 21% of the firm-vs-firm error (34% of each meeting's 10 biggest misses); excluding them does not reveal a hidden winner, and the big-meeting b gain survives |
| Static trait or sector loadings fail out of sample | G1, G2 and G4 gate passes all lose out of sample; sector patterns flip between eras |

## 3. Hypotheses (fixed before any run)

- **H1 (primary).** A correction to b restricted outside the span of [1, beta63, beta_smooth]
  lowers firm-vs-firm error against `capm_beta_both` on meetings with \|s\| >= 2.5bp.
  This is a narrower claim than better Q3 prediction overall: it is about firm sensitivity on
  large surprises, the only meetings where b x s is big enough to measure. Overall Q3
  prediction is covered by the guardrail, not claimed.
- **H2.** A small, named set of balance-sheet and state variables (the distress channel) carries
  most of that gain, so a ridge model on them matches the networks.
- **H3.** The stress of a firm's sector going into the meeting (peer state) adds to H2 beyond
  the firm's own state and fixed sector effects.
- **H4.** Training with the drift correction switched off (b only) does better than training a
  and b jointly, with everything else the same. The v3 ablation removed `da` only from saved
  predictions, and v2's b-only failure used a different base, so neither settles this.

## 4. Design

**Base.** `capm_beta_both`. Two-stage fit (fixes the v3 handicap of fitting the base on 85%):
choose epochs or penalties on the first 85% of each training window with the base fit there
only, then refit the base on the full window and the correction for the selected number of
optimizer steps.

**Correction.** b only: `mu = mu_base + P_m(delta_b) * s`, with `P_m` the exact within-meeting
projection off [1, beta63, beta_smooth] and `delta_a = 0`. On a meeting with s = 0 the model is
the base exactly, so early stopping is decided only by meetings with a surprise. Readout
zero-initialized. One surprise series (STMT); no event-day stock move as an input.

**Arms.**

| Arm | Model | Inputs | Tests |
|---|---|---|---|
| B0 | `capm_beta_both` | betas | Reference |
| B1 | Ridge, b only, own penalty | summary features + 6 fundamentals | Is there linear b beyond beta? |
| B2 | Ridge, b only | cash, leverage, log size, book-to-market, 252-day relative return, 252-day drawdown | H2: the named distress channel |
| B3c | B2 + own state and sector | B2 + own 63-day market-relative return, sector effects (both x s) | Control for B3 |
| B3 | B3c + peer stress | B3c + leave-one-out sector trailing 63-day market-relative return (x s) | H3: peer stress alone, vs B3c |
| B4 | Summary MLP, b only, projected | as v3 A1 | Nonlinear, no sequence |
| B4j | Summary MLP, a and b, projected | same, same refit | H4 control for B4 |
| B5 | Transformer, b only, projected | as v3 A3 | Does sequence add to B4? |
| B5j | Transformer, a and b, projected | same, same refit | H4 control for B5 |

Deferred (each needs its own case): short-term debt share and net debt / OIBDA (close cousins
were null in the gates), market-loading-invariant VICReg pretraining, cross-firm attention,
direct floating-rate debt exposure (a data project, section 8).

## 5. Evaluation (fixed before any run)

Development only: the same walk-forward as v2/v3 (161 test meetings, S&P 500, raw day-0
returns). 2022-2025 is not used for choosing anything.

- **Primary:** firm-vs-firm MSE difference vs B0 on the 42 test meetings with \|s\| >= 2.5bp,
  meeting-bootstrap 95% range. Supported if the range excludes zero in the arm's favour.
- **Required alongside the primary:** the same with 2004-01-28 and 2011-09-21 removed. A gain
  that disappears without them is reported as concentrated, not supported.
- **Guardrail:** all-meeting raw MSE no worse than B0 by more than 0.005.
- **Comparisons:** each arm vs B0 (H1); B2 vs B1, B4, B5 (H2); B3 vs B3c (H3); B4 vs B4j and
  B5 vs B5j (H4); B5 vs B4 (sequence). For projected corrections on the full firm cross-section,
  firm-vs-firm and raw MSE differences coincide; what narrows the claim is the meeting subset.
- **Secondary:** \|s\| >= 1bp and >= 5bp slices; 2001-2008 vs 2009-2021; slope of B0's miss
  on each arm's b x s; earnings-day rows excluded (labelled sensitivity, not primary);
  correlation of each arm's b with the named distress variables.
- Thresholds 1, 2.5 and 5bp are the v1 diagnostic bins; the large-surprise focus was chosen
  after seeing v3, so every v4 result on development data is exploratory.

**Confirmation.** If a B arm is supported on development data, fix it and run it once on
2022-2025, which has 10 meetings at 2.5bp or more and 7 at 5bp or more. That window was used
by v1 and v2, so it is a labelled secondary check. The 2026 holdout stays closed.

**Stop rule.** If no arm meets the primary with the two standout meetings removed, stop
model work on Q3 day-0 b, write the null, and move to section 8.

## 6. Engineering prerequisites (before any run)

1. Two-stage fit with step-matched refit; a test that inner-validation labels never move the
   base, and that s = 0 meetings reproduce B0 to 1e-12.
2. Save `mu_base`, `delta_b`, s and every member's selected epoch with each prediction; record
   the panel fingerprint. Resolve why CPU and GPU v3 runs recorded different CRSP hashes.
3. Move the scoring scripts into a committed `python -m fedenc.v4 score` (slices, a/b
   decomposition, leave-two-out, era split, earnings sensitivity).
4. Point-in-time inputs only: sectors as of the meeting, `rdq` before the meeting, no filling.
5. Add per-event sources to big-movers-news.md so the news attributions can be audited.

## 7. Compute and order

| Step | What | Where | Cost |
|---|---|---|---|
| 1 | Prerequisites and tests | local | none |
| 2 | B0-B3 (with B3c) | local CPU | minutes |
| 3 | B4, B4j, B5, B5j (3 seeds x 3 members) | 2-4 vast GPUs, about 70 min | about $1.50 |
| 4 | Score, write up | local | none |
| 5 | If supported: one 2022-2025 run of the fixed arm | 1 GPU | under $1 |

## 8. Parallel tracks (not part of the v4 run)

- **Q4 reversal.** Do the big day-0 misses reverse over the next 7, 14 or 30 days, more than
  same-size misses on matched ordinary days? Uses saved gaps and CRSP prices; CPU only.
- **Floating-rate debt exposure.** The most direct measure of the distress channel. First check
  whether the WRDS subscription includes Capital IQ Capital Structure (fixed vs floating debt);
  otherwise a 50-filing feasibility pilot, as in Astra's roadmap.
- **Unscheduled meetings (decision D1).** 18 emergency actions, among the largest surprises.
  Adding them changes the spec and needs the user's call.

## 9. Decisions for the user

| ID | Decision | Default |
|---|---|---|
| D1 | Include unscheduled meetings | No (keeps the spec) |
| D2 | Primary threshold 2.5bp | Yes |
| D3 | Confirm on 2022-2025 if supported | Yes, once, labelled secondary |
| D4 | Check Capital IQ access | User to check |
