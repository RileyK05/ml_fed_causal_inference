# Q3 v2 results: gates, two-shock baselines, and the modular ablation

Run 2026-10-17 on branch `q3-v2`, development meetings only (through 2021-12-31). Plan:
[q3-v2-plan.md](q3-v2-plan.md). Two tracks ran side by side, as agreed:

- **Track 1 (plan, CPU):** two-shock CAPM baselines, descriptive gates G1-G3, then an
  out-of-sample check of whatever passed.
- **Track 2 (the other model's plan, GPU):** the 8-arm modular ablation in `fedenc.v2`, scored
  against `capm_two_shock` as well as `capm_pooled` and `pooled`.

Rules fixed before results: track 2 runs are `exploratory`; the gates are the pre-registered
test; an arm that beats the two-shock CAPM while every gate fails is a lead, not a finding.

## Code changes

- `fedcore.q3.shocks`: the policy / information split (`s * SP500 > 0` in the USMPD
  announcement window is an information meeting) and the slope split (`10 * (UST10Y - UST2Y)`).
  37% of development meetings are information meetings with the panel's surprise (STMT); the 23%
  quoted in the plan used MP1. The 14 meetings without an S&P 500 window move (all 1994-95) count
  as policy meetings, stated in the module; they only ever sit in training folds.
- `capm_two_shock` and `capm_target_slope` in `fedcore.q3.baselines` (beta-scaled loadings by OLS).
- `fedenc.v2` fixes: the target/slope arm now keeps `mu = a + b*s` (the slope response is part of
  `a`); the silent "policy" default is now the stated rule above.
- `fedenc.gates`: G1-G3. `SectorTwoShock` in `fedenc.v2`: the out-of-sample check for G1.

## Track 1: gates (top 1,500, 221 meetings, 331,464 firm-meetings)

Bar: some trait (within sector, or sector itself) at p < 0.01 in **both** 1994-2008 and
2009-2021, same window and term. Saved run: `encoder/results/q3_susceptibility/*_v2-gates`.

| Gate | Tests | p < 0.01 | Chance | Result |
|---|---|---|---|---|
| G1 policy / info / slope | 306 | 34 | 3.1 | **Pass, sector only**: sector x s_policy and sector x s_info, day 0 and days 0-2 |
| G2 regime | 468 | 45 | 4.7 | **Pass, thin**: rate beta x s x drawdown, days 0-2 only (p 0.0009 / 0.0004) |
| G3 magnitude on big days | 98 | 3 | 1.0 | **Fail** (as in v1) |

Reading:

- **Sector responses differ by shock type, but flip between eras.** Sector x s_info is
  p 0.008 (1994-2008) and 0.003 (2009-2021) but 0.13 over the full period: the pattern reverses.
- **The slope change matters only after 2009** (sector x slope p 0.76 before, < 0.0001 after),
  consistent with forward guidance moving the curve, not the short rate. It fails the
  both-eras bar by design.
- **The zero-lower-bound regime cannot be tested before 2009** (one meeting), so its strong
  full-period results (p < 0.001) are one era only, and their strongest groups are mostly the
  "missing data" groups, a composition warning.
- Caveat: clustered tests over-reject when a cell has few meetings (Real Estate exists from 2016).
  Groups need 10 meetings where the tested term is nonzero; rank-deficient tests return NaN.

## Track 1: out of sample (S&P 500, walk-forward, 161 test meetings)

Same folds as v1 (`min_train=60, test_size=16`). Per-meeting MSE; ranges are meeting-bootstrap 95%.

| Model | MSE | vs `capm_two_shock` | Cross-firm part vs `capm_two_shock` |
|---|---|---|---|
| `capm_pooled` | 5.914 | -0.073 (-0.347, +0.137) | -0.023 (-0.073, +0.013) |
| `pooled` | 5.928 | -0.058 | -0.002 |
| `capm_target_slope` | 5.961 | -0.025 | -0.026 |
| `capm_two_shock` | 5.987 | | |
| `sector_two_shock` | 5.992 | +0.005 (-0.036, +0.039) | **+0.033 (+0.008, +0.060)** |
| `sector_two_shock_slope` | 6.029 | +0.042 | **+0.054 (+0.028, +0.080)** |

- **The split does not help prediction.** The two-shock CAPM is no better than the one-shock
  CAPM, and its b grade is worse (0.19 vs 0.99; 1 is calibrated).
- **Sector loadings hurt out of sample.** Fitted on past meetings they point the wrong way on
  future ones, the era flip above. Same as v1, where sector "hurt as a model input".
- Leakage note: the information flag uses the S&P 500's move in the announcement window, which
  is part of the day-0 return. It can only help these models, and they still do not gain.

## Track 1: out of sample for the G2 pass (top 1,500, 161 test meetings)

Cross-firm part of the market-adjusted return; null = nothing beyond beta (predict 0). OLS on
rate-beta terms fit on the training folds (z-scored, clipped at the training 1st/99th
percentiles). Positive = the model is **worse** than the null. Script: scratchpad `g2_oos.py`.

| Window | Terms | All test meetings | 32 drawdown meetings |
|---|---|---|---|
| days 0-2 | rate beta, x s, x drawdown, x s x drawdown | +0.027 (+0.001, +0.062) | +0.093 (-0.018, +0.246) |
| days 0-2 | rate beta x s x drawdown only | +0.015 (-0.006, +0.048) | +0.077 (-0.025, +0.226) |
| day 0 | full set | +0.008 (+0.002, +0.015) | +0.032 (+0.011, +0.058) |

The G2 pass does not predict either.

**Track 1 verdict:** every gate pass is an in-sample pattern that does not carry to the next
meetings. Under the plan's rules no network is warranted from the gates; the track 2 arms are
the only test of whether a network finds something nonlinear the linear checks miss.

## Track 2: modular ablation (4 vast RTX 3090 boxes, about $2)

S&P 500, the same 161 walk-forward test meetings, 3 seeds x 3 members per arm. Each arm adds one
switch to the one before it (except `v2_target_slope`, which swaps the split). Ranges are
meeting-bootstrap 95%. Combined run: `*_v2-ablation-combined-diagnostics`.

| Arm | What it adds | MSE | vs `capm_pooled` | Cross-firm vs `capm_two_shock` | b grade |
|---|---|---|---|---|---|
| `capm_pooled` | one-shock CAPM | 5.914 | | -0.023 | 0.99 |
| `capm_two_shock` | two-shock CAPM | 5.987 | +0.073 (-0.137, +0.347) | | 0.19 |
| `v1_two_part` | v1 model, pooled base | 5.905 | -0.009 (-0.043, +0.029) | -0.025 | 2.75 |
| `v2_capm_base` | network on the CAPM residual | 5.904 | -0.010 (-0.023, +0.002) | -0.034 | 0.76 |
| `v2_macro_hist` | + yield, HML, SMB history; rate and value betas | 5.907 | -0.007 (-0.022, +0.009) | -0.030 | 0.94 |
| `v2_regime` | + regime inputs, regime branch on b | 5.910 | -0.004 (-0.018, +0.011) | -0.027 | 0.92 |
| `v2_policy_info` | + policy / information b heads | 5.903 | -0.011 (-0.025, +0.002) | -0.035 | 0.78 |
| `v2_target_slope` | slope split instead | 5.894 | -0.020 (-0.070, +0.013) | -0.043 | 0.86 |
| `v2_shock_weighted` | + big-surprise weighting (alpha 2, cap 5) | 5.898 | -0.015 (-0.035, +0.003) | -0.039 | 0.76 |
| `v2_b_only` | + network predicts b only | 5.957 | +0.043 (-0.013, +0.132) | +0.020 | 0.10 |

Large-surprise meetings (\|s\| >= 5bp, 10 test meetings): every arm's cross-firm error is within
about 0.4 of `capm_two_shock`, with ranges about +-1. The test cannot tell them apart.

Reading:

- **No arm beats CAPM.** Every range includes zero. The best, `v2_target_slope` and
  `v2_shock_weighted`, gain 0.015-0.020 on 5.9 (0.3%), with 7 arms tried.
- **The CAPM-base arms lean the same way** (all six slightly below `capm_pooled`). That is the
  most a network adds here: a consistent sliver, not a finding.
- **b-only is worse.** Without its own a, the network's b spreads too much (b grade 0.10).
- The b grades of the CAPM-base arms (0.76-0.94) sit near the CAPM's 0.99: b is still beta.

## Verdict

Both tracks agree. Splitting the surprise by meaning, showing the model the regime, and weighting
the big meetings do not produce firm-level sensitivity beyond beta that predicts the next
meetings. The descriptive patterns are real in-sample (sector by shock type, the slope after
2009, rate beta in drawdowns), but they change between eras, so models fitted on the past miss
the future. v1's conclusion stands with a stronger null behind it: at the one-day horizon, b is
beta.

Not tried here: FRED regime series (VIX, credit spreads; FRED blocked scripted downloads from
both this machine and vast), statement text, unscheduled meetings (decision D1), and horizons
beyond 2 days for the network.

## Where the networks disagree with CAPM (development, out of sample)

Added 2026-10-17 at the user's request: not the headline MSE, but where the predictions differ.
Script: scratchpad `disagree.py`; d = network minus CAPM within each meeting.

- **Small but right.** The six CAPM-base networks deviate from CAPM by 0.08-0.11% per stock (CAPM's
  miss: 2.0%). When they deviate, the actual return goes their way, by more than they say: the
  slope of CAPM's miss on the six-network consensus d is 1.56 (mean per-meeting 2.10, t 2.4).
  Top decile of |d|: 56% right direction, 0.28% realized vs 0.22% predicted. The MSE gain matches
  var(d) x (2 x slope - 1) = 0.015.
- **Systematic.** The six agree with each other (correlations 0.6-0.9); `v2_b_only` does not (-0.2).
- **Drift, not sensitivity.** All of d is in `a`; the `b x s` part is 3% of its variance.
- **Where.** Top 10 of 161 meetings hold 42% of it (all 2001-03); high-beta, high-volatility and
  Info Tech names hold about half. It still holds after 2004 (t 2.2) and 2009-2021 (t 2.1).
- **What it tracks.** The firm's 1-year average beta (correlation 0.58; 0.94 in 2001-03) more than
  the 63-day beta CAPM uses (0.43).
- **Big movers: nothing.** The top 1% of CAPM misses (mean 7.1%, 6.3x own volatility): every
  network leans the right way 48-52% of the time and calls 0% of the miss.

### Test 1: does a simple model reproduce it?

| Model | MSE vs `capm_pooled` |
|---|---|
| beta x context tilt, OLS / ridge / centered ridge | +1.23 / +1.13 / +0.14 (overfits; tilt points the wrong way) |
| CAPM with the 1-year average beta (`CAPMBetas(252)`) | -0.0105 (-0.030, +0.008) |
| CAPM with both betas (`CAPMBetas(252, both=True)`) | -0.0115 (-0.029, +0.002) |
| six-network consensus | -0.0155 (-0.033, +0.000) |

A steadier beta gets about 70% of the consensus gain with five numbers. The networks keep a small,
different signal: their disagreement with the both-betas CAPM still predicts its miss (slope 1.12,
mean per-meeting 1.32, t 2.2; correlation of the two tilts 0.14).

## Pre-registered 2022-2025 check

Written and committed before any 2022-2025 prediction is made. 2022-2025 was used once for v1's
locked model and never for v2 (plan option A: a labelled secondary check, weaker evidence).

- **Data:** S&P 500 panel. Train on all 221 meetings through 2021-12-15; test the 32 meetings
  2022-01-26 to 2025-12-10. One fold, raw returns, the ablation's configs, fresh seeds 6, 7, 8,
  3 members. Runner: `python -m fedenc.v2 final`.
- **Models:** the six CAPM-base arms (`v2_capm_base`, `v2_macro_hist`, `v2_regime`,
  `v2_policy_info`, `v2_target_slope`, `v2_shock_weighted`); `capm_pooled`; `capm_beta_both`.
  Consensus = the mean of the six arms' mu.
- **P1 (primary):** CAPM's within-meeting miss on the consensus d (consensus minus `capm_pooled`,
  within meeting): mean per-meeting slope > 0, one-sided meeting-bootstrap p < 0.05.
- **P2:** consensus per-meeting MSE below `capm_pooled` (one-sided meeting-bootstrap p < 0.05).
- **P3:** `capm_beta_both`'s miss on (consensus minus `capm_beta_both`): mean per-meeting slope > 0,
  one-sided p < 0.05.
- **Also reported, no claim:** each arm alone, `capm_beta_both` vs `capm_pooled`, the 2022 hiking
  meetings separately.
- **Power:** at the development effect size (t 2.4 on 161 meetings), 32 meetings give an expected
  t of about 1.1. A null is likely even if the effect is real, and would not refute it; a correct
  sign is weak support. Everything is reported whatever the outcome.

### Result (run 2026-10-17, after the pre-registration commit 9d7603c)

32 test meetings, 16,101 firm-meetings. One-sided meeting-bootstrap p-values; 90% ranges.

| Claim | Estimate | t | p | Verdict |
|---|---|---|---|---|
| P1: consensus d predicts CAPM's miss (slope > 0) | +1.73 (-0.20, +3.82) | 1.39 | 0.074 | Not significant; same sign and size as development (1.56 / 2.10) |
| P2: consensus MSE below `capm_pooled` | +0.0026 (-0.014, +0.020) | 0.24 | 0.41 | Not significant |
| P3: consensus adds beyond `capm_beta_both` | -0.17 (-1.62, +1.40) | -0.18 | 0.59 | No |

Reported without a claim: per-meeting MSE `capm_pooled` 5.725, consensus 5.722, `capm_beta_both`
5.707 (gain +0.018, p 0.16). Single arms' slopes range from +0.08 to +1.85; three reach p < 0.05
alone (`v2_capm_base`, `v2_macro_hist`, `v2_shock_weighted`), which with six arms tried is not a
claim. Hiking-cycle meetings (2022-01 to 2023-07): slope -0.01. Top decile of |d|: 52% right
direction, 0.15% realized vs 0.17% predicted.

**Reading.** The drift tilt kept its sign and size on unseen meetings but, as the power note
expected, 32 meetings cannot confirm it (p 0.07). The part that would make it a transformer
result failed: out of sample the networks add nothing beyond a CAPM with a steadier beta, which
itself did slightly better than the networks (not significant). The defensible claim is the
beta-estimation one: on Fed days, a one-year average beta beats the last 63-day beta.
