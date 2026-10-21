# Q3 final test: the summary MLP (F0s) on 2022-2025

DRAFT for the user's review. Written 2026-10-20, before any 2022-2025 data is scored. Nothing in
it may change after the run.

## Why this arm, and what it is not

- **Why F0s:** it is the steadiest arm across v4 and v6. On development meetings with |s| >= 5bp
  it ranks firms beyond beta by about +100bp per 10bp (v4 B4 +100, v6 F0s +97; intervals exclude
  zero).
- **It did not pass** the development pass bar (headline slice +48, interval includes zero).
  This run is a confirmatory test of the large-surprise result. It is not the confirmation of a
  passed arm.
- **The holdout gets used up.** 2022-2025 is used once, for this arm only. After it, no tuning
  or second arm is scored on 2022-2025. The 2026 yfinance holdout (8 meetings) stays separate.

## Setup

- **Model:** `fedenc.v6` arm F0s exactly as in the development run:
  - summary MLP, b-only, projected off the beta span;
  - two-stage step-matched refit;
  - model selection on big-surprise validation meetings;
  - 3 members.
- **Seeds:** 6, 7 and 8 (fresh, as v4's final).
- **Fold:** one fold. Train on every scheduled meeting 1994-2021 (inner validation = the last
  15% of those), then test on the 32 scheduled meetings of 2022-2025. B0 (two-beta CAPM) runs on
  the same fold.
- **Where:** locally on CPU (one fold, small model), so no rental. Estimated 15-40 minutes.

## Pass bar (fixed before the run)

Primary: the beyond-beta spread (`fedcore.q3.beyond_beta`) on the 10 test meetings with
|s| >= 2.5bp, meeting-bootstrap 95% interval.

| Outcome | Verdict |
|---|---|
| Spread > 0 and lower bound > 0 | **Replicates**: the model ranks firms beyond beta on big surprises out of sample |
| Spread > 0, interval includes 0 | **Consistent, not confirmed** |
| Spread <= 0 | **Does not replicate**: the development result was period-specific |

**Second pre-stated measure: post-meeting drift.** This is the user's core question: do the
model's calls come true in the following weeks? (`fedcore.q3.drift`; development evidence in
`q3-v6-drift.md`.)
- Measure: the 20-day drift spread on the same 10 meetings, meaning CAR over days 1-20, ranked by
  the model's call, in bp per 10bp. The 2025-12-10 meeting's window runs past CRSP's end, so that
  meeting drops out.
- **Drift replicates:** spread > 0, bootstrap lower bound > 0, and the sign-flip placebo
  p < 0.05.
- **Consistent, not confirmed:** spread > 0 otherwise.
- **Does not replicate:** spread <= 0.

**Expectation written before the run.** On development data the model's call is a credit tilt:
distressed, levered, value firms come out more sensitive, cash-rich tech less. Its payoff
concentrates in 2001-2003 and 2008 easing meetings. 2022-2025 is a hiking cycle without a
credit crisis, where long-duration growth stocks were hit hardest, the opposite of the tilt. We
expect "does not replicate" or "consistent, not confirmed" on day 0. The drift is the less
predictable of the two.

- **Guardrail:** all-meeting MSE vs B0 at most +0.005. If it fails, the verdict reads "fails the
  guardrail" whatever the spread.
- **Reported alongside, not deciding:** the 7 meetings with |s| >= 5bp, rank IC, calibration,
  claimed spread, and the top-decile hit rate.
- **Trading results are not scored here.** The trading project (`docs/trading/HANDOFF.md`)
  freezes its v1 books before this run and scores them once on the F0s predictions this run
  saves.
