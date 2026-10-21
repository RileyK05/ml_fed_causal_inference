# Q3 v5 Track A: which channels make a firm Fed-sensitive beyond beta

Run 2026-10-18, exploratory, development data only (221 scheduled meetings 1994-2021, 110,753
S&P 500 firm-meetings; 2022-2025 untouched). Channels from `fedcore.q3.channels` (built from data
on disk and free sources; plan section 2). Scripts: scratchpad `track_a.py`,
`track_a_single.py`; nothing saved to a ResultStore.

Unit: change in the firm's day-0 return (percentage points) per 1 within-meeting standard
deviation of the channel, per 10bp hawkish surprise. Positive = firms high on the channel do
better (fall less) when the Fed surprises hawkish. Every regression holds fixed both betas x s,
GICS sector x s, the channels' own levels, and meeting fixed effects; standard errors are
clustered by meeting.

Pass bar, written before the run: p < 0.003 (0.05 over 17 channels) on all meetings, same sign
with p < 0.05 in 1994-2008 and in 2009-2021, and the matched-pair slope the same sign.
"Monetary-specific" also needs the placebo term (below) the same sign with p < 0.05.

## Verdict

**No channel clears the bar.** One comes close, and it is the same story the v4 conditional view
found: **distress**.

## Each channel alone (earnings-day firms excluded)

| Channel | All 221 | t | Big \|s\| >= 2.5bp (65) | 1994-2008 | 2009-2021 |
|---|---|---|---|---|---|
| **Distance to default** | **+0.195** | **2.86** | +0.188 (t 2.58) | +0.156 (t 2.19) | +0.357 (t 1.95) |
| Leverage | -0.083 | -2.14 | -0.070 | -0.049 | -0.267 (t -2.80) |
| Book-to-market | -0.202 | -2.04 | -0.184 | -0.184 | -0.358 |
| Interest coverage | +0.103 | 1.90 | +0.085 | +0.070 | +0.241 (t 3.56) |
| Cash | +0.097 | 1.78 | +0.085 | +0.080 | +0.144 |
| Implied duration | +0.064 | 1.30 | +0.081 | +0.068 | -0.011 |
| Rate beta (2y) | -0.039 | -0.60 | -0.077 | -0.123 | +0.544 (t 2.18) |
| Past beyond-beta Fed response (own) | -0.035 | -0.75 | -0.045 | -0.043 | +0.047 |
| Same, relative to TNIC peers | +0.006 | 0.20 | +0.009 | -0.001 | +0.032 |
| Announcement premium | -0.023 | -0.85 | -0.045 | -0.035 | +0.060 |
| Floating-rate pass-through | -0.033 | -1.01 | -0.041 | -0.040 | +0.005 |
| TNIC peers' 63-day return | +0.001 | 0.01 | -0.052 | -0.069 | +0.542 (t 2.12) |

Rate paid, short-term debt share, debt issuance, FCF and the volatility run-up are all |t| < 1.8
on all meetings.

## All channels together, placebo, matched pairs

- **Horse race (all 17 at once):** nothing above |t| 2.3. Distance to default is again the
  largest (+0.120, t 1.98; big meetings t 2.26). The distress channels (leverage, coverage,
  distance to default, rate paid) share the signal, so together each is weaker.
- **Matched pairs** (same meeting, sector, both betas and size; top vs bottom third): distance to
  default +0.26 per 10bp (-0.02, +0.55), the only pair slope near significance.
- **Placebo:** Fed days stacked with the 5 largest-market-move ordinary days after each meeting
  (1,105 days, no FOMC, CPI or jobs release). For distance to default the Fed-day excess is -0.009
  (t -0.36). On ordinary days, distressed firms also amplify market moves beyond what their betas
  say (t -2.4). **The distress effect is real beyond measured beta, but it is not specific to the
  Fed:** distressed firms' true beta on a big day is higher than their trailing beta, whoever
  moved the market.
- Two Fed-specific leads in the placebo (joint model only, not in the main test): FCF (Fed-day
  excess t 3.5, opposite sign to its main term) and the volatility run-up (t 2.6). Leads, not
  findings.

## Reading

1. **Distress is the one consistent channel.** Firms closer to default fall more on hawkish
   surprises and rise more on dovish ones, holding both betas, sector and size fixed. Size: about
   0.2pp per standard deviation per 10bp; a 2-SD-distressed firm on a 25bp surprise moves about
   1pp more than its betas predict. It narrowly misses the multiple-testing bar.
2. **It is fragility, not monetary policy.** The same firms over-react to big ordinary market
   moves too. For the trading and outlier questions this still matters (trailing beta understates
   distressed firms on big days); for the causal question it says the channel is "credit
   fragility amplifies shocks", not "this firm is exposed to rates".
3. **A firm's own past Fed response does not predict its next one** (t -0.75), nor does its rate
   beta, its floating-debt proxy, or its peers'. Sensitivity beyond beta is not a stable firm
   trait. That is the Amazon point measured: what predicts is the firm's current state (distress),
   not its identity or history. The event-history encoder (plan B1) must condition on state at
   each event, as the plan says; a firm-identity model would learn noise.
4. The 2009-2021 era shows larger, different effects (rate beta, peer returns, coverage), in line
   with every earlier finding that loadings shift between eras.

## Caveats

- Stand-ins, not the WRDS measures: distance to default is not a CDS spread; pass-through is a noisy
  floating-rate proxy (12 quarters of interest expense). The EDGAR market-risk text and, after
  Duo, CDS and options would test the same channels more directly.
- Day-0 close-to-close returns still carry non-earnings firm news.
- Exploratory: the bar was set before the run, but the channel list came after the v4 findings.
