# Q3: do the model's calls come true after the meeting? (development data)

Run 2026-10-20, exploratory and post hoc (asked after the v6 results), development data only.
- Windows that would reach past 2021-12-31 are dropped, so no 2022 returns enter.
- Scripts: scratchpad `dig_drift.py` and `dig_drift_check.py`; shared code `fedcore.q3.drift`.
  Nothing saved to a ResultStore.
- Not a pre-stated test. It has been added to the final test as a pre-stated measure
  (`q3-v6-final-plan.md`).

The question: when the model moves a firm away from beta, does the stock keep moving that way
over the next days and weeks?
- "The call": dev, the model's prediction minus the two-beta base, demeaned within the meeting.
- Abnormal return: return minus beta63 x market, daily.
- CAR_k: the sum over days 1..k after the meeting, demeaned within the meeting.
- Big meetings: |s| >= 2.5bp, 40 without the two standouts.

## 1. Drift spread (top vs bottom decile of the call, bp of k-day drift per 10bp surprise, 40 meetings without the standouts)

| Arm | 1 day | 5 days | 10 days | 20 days |
|---|---|---|---|---|
| F0s (v6 summary MLP) | +41 (-5, +85) | +186 (+60, +316) | +268 (+75, +438) | **+374 (+122, +613)** |
| B4 (v4 summary MLP, separate run) | +26 (-28, +67) | +180 (+78, +284) | +243 (+55, +377) | **+358 (+132, +552)** |
| F0 (v6 transformer) | +19 (-52, +66) | +33 (-48, +120) | +99 (-25, +216) | +144 (+3, +318) |
| B5 (v4 transformer) | +31 (-39, +80) | +2 (-86, +100) | +102 (-40, +225) | +134 (-13, +289) |

For a typical 4bp surprise, +374 per 10bp is about 1.5 percentage points between the summary
model's top and bottom deciles over 20 trading days.

## 2. Is it the Fed, or a characteristic drift?

| Arm, 20 days | Spread | Sign-flip placebo p | Net of size, B/M, leverage, cash, DD, beta63, 63-day momentum | Placebo p |
|---|---|---|---|---|
| F0s | +374 | 0.003 | **+186 (+85, +327)** | 0.001 |
| B4 | +358 | 0.002 | **+190 (+68, +367)** | 0.004 |

- **Sign-flip placebo:** each meeting's surprise sign is randomly flipped 2,000 times, keeping
  |s|. The real spread beats almost every flip, so the drift follows the Fed's direction. A
  static tilt, such as value or small caps rallying in January, would not pass this.
- **Characteristic control:** about half of the drift is explained by the usual
  characteristics, measured within the meeting. The other half survives the control and the
  placebo.

## 3. When the market disagreed on day 0

The 10% strongest calls (largest |dev|, either direction) in each big meeting, all 42 big meetings
(standouts included). "Against" means the day-0 move beyond beta had the opposite sign to the
call. Signed CAR is in the call's direction, in bp (not per 10bp).

| F0s | 5 days | 20 days | 20-day hit rate |
|---|---|---|---|
| Day-0 move went against the call (n ~1,065) | +38 (-11, +85) | +75 (-24, +187) | 0.53 (0.49, 0.58) |
| Day-0 move agreed with the call (n ~1,035) | +53 (+4, +103) | **+148 (+35, +261)** | 0.56 (0.51, 0.61) |

- Both groups lean the model's way; only the agreed group's interval excludes zero. The gap
  between the groups (+148 vs +75) is not tested and the intervals overlap heavily.
- The day-0 move on its own does not predict the drift: in a within-meeting regression of CAR_20
  on the call and the day-0 move, the day-0 coefficient has t ~ 0.9. So the drift is not generic
  momentum in day-0 moves; it is tied to the model's ranking.
- So the model doesn't reliably catch day-0 mistakes. Where the market agreed, the move kept
  going.

## 4. Why it works where it works, and what to distrust

1. **Concentration.**
   - 2008-12-16 (the cut to zero) carries about 35% of the 20-day spread.
   - Then 2001-11, 2002-08, 2002-11, 2003-05: dovish crisis-era meetings, after which the model's
     distressed firms kept rallying.
   - 2008-09-16 (Lehman week) goes the other way.
2. **The call is a credit tilt.** Its strongest link is distance to default (rank correlation
   +0.39); the summary MLP learned Track A's distress channel. The drift is consistent with
   markets taking weeks to price how Fed easing relieves distressed firms.
3. **The model's sizes are wrong.** A pooled regression of CAR on the raw call shows nothing
   (t ~ 0): the folds with the largest corrections (2004-2007, 2011-2019) carry little drift.
   The useful part is the ranking. In the 2001-2003 and 2008 folds the model barely departed
   from beta, but its ordering was right.
4. **Multiple testing.** 4 arms x 4 horizons x 2 slices were looked at after the fact. The two
   summary-MLP runs agree, but this stays exploratory until the final test.
5. **Delistings don't drive it.** Firms dropped from the 20-day window (under 80% of days
   observed) are 0.19% of the bottom (distressed) decile, 0.67% of the top, and 0.27% of the
   middle.

## 5. Rough trading backcheck (descriptive only, not used to select anything)

Scratchpad `backcheck.py`.
- **Trade:** at the day-0 close of each big meeting, long the decile the model says gains most
  from this surprise, short the one it says loses most. Hold days 1-20, beta-hedged.
- **Units:** % of one leg, before costs.

| F0s, 42 big meetings | Mean / trade | Median | Win rate | t | Worst | Annualized Sharpe (2.4 trades/yr) |
|---|---|---|---|---|---|---|
| Follow the call, equal-weighted | +1.65 | +0.50 | 0.57 | 2.7 | -5.5 | 0.64 |
| Follow the call, value-weighted | +1.42 | +1.35 | 0.62 | 2.2 | -12.0 (2008-09-16) | 0.51 |
| Reversion (day 0 went against the call) | +0.89 | +0.19 | 0.50 | 1.6 | -7.2 | 0.39 |

- B4 is close: +1.50 equal-weighted and +1.24 value-weighted.
- **By period, per trade (equal-weighted):** 2001-03 +3.5, 2004-07 +0.6, 2008-09 +1.6, 2010-15
  +0.6, 2016-19 +2.2.
- **Costs:** both legs open and close once per trade.
  - Break-even is about 0.7-0.8% round trip per leg.
  - Large caps clear that; small distressed names and shorting them after hawkish surprises
    likely don't.

## 6. Reversion to the model's fair value, firm by firm: no

Scratchpad `revert.py`.
- **Trade:** gap = day-0 return minus the model's predicted return, demeaned within the meeting.
  Long the decile most overshot down, short the decile most overshot up; hold k days,
  beta-hedged.
- **Result:** no edge at any horizon. Big meetings, 20 days: -0.61% per trade (t -0.8). All 161
  meetings, 5 days: +0.29% (t 1.6), the same as plain day-0 reversal (+0.25%), so that isn't
  the model.
- **Why:**
  - Within a meeting, firms' day-0 returns vary by about 2%; the model's predictions vary by
    about 0.13%, and its departure from CAPM by 0.02-0.04%.
  - So the gap is firm news. It is 0.9999 correlated with the gap to CAPM, and 98% of the
    "overshot" decile is the same firms under either model.
  - Model minus CAPM version: under 0.02% per trade at every horizon.
- **Where a gap can be measured:** only at the decile-portfolio level, where firm news averages
  out. There the market under-moves on day 0 and keeps drifting the model's way (sections 1 and
  5). That is continuation, not reversion.

## 7. The same trade on raw prices (nothing removed)

Scratchpad `revert_raw.py`.
- **Setup:** gap = actual day-0 return minus the model's predicted return, both raw. Forward
  returns are raw and compounded, with no hedge.
- **Firm level:** still nothing. Big meetings, 20 days: -0.44% per trade (t -0.9). All meetings,
  5 days: +0.31% (t 1.5).
- **Market level:** average actual minus average predicted move across firms. If the market fell
  more than predicted, buy the average stock; if it rose more, short it.
  - All 161 meetings, 5 days: +0.40% per trade (t 2.2, win 0.56).
  - Fading the day-0 move with no model: +0.35% (t 1.9).
  - Slope of the next-5-day return on the gap: -0.23, so about a quarter of the overshoot
    reverses.
  - Big meetings: noise at every horizon past day 1.
- **Read:** the market-level fade is the only place the gap idea shows anything. Most of it is
  generic reversal after Fed days, and the model adds about 0.05% per trade. One of 9 cells
  tested, so treat it as a lead only.

## 8. Firm-level gap trade: horizons, news purge, double confirmation

Scratchpad `revert_purge.py`; checks suggested by another agent. Same firm-level raw-gap book.
- **News purge:** drop firms with a Compustat earnings date (rdq) or an 8-K in days -1 to +1.
  - That is 20% of firm-meetings.
  - Such firms are over-represented in the gap extremes: 28% vs 18% in the middle, all meetings.
- **Horizons:** day 1, days 1-3, 1-5, 2-20 (Q4's spec) and 1-20.
- **Double confirmation:** the model's call is in the top 20% by size, and day 0 went the other
  way by more than 1 sd.
- **Result:** no cell reaches |t| = 2. Best: all meetings, days 1-5, +0.31% (t 1.5), before the
  purge. Purging changes each cell by under 0.2%. Double confirmation on big meetings, days 2-20:
  -1.46% (t -1.2).
- The firm-level reversion idea is closed on development data.

## 9. Why the losing decile trades lost

Scratchpad `why_fail.py` and `why_fail_legs.py`.
- 18 of 42 trades lost.
- Firm names stay out of this doc: per-firm data is licensed.

1. **The edge is the distressed leg, in both directions.** Each leg's 20-day CAR relative to the
   meeting's average firm, % per trade:

   | | Distressed leg | Cash-rich / tech leg |
   |---|---|---|
   | Dovish (20): long distressed, short tech | **+1.57 (t 2.2, 80% wins)** | +0.65 (t 1.8) |
   | Hawkish (22): short distressed, long tech | **+1.58 (t 2.4)** | -0.44 (t -1.0, 41% wins) |

   Distressed firms follow the Fed's direction over the next month after hikes as well as cuts.
   The hawkish trades are weak (+1.1%, 50% wins) because the long-tech leg loses.
2. **Sector bets cause the clustered losses.**
   - After hawkish surprises the long leg is about 37-60% information technology.
   - A bad month for tech or semiconductors sinks the trade: 2004-08-10 lost -5.5%, of which
     the sector part was -4.75; 2004-01-28 lost -2.8, sector -2.6.
   - Across all trades, the sector part is +0.27% per trade (t 0.9) and the stock part is +1.39%
     (t 3.0). The sector exposure adds variance, not return.
3. **The dovish losers fall into three kinds.**
   - **Credit not yet bottomed:** 2008-10-29. Distressed financials, REITs and retailers kept
     collapsing (-40% to -90%), and easing didn't save insolvent firms.
   - **Commodity shocks:** 2014-12-17 and 2019-06-19. Levered energy names in the long leg fell
     with oil.
   - **Tech rallies against the short leg:** 2004-03-16 and 2007-09-18.
4. **Single-name blowups.** In 10 of the 18 losers, the worst 5 of about 100 names lost more
   than the whole trade did, so the rest of the book was positive. Firms losing more than 20%
   cost 1.4% per trade on average. They are somewhat more common with earnings or an 8-K in the
   window (2.9% vs 1.9% of names).
5. **Beta-hedge error in crashes.** In 2008-09-16 the worst short was a financial with a very
   high beta63. Its beta hedge over-shot during the market's 18% fall, giving a +100% "abnormal"
   return. Within the long leg, higher beta63 predicts worse outcomes (-1.0pp per sd, t -2.3),
   the only significant firm trait.
6. **Not explained by:** 20-day SMB and HML moves (R2 0.10), market direction, or size,
   leverage, cash or momentum within the legs (all |t| < 1.6).
   - VIX and credit spreads aren't on disk before 2023.
   - `crsp_delist` is too sparse to test delistings.
