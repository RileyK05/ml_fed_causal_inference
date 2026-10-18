# Who moved on the biggest Fed surprises, and why

2026-10-18, at the user's request: instead of scoring models, look at the stocks the two-beta
CAPM (`capm_beta_both`, out of sample) missed most on the 10 development meetings with
|s| >= 5bp, and find the news. 10 movers per meeting (5 most under, 5 most over), 100 firm-events.
News lookups by 10 research agents (web search; every call needs a dated source; "not found"
is allowed). Exact firm returns are omitted here (CRSP data is licensed).

**Coverage caveat.** Searches hit rate limits on several meetings, so "not found" often means
"not searched to the end", not "no news existed". The firm-news count is a lower bound.

## Tally

| Meeting | Fed surprise | Firm news found | Fed/rate attributed | Not found | Notable |
|---|---|---|---|---|---|
| 2001-11-06 | 50bp cut, dovish | 4 | 0 | 6 | HP up / Compaq down: the Hewlett family opposed the merger (high confidence) |
| 2002-08-13 | hold, easing bias | 2 | 0 | 8 | Deere, Tiffany earnings; four software names fell with no news |
| 2002-11-06 | 50bp cut, dovish | 2 | 0 | 8 | Distressed power producers and an airline rallied; little company news found |
| 2003-06-25 | 25bp, hawkish | 1 | 0 | 9 | Mostly rate-limited |
| 2004-01-28 | "considerable period" dropped | 6 | 0 | 4 | Earnings season: Tenet, Tupperware, PerkinElmer, St. Jude, Applied Biosystems, Avaya |
| 2007-09-18 | 50bp cut, dovish | 1 | 1 | 8 | AP credits Lehman's +10% to the cut (it also beat earnings); Moody's, McGraw-Hill, Federated: no news |
| 2008-01-30 | 50bp cut | 8 | 0 | 2 | Ambac and MBIA on downgrade speculation; homebuilders and others on earnings |
| 2008-09-16 | hold after Lehman | 4 (+1 industry) | 0 | 5 | AIG collapse, Morgan Stanley earnings, Micron downgrade, WaMu takeover talk |
| 2008-12-16 | cut to zero + asset purchases | 1 | 0 | 9 | Five insurers and a REIT jumped with no company news; reports say insurers led the Fed-day rally |
| 2011-09-21 | Operation Twist, growth warning | 4 | 0 | 6 | All risers had firm news (Oracle, Range, HP, General Mills); coal, iron ore and financials fell with none |
| **Total** | | **33** | **1** | **65** | |

## Earnings days, measured directly

Compustat report dates (`rdq`) on the meeting day or the trading day before. Same out-of-sample
two-beta CAPM misses, S&P 500, development meetings.

| Meetings | Earnings-day share of firm-meetings | Share of squared firm-vs-firm miss | Share of each meeting's 10 biggest misses |
|---|---|---|---|
| all 161 | 6.2% | **21.4%** | **34%** |
| \|s\| >= 2.5bp (42) | 4.1% | 15.3% | 23% |
| \|s\| >= 5bp (10) | 4.1% | 8.7% | 19% |

## What this says

1. **The single biggest misses are mostly firm news that happened to land on a Fed day:**
   earnings, mergers, downgrades, rescues. Over all meetings, the 6% of firms reporting earnings
   carry a fifth of the error the models are trained and scored on. That error is noise for a
   question about Fed sensitivity, and no pre-meeting feature can predict it.
2. **What is left looks like industry-in-the-crisis, not stable firm traits.** The movers with no
   company news move as groups, and the group changes each time:
   - distressed power producers and airlines on the November 2002 cut;
   - ratings agencies and asset managers on the September 2007 cut;
   - life insurers and REITs on the zero-rate cut in December 2008;
   - software on the August 2002 weakness warning;
   - coal, iron ore and financials on Operation Twist in 2011.

   That would also explain why every sector or trait loading flipped between eras: the
   sensitive industry is whichever one sits at the center of that episode's stress.
3. **Implication for the model.** A static trait x shock loading cannot capture this. Two inputs
   that could:
   - which industries are under stress *going into* the meeting (recent industry drawdown,
     recent sensitivity to rates or credit);
   - firm news flags (earnings within a day) that keep idiosyncratic jumps out of training and
     scoring.

## Proposed next checks (not run)

- **Earnings filter:** re-score every saved model with earnings-day firms excluded (or flagged),
  as a pre-stated secondary score. No retraining.
- **Recent-stress feature:** each GICS industry's return, drawdown and rate sensitivity over the
  63 trading days before the meeting, interacted with s; tested out of sample on |s| >= 2.5bp
  meetings against the two-beta CAPM.
- **Second news pass** on the unexplained group moves (2002-11-06, 2007-09-18, 2008-12-16,
  2011-09-21) with sector-level sources, to confirm they were Fed-day sector moves.
