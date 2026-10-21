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

## Sources for the attributed events

Events the research agents tied to dated news (confidence as they reported it). Unlisted events
were "not found". Move sizes are omitted (CRSP data is licensed).

| Meeting | Ticker | Cause | Source | Confidence |
|---|---|---|---|---|
| 2001-11-06 | HWP, CPQ | Hewlett family opposes the HP-Compaq merger | https://money.cnn.com/2001/11/06/deals/hp_compaq/ | high |
| 2001-11-06 | ENE | Enron seeking at least $2B in private cash | CNN Enron timeline (page URL not captured) | low |
| 2001-11-06 | NT | $1.1B Sprint supply deal announced the day before | https://northernminer.com/news/trading-summary-765/1000013943 | low |
| 2002-08-13 | DE | Fiscal Q3 results released that day | https://www.sec.gov/Archives/edgar/data/0000315189/000031518902000046/htmltemplate.htm | medium |
| 2002-08-13 | TIF | Q2 earnings released that day | https://www.sec.gov/Archives/edgar/data/0000098246/000009824602000039/form8_k081302.txt | low |
| 2002-11-06 | PBG | Pepsi-Gemex tender completed; 2003 guidance | https://www.sec.gov/Archives/edgar/data/0001076405/000095012302010290/y65245ae8vk.txt | medium |
| 2002-11-06 | CPN | Lodi contract renegotiated; Q3 results | https://www.sec.gov/Archives/edgar/data/0000916457/000091645702000035/i10502.txt | medium |
| 2003-06-25 | NFB | Balance-sheet restructuring, buyback, dividend (8-K the day before) | https://www.sec.gov/Archives/edgar/data/0000352510/000095012303007386/y87816e8vk.txt | low |
| 2004-01-28 | THC | Restructuring: seeking buyers for 27 hospitals | https://www.sec.gov/Archives/edgar/data/0000070318/000110465904002089/a04-1707_18k.htm | medium |
| 2004-01-28 | TUP | Q4 results, weak North America, lower guidance | https://www.sec.gov/Archives/edgar/data/0001008654/000089109204000340/e16716ex99_1.txt | medium |
| 2004-01-28 | PKI | Q4 beat, raised 2004 guidance | https://www.sec.gov/Archives/edgar/data/0000031791/000095013504000378/b49279pkexv99w1.txt | medium |
| 2004-01-28 | STJ | Q4 sales +27% | https://www.sec.gov/Archives/edgar/data/0000203077/000089710104000160/stjude040391_ex1.htm | medium |
| 2004-01-28 | ABI | Fiscal Q2 results | https://www.sec.gov/Archives/edgar/data/0000077551/000112528204000242/b329816_ex99-1.htm | medium |
| 2004-01-28 | AV | Fiscal Q1 results after the prior close | https://www.sec.gov/Archives/edgar/data/0001116521/000110465904001841/a04-1595_1ex99d1.htm | medium |
| 2007-09-18 | ETFC | Profit warning the evening before | https://www.calculatedriskblog.com/2007/09/etrade-warns-of-mortgage-fallout.html | medium |
| 2007-09-18 | LEH | Earnings beat, but AP credits the close to the Fed cut (fed-rate) | https://honoluluadvertiser.com/article/2007/Sep/19/bz/hawaii709190391.html | medium |
| 2008-01-30 | ABK, MBI | Downgrade speculation on the bond insurers | https://www.finance.cz/zpravy/reuters/MTFH70327nHKG280053-global-markets-u-s-bond-insurer-woes-shadow-big-fed-rate-cut | medium |
| 2008-01-30 | CTX | Fiscal Q3 loss announced the day before | https://www.sec.gov/Archives/edgar/data/0000018532/000095013408001283/d53373exv99w1.htm | medium |
| 2008-01-30 | PHM | Q4 write-downs and loss | https://www.sec.gov/Archives/edgar/data/0000822416/000095012408000422/k23417exv99w1.htm | medium |
| 2008-01-30 | IFF | Q4 results | https://ir.iff.com/node/9556 | low |
| 2008-01-30 | UIS | Q4 results the day before | https://www.sec.gov/Archives/edgar/data/0000746838/000074683808000013/fourthqearnings.txt | low |
| 2008-01-30 | DOV | Q4 results | https://www.sec.gov/Archives/edgar/data/29905/000095012308000917/y47508exv99w1.htm | medium |
| 2008-01-30 | RHI | Q4 results the day before | https://www.sec.gov/Archives/edgar/data/0000315213/000119312508014459/dex991.htm | medium |
| 2008-09-16 | CEG | Post-Lehman contagion fears (industry) | https://thedailyrecord.com/2008/09/16/street-dings-constellation/ | medium |
| 2008-09-16 | AIG | Own distress before the evening rescue | https://money.cnn.com/2008/09/16/news/companies/AIG/ | medium |
| 2008-09-16 | MS | Fiscal Q3 results that day | https://investmentnews.com/industry-news/morgan-stanley-rushes-out-earnings/17402 | medium |
| 2008-09-16 | MU | JMP downgrade | https://idahobusinessreview.com/?p=13083 | low |
| 2008-09-16 | WM | CEO comments on the downgrade; takeover talk | https://consumerist.com/2008/09/16/wamus-stock-bumps-upwards | low |
| 2008-12-16 | ETN | Q4 guidance cut that day | https://www.sec.gov/Archives/edgar/data/31277/000095015208010301/l34862aexv99.htm | medium |
| 2011-09-21 | ORCL | Fiscal Q1 results the evening before | https://www.bloomberg.com/news/articles/2011-09-21/oracle-rises-after-corporate-software-spending-boosts-earnings | high |
| 2011-09-21 | RRC | Report of a possible Shell bid | https://www.aol.com/2011-09-21-range-resources-shares-soared-what-you-need-to-kno.html | high |
| 2011-09-21 | HPQ | Reports Whitman would replace Apotheker | https://www.theregister.co.uk/2011/09/22/whitman_hp_vote/ | high |
| 2011-09-21 | GIS | Fiscal Q1 results | https://www.sec.gov/Archives/edgar/data/40704/000119312511252810/d234780dex99.htm | medium |
