# trading: post-meeting trades on frozen Q3 predictions (`fedtrade`)

The Q3 encoder work (frozen at git tag `fed_pred_encoder`) produces, before each meeting, a ranking
of how much more or less each firm should react to a Fed surprise than its beta says. On
development data, that ranking predicts the next month's drift, not the day-0 move
(`docs/encoder/q3-v6-drift.md`).

This project turns the ranking into trades (Q4/Q5).
- **Stage 1 (frozen):** Q3's saved predictions give delta_b per firm.
- **Stage 2 (here):** which firms to hold, which way, and for how long.

Design and plan: [docs/trading/HANDOFF.md](../docs/trading/HANDOFF.md).

| Module | What |
|---|---|
| `signals.py` | `load_q3` reads a Q3 run's saved predictions (as files; no encoder import); `rate_sensitivity` gives delta_b |
| `books.py` | v1 rule books: `decile`, `sensitive`, `sensitive_sector` |
| `backtest.py` | `trades` (one row per big meeting) and `summary` (by surprise direction, meeting bootstrap, costs); CLI |

```powershell
.venv\Scripts\python -m pip install -e "core[dev]" -e "trading[dev]"
.venv\Scripts\python -m pytest trading
# No fitting; prints only. Saving a run needs the user's go-ahead.
.venv\Scripts\python -m fedtrade.backtest dev --no-save
```

```text
trading/
  fedtrade/         the package (imports fedcore only)
  tests/
  results/          this project's saved runs (append-only; per-firm tables never committed)
```
