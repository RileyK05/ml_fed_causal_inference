# encoder: patch-transformer Q3 model (`fedenc`)

Reads each firm's last 252 trading days as 51 weekly patches (BERT-style), fuses fundamentals
and market context into a firm state z, and predicts the meeting-day move as a(z) + b(z)·S.
b(z) is the firm's Fed sensitivity.

```text
encoder/
  HANDOFF.md        build spec for this project
  docs/             transformer.md (full design note)
  fedenc/           the package (imports fedcore only)
  tests/
  results/          this project's saved runs
```

```powershell
.venv\Scripts\python -m pip install -e "core[dev]" -e "encoder[dev,dl]"
.venv\Scripts\python -m pytest encoder
```

Prerequisite: the shared Q3 contract, `core/docs/HANDOFF-q3-foundation.md`.
