# encoder: patch-transformer Q3 model (`fedenc`)

Reads each firm's last 252 trading days as 51 weekly patches (BERT-style), fuses fundamentals
and market context into a firm state z, and predicts the meeting-day move as a(z) + b(z)·S.
b(z) is the firm's Fed sensitivity.

Pretraining predicts the **next ordinary trading day's raw return** from history: its
expected value m and variance v, scored with `log(v) + (r_next - m)²/v`. Transfer retains the encoder and its history scaler,
then trains the existing Fed response heads. A dated masked-reconstruction comparator
is available. [Setup, chronology, commands and artifact format](../docs/encoder/next-return-pretraining.md).

```powershell
# No data loading or training:
.venv\Scripts\python -m fedenc.workflow plan --cutoff 2000-10-03
```

Training is a separate explicit command and still needs the user's go-ahead in chat.
Pretrained arms load an explicit checkpoint; they never start pretraining automatically.

```text
encoder/
  fedenc/           the package (imports fedcore only)
  tests/
  results/          this project's saved runs
```

```powershell
.venv\Scripts\python -m pip install -e "core[dev]" -e "encoder[dev,dl]"
.venv\Scripts\python -m pytest encoder
```

Prerequisite: the shared Q3 contract, `docs/core/HANDOFF.md`.
