# encoder: Q3 two-part patch-transformer model (`fedenc`)

Predicts each firm's Fed-day return as `mu = a(z) + b(z) x shock`, where z is the firm's last
252 trading days (51 weekly patches through a small transformer) fused with fundamentals and
meeting context. The locked model (`TwoPartArm`) takes the meeting-wide move from pooled OLS and
the firm-to-firm differences from an average of centered-a networks.

**Status (2026-10-16):** final test done. The two-part model ties the CAPM baseline
(`capm_pooled`) on 2022-2025; b beyond beta is not detectable. Full record:
[encoder-report.md](../docs/reports/encoder-report.md). The earlier pretraining code
(masked-patch and next-return pretraining, checkpoints, staged fine-tuning) was removed; it is
at git tag `q3-pretraining-archive`, its docs in `docs/archive/encoder/`.

| Module | What |
|---|---|
| `decouple.py` | `StructModel` (centered a), `DecoupleArm`, `TwoPartArm`; commands `run`, `confirm`, `final` |
| `diagnostics.py` | MSE split, window and surprise tables, a/b anatomy, b grades, from saved predictions |
| `workflow.py` | `bounded_panel` and its variants: `universe`, `sector`, `target="market_adjusted"`, `shock="ME"` |
| `runs.py` | `save_runs` (fit arms on shared folds, save one run per arm) and `b_grade` |
| `encoder_transformer.py` | `PatchTransformerEncoder` |
| `cheap.py`, `cheap_gradients.py` | Cheap premise tests (`docs/encoder/cheap-test-menu.md`) |

```powershell
# No fitting: re-score saved runs (add --no-save to print only)
.venv\Scripts\python -m fedenc.diagnostics --no-save two_part=<run dir> pooled=<run dir> --reference pooled
# Training: needs the user's go-ahead in chat
.venv\Scripts\python -m fedenc.decouple confirm --through 2021-12-31 --test-size 32 --device cuda
```

`decouple final` is the one-shot 2022-2025 test; it has been run and must not be rerun or tuned on.

```text
encoder/
  fedenc/           the package (imports fedcore only)
  tests/
  results/          this project's saved runs (gitignored, append-only)
```

```powershell
.venv\Scripts\python -m pip install -e "core[dev]" -e "encoder[dev,dl]"
.venv\Scripts\python -m pytest encoder
```

Prerequisite: the shared Q3 contract, `docs/core/HANDOFF.md`.
