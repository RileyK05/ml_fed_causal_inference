# cde: Neural CDE Q3 model (`fedcde`)

The continuous-time alternative to the transformer: a hidden state driven by each firm's
price path, dh = f_θ(h) dX. Includes the signature + ridge baseline (the "linear CDE").
Same contract and scoring as the encoder, so the two are directly comparable.

```text
cde/
  fedcde/           the package (imports fedcore only)
  tests/
  results/          this project's saved runs
```

```powershell
.venv\Scripts\python -m pip install -e "core[dev]" -e "cde[dev,dl]"
.venv\Scripts\python -m pytest cde
```

Prerequisite: the shared Q3 contract, `docs/core/HANDOFF.md`.
