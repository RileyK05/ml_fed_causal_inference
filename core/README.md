# core: shared data layer (`fedcore`)

Everything the three projects share, and nothing else.

```text
core/
  data/                catalog.yaml, PROVENANCE.md, raw/ (never edited), interim/, processed/
  fedcore/
    data/              catalog, load(), query() (DuckDB views), meetings(), event_panel()
    protocol.py        walk_forward(), meeting_bootstrap(): splits by whole meetings
    results/           ResultStore (each project passes its own results/ folder)
    questions/         QuestionSpec + the five question definitions (metadata only)
    ingest/            future data pulls (WRDS etc.)
    cli.py             fedcore check | catalog | sql
  tests/               includes test_isolation.py: projects never import each other
```

```powershell
.venv\Scripts\python -m pip install -e "core[dev]"
.venv\Scripts\fedcore check
.venv\Scripts\python -m pytest core
```

Adding data = file under `data/{raw,interim,processed}/` + entry in `data/catalog.yaml` +
note in `data/PROVENANCE.md` + `fedcore check`.

The Q3 shared contract (panel format, synthetic data, scoring harness) that encoder/ and cde/
both plug into will live here as `fedcore.q3`. It's specified in
[docs/core/HANDOFF.md](../docs/core/HANDOFF.md), so both models are scored by
the same code.
