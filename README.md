# FOMC research

How do firms respond to unanticipated Fed news, and do deviations from their predicted
responses forecast subsequent returns? Spec: [docs/question.md](docs/question.md).

Three isolated projects on one shared data layer:

| Folder | What | Start here |
|---|---|---|
| [core/](core/README.md) | Shared data layer: datasets, `load()`/SQL, walk-forward splits, results store, question definitions | `fedcore check` |
| [dml/](dml/README.md) | Q1: aggregate response via DML, plus the web app and viewer | `dml\start.cmd` |
| [encoder/](encoder/README.md) | Q3: patch-transformer firm-sensitivity model | [encoder/HANDOFF.md](encoder/HANDOFF.md) |
| [cde/](cde/README.md) | Q3: Neural CDE firm-sensitivity model | [cde/HANDOFF.md](cde/HANDOFF.md) |

Projects import `core` and never each other (enforced by `core/tests/test_isolation.py`).

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e "core[dev]" -e "dml[dev,model]" -e "encoder[dev]" -e "cde[dev]"
.venv\Scripts\fedcore check
.venv\Scripts\python -m pytest core dml encoder cde
```

Or double-click `dml\start.cmd` to set up and launch the DML web app.
