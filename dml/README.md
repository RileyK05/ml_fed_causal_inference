# dml: Q1 aggregate response (`fedci`)

How much does the S&P 500 move per unit of unexpected Fed news? Partially linear DML on
the 30-minute announcement window, cross-fitted by walk-forward over meetings.
Result so far: θ ≈ −0.96% per 10bp hawkish surprise (163 meetings).

```text
dml/
  fedci/
    questions/q1_total_effect/   samples.py (estimation samples) + pipeline.py
    questions/q2_channels/       parked stub
    eval/dml.py                  PLR-DML, OLS baselines, leave-one-out
    api/                         FastAPI backend for the web app
    cli.py, serve.py             `fedci` command
  frontend/                      React web app (npm) + streamlit/ legacy viewer
  results/                       saved runs, results/<question>/<run_id>/
  tests/
  docs/                          frontend plans
  start.cmd                      double-click launcher
```

```powershell
.venv\Scripts\python -m pip install -e "core[dev]" -e "dml[dev,model]"
.venv\Scripts\fedci run q1             # saves to dml/results/
.venv\Scripts\fedci serve              # web app (or double-click dml\start.cmd)
.venv\Scripts\fedci app                # legacy Streamlit viewer
.venv\Scripts\python -m pytest dml
```

Data comes from `fedcore` (core/). This project never imports encoder/ or cde/.
