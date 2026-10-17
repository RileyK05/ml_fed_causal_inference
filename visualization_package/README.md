# visualization_package: FedViz (`fedviz`)

A Streamlit app for exploring the FOMC data and every saved model run in this repo.

```powershell
.venv\Scripts\python -m fedviz            # or: fedviz   (opens http://localhost:8501)
.venv\Scripts\python -m pytest visualization_package
```

`python -m fedviz` starts Streamlit from this folder so the theme in `.streamlit/config.toml`
applies (light and dark, Inter font); extra Streamlit flags pass through, e.g.
`python -m fedviz --server.port 8601`. Viewers switch light/dark under Settings; the charts follow.

| Section | Page | What it shows |
|---|---|---|
| Markets | Market Overview | Sector cumulative returns, volatility, rates and drawdowns, FOMC meetings marked |
| | Event Study | Returns around each announcement, by event day, and against the policy surprise |
| | Cross-Section | Distributions, scatter matrix, rankings and correlations across sectors |
| Models | Model Diagnostics | One run: per-meeting error, fit, training curves; DML residuals, estimates, influence |
| | Results Browser | Every saved run (`<project>/results/<question>/<run>/`): metrics, parameters, tables |
| | Comparison | Runs side by side; paired per-meeting MSE difference with a meeting-bootstrap range |
| Data | Data Explorer | Catalog, date coverage, any dataset, and a read-only SQL console |

Design system: `fedviz/themes.py` (palette, sector colours, Plotly template) and
`fedviz/components/layout.py` (`page_header`, `metric_card`, `show_chart`, `guarded`). Render
every chart through `show_chart` so it picks up the template for the viewer's mode.
