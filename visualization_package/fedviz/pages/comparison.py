"""Comparison page — compare saved runs side by side.

Metrics come from each run's manifest (cheap). Per-meeting comparisons use the runs'
``predictions`` tables (loaded only when two runs are being compared): the paired
per-meeting MSE difference with a meeting-bootstrap range is the project's standard test.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fedviz.components.charts import bar_chart
from fedviz.components.filters import multi_select_filter
from fedviz.components.layout import guarded, metric_card, page_header, show_chart, tabs
from fedviz.data import VizData, _read_table
from fedviz.export import chart_export_menu
from fedviz.session import init_session_state
from fedviz.themes import ACCENT, NEGATIVE, POSITIVE, REFERENCE, WARNING

N_BOOT = 2000


def render(data: VizData | None = None) -> None:
    """Render the Comparison page."""
    init_session_state()
    data = data or VizData()
    page_header(
        "Comparison",
        "Saved runs side by side: metrics, rankings, and paired per-meeting error differences.",
    )

    runs_df = data.list_results()
    if runs_df.empty:
        st.info("No saved runs found. Run models first to compare them.")
        return

    with st.sidebar:
        st.markdown("**Comparison settings**")
        projects = multi_select_filter(runs_df, "project", "Projects")
        questions = multi_select_filter(runs_df, "question", "Questions")

    filtered = runs_df[runs_df["n_metrics"] > 0].copy()
    if projects:
        filtered = filtered[filtered["project"].isin(projects)]
    if questions:
        filtered = filtered[filtered["question"].isin(questions)]
    if filtered.empty:
        st.info("No runs with metrics match the filters.")
        return

    labels = {r["run_dir"]: f"{r['name']}  ·  {str(r['created'])[:16].replace('T', ' ')}"
              for r in filtered.to_dict("records")}

    tab_metrics, tab_paired, tab_ranking, tab_time = tabs([
        "Metrics", "Paired test", "Rankings", "Over time",
    ])
    with tab_metrics:
        guarded(_render_metrics_comparison, filtered, labels, label="Metrics")
    with tab_paired:
        guarded(_render_paired_analysis, filtered, labels, data, label="Paired test")
    with tab_ranking:
        guarded(_render_rankings, filtered, labels, label="Rankings")
    with tab_time:
        guarded(_render_performance_over_time, filtered, labels, data, label="Over time")


def _metrics(run_dir: str) -> dict:
    """Metrics from the manifest only (no tables read)."""
    path = Path(run_dir) / "manifest.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("metrics") or {}
    except (OSError, ValueError):
        return {}


def _numeric(metrics: dict) -> dict:
    return {k: float(v) for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)}


def _default_runs(runs_df: pd.DataFrame, n: int = 3) -> list[str]:
    """The newest primary run and its siblings (same name stem), else the newest runs."""
    primary = runs_df[runs_df["role"] == "primary"]
    if len(primary):
        stem = primary.iloc[0]["name"].rsplit("-", 1)[0]
        siblings = runs_df[runs_df["name"].str.startswith(stem)]
        if len(siblings) >= 2:  # the model first, baselines (pooled, capm) last: A = model, B = reference
            baseline = siblings["name"].str.contains("pooled|capm|baseline", regex=True)
            return pd.concat([siblings[~baseline], siblings[baseline]])["run_dir"].tolist()[:n]
    return runs_df["run_dir"].tolist()[:n]


def _predictions(run_dir: str) -> pd.DataFrame | None:
    path = Path(run_dir) / "tables" / "predictions.csv"
    if not path.exists():
        return None
    return _read_table(str(path), os.path.getmtime(path))


def _per_meeting_mse(run_dir: str) -> pd.Series | None:
    pred = _predictions(run_dir)
    if pred is None or not {"meeting", "target", "mu"} <= set(pred.columns):
        return None
    err = (pred["target"] - pred["mu"]) ** 2
    # per seed, per meeting, then the seed average: every meeting weighs the same
    keys = ["meeting", "seed"] if "seed" in pred.columns else ["meeting"]
    mse = err.groupby([pred[k] for k in keys]).mean()
    if "seed" in pred.columns:
        mse = mse.groupby(level=0).mean()
    mse.index = pd.to_datetime(mse.index)
    return mse


def _render_metrics_comparison(runs_df: pd.DataFrame, labels: dict) -> None:
    """Every metric for the chosen runs, then one metric as a chart."""
    selected = st.multiselect("Runs", list(labels), default=_default_runs(runs_df), format_func=labels.get)
    if not selected:
        st.info("Select at least one run.")
        return
    metrics = {labels[d].split("  ·  ")[0]: _metrics(d) for d in selected}
    keys = sorted({k for m in metrics.values() for k in m})
    table = pd.DataFrame({name: [m.get(k) for k in keys] for name, m in metrics.items()}, index=keys)
    shown = table.map(lambda v: "—" if v is None else (f"{v:,.4g}" if isinstance(v, (int, float)) else str(v)))
    st.dataframe(shown.rename_axis("Metric").reset_index(), width="stretch", hide_index=True,
                 height=min(38 * (len(keys) + 1), 480))

    numeric = sorted({k for m in metrics.values() for k in _numeric(m)})
    if numeric:
        default = "mse_per_meeting" if "mse_per_meeting" in numeric else numeric[0]
        metric = st.selectbox("Chart one metric", numeric, index=numeric.index(default))
        d = pd.DataFrame({"run": list(metrics), "value": [_numeric(m).get(metric, np.nan) for m in metrics.values()]})
        fig = bar_chart(d.dropna(), x="run", y="value", orientation="h", title=metric, sort=True,
                        height=120 + 44 * len(d), text_fmt=",.4g")
        fig.update_layout(yaxis_title=None, xaxis_title=None)
        show_chart(fig)


def _render_paired_analysis(runs_df: pd.DataFrame, labels: dict, data: VizData) -> None:
    """Per-meeting MSE of A minus B with a meeting-bootstrap 95% range (negative: A is better)."""
    has_pred = [d for d in labels if (Path(d) / "tables" / "predictions.csv").exists()]
    if len(has_pred) < 2:
        st.info("Paired tests need two runs that saved predictions.")
        return
    default = [d for d in _default_runs(runs_df) if d in has_pred] + has_pred
    c1, c2 = st.columns(2)
    with c1:
        dir_a = st.selectbox("Run A", has_pred, index=has_pred.index(default[0]), format_func=labels.get)
    with c2:
        rest = [d for d in has_pred if d != dir_a]
        b_default = next((d for d in default if d != dir_a), rest[0])
        dir_b = st.selectbox("Run B (reference)", rest, index=rest.index(b_default), format_func=labels.get)

    with st.spinner("Loading predictions…"):
        a, b = _per_meeting_mse(dir_a), _per_meeting_mse(dir_b)
    if a is None or b is None:
        st.info("One of these runs has no usable predictions.")
        return
    diff = (a - b).dropna()
    if diff.empty:
        st.info("The two runs share no test meetings.")
        return
    draws = np.random.default_rng(0).choice(diff.to_numpy(), size=(N_BOOT, diff.size)).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Shared meetings", f"{diff.size}")
    with c2:
        metric_card("MSE A", f"{a.loc[diff.index].mean():.3f}")
    with c3:
        metric_card("MSE B", f"{b.loc[diff.index].mean():.3f}")
    with c4:
        verdict = "A better" if hi < 0 else ("B better" if lo > 0 else "no clear winner")
        metric_card("A − B", f"{diff.mean():+.3f}", help=f"95% meeting-bootstrap range {lo:+.3f} to {hi:+.3f}: {verdict}")

    colors = [POSITIVE if v < 0 else NEGATIVE for v in diff]
    fig = go.Figure(go.Bar(x=diff.index, y=diff.values, marker_color=colors,
                           hovertemplate="%{x|%Y-%m-%d}: %{y:+.3f}<extra></extra>", name="A − B"))
    fig.add_hrect(y0=lo, y1=hi, fillcolor=WARNING, opacity=0.15, line_width=0)
    fig.add_hline(y=diff.mean(), line=dict(color=WARNING, width=2),
                  annotation_text=f"mean {diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]", annotation_position="top left")
    fig.add_hline(y=0, line=dict(color=REFERENCE, dash="dot", width=1))
    fig.update_layout(title="Per-meeting MSE difference, A minus B (green: A did better)", height=380,
                      bargap=0.15, showlegend=False)
    show_chart(fig)
    chart_export_menu(fig, diff.rename("mse_a_minus_b").reset_index(), prefix="paired_mse")


def _render_rankings(runs_df: pd.DataFrame, labels: dict) -> None:
    """Rank every run that reports the chosen metric."""
    all_metrics = {d: _numeric(_metrics(d)) for d in labels}
    counts = pd.Series([k for m in all_metrics.values() for k in m]).value_counts()
    if counts.empty:
        st.info("No numeric metrics available for ranking.")
        return
    options = counts.index.tolist()
    default = "mse_per_meeting" if "mse_per_meeting" in options else options[0]
    c1, c2 = st.columns([3, 1])
    with c1:
        metric = st.selectbox("Rank by", options, index=options.index(default),
                              format_func=lambda k: f"{k}  ({counts[k]} runs)")
    with c2:
        lower = st.toggle("Lower is better", value=any(s in metric for s in ("mse", "loss", "error")))
    d = pd.DataFrame([{"run": labels[r], "value": m[metric]} for r, m in all_metrics.items() if metric in m])
    d = d.sort_values("value", ascending=lower).head(25)
    d["run"] = d["run"].str.split("  ·  ").str[0]
    fig = bar_chart(d, x="run", y="value", orientation="h", sort=False, height=120 + 30 * len(d),
                    title=f"Top {len(d)} by {metric}", text_fmt=",.4g")
    fig.update_traces(marker_color=[ACCENT] + [REFERENCE] * (len(d) - 1))
    fig.update_layout(yaxis=dict(autorange="reversed", title=None), xaxis_title=None)
    show_chart(fig)


def _render_performance_over_time(runs_df: pd.DataFrame, labels: dict, data: VizData) -> None:
    """Per-meeting MSE through time for runs that saved predictions."""
    has_pred = [d for d in labels if (Path(d) / "tables" / "predictions.csv").exists()]
    if not has_pred:
        st.info("None of these runs saved predictions.")
        return
    default = [d for d in _default_runs(runs_df) if d in has_pred] or has_pred[:2]
    selected = st.multiselect("Runs", has_pred, default=default[:3], format_func=labels.get)
    fig = go.Figure()
    for run_dir in selected:
        mse = _per_meeting_mse(run_dir)
        if mse is not None:
            fig.add_trace(go.Scatter(x=mse.index, y=mse.values, mode="lines+markers",
                                     name=labels[run_dir].split("  ·  ")[0], marker=dict(size=5)))
    if fig.data:
        fig.update_layout(title="Squared error per test meeting", height=420, yaxis_title="MSE",
                          hovermode="x unified")
        show_chart(fig)
