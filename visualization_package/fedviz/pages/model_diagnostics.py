"""Model Diagnostics page — what a saved run learned and how well it fits.

Works from whatever a run saved: Q3 runs keep per-firm predictions (``predictions``) and
training curves (``curves``); Q1 DML runs keep ``residuals``, ``estimates`` and ``influence``.
Both keep their walk-forward ``folds``. Only the tabs a run has data for are shown.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fedviz.components.charts import (
    bar_chart,
    distribution,
    forest_plot,
    qq_plot,
    scatter,
)
from fedviz.components.layout import guarded, metric_card, page_header, show_chart
from fedviz.data import VizData, _read_table
from fedviz.export import export_summary_report
from fedviz.session import init_session_state
from fedviz.themes import ACCENT, NEGATIVE, REFERENCE, WARNING

SAMPLE = 20_000  # points drawn in scatter-type charts; summaries use every row


def render(data: VizData | None = None) -> None:
    """Render the Model Diagnostics page."""
    init_session_state()
    data = data or VizData()
    page_header(
        "Model Diagnostics",
        "How well a saved run fits: errors over time, residual shape, estimates and walk-forward windows.",
    )

    runs_df = data.list_results()
    if runs_df.empty:
        st.info("No saved runs found. Run a model first to see diagnostics.")
        return

    labels = {r["run_dir"]: f"{r['name']}  ·  {r['project']}  ·  {str(r['created'])[:16].replace('T', ' ')}"
              for r in runs_df.to_dict("records")}
    dirs = list(labels)
    primary = runs_df.loc[(runs_df["role"] == "primary") & (runs_df["n_metrics"] > 0), "run_dir"]
    default = st.session_state.get("fv_selected_run") or (primary.iloc[0] if len(primary) else None)
    run_dir = st.selectbox("Run", dirs, index=dirs.index(default) if default in dirs else 0,
                           format_func=labels.get)
    st.session_state["fv_selected_run"] = run_dir

    run_data = data.load_run(run_dir)
    manifest = run_data.get("manifest", {})
    tables = run_data.get("tables", {})
    large = run_data.get("large_tables", {})

    predictions = tables.get("predictions")
    if predictions is None and "predictions" in large:
        path = large["predictions"]
        with st.spinner("Loading predictions…"):
            predictions = _read_table(path, os.path.getmtime(path))

    sections = [("Overview", lambda: _render_overview(manifest, tables))]
    if predictions is not None and {"target", "mu"} <= set(predictions.columns):
        sections.append(("Predictions", lambda: _render_predictions(predictions)))
    if "curves" in tables and {"epoch", "val_loss"} <= set(tables["curves"].columns):
        sections.append(("Training", lambda: _render_curves(tables["curves"])))
    if "residuals" in tables:
        sections.append(("Residuals", lambda: _render_residuals(tables["residuals"])))
    if "estimates" in tables:
        sections.append(("Estimates", lambda: _render_estimates(tables["estimates"])))
    if "influence" in tables:
        sections.append(("Influence", lambda: _render_influence(tables["influence"])))
    if "folds" in tables:
        sections.append(("Walk-forward", lambda: _render_walk_forward(tables["folds"])))

    for (name, fn), tab in zip(sections, st.tabs([s[0] for s in sections])):
        with tab:
            guarded(fn, label=name)

    with st.sidebar:
        if st.button("Export report", icon=":material/download:", width="stretch"):
            export_summary_report(
                title=f"Diagnostics: {manifest.get('name', 'Unknown')}",
                metrics=manifest.get("metrics", {}),
                figures=[],
                dataframes=list(tables.values())[:3],
            )


def _fmt(value) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return str(value)
    return f"{value:,.4g}"


def _render_overview(manifest: dict, tables: dict) -> None:
    """Role, the headline metrics, the parameters, and the notes saved with the run."""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Question", manifest.get("question", "—"))
    with c2:
        metric_card("Role", manifest.get("role", "—"))
    metrics = manifest.get("metrics") or {}
    headline = [k for k in ("mse_per_meeting", "rank_ic", "b_slope", "theta", "estimate") if k in metrics]
    headline += [k for k, v in metrics.items() if k not in headline and isinstance(v, (int, float))]
    for col, key in zip((c3, c4), headline[:2]):
        with col:
            metric_card(key.replace("_", " "), _fmt(metrics[key]))
    if manifest.get("notes"):
        st.info(manifest["notes"], icon=":material/notes:")

    left, right = st.columns([3, 2], gap="large")
    with left:
        if metrics:
            st.markdown("### Metrics")
            st.dataframe(pd.DataFrame({"Metric": list(metrics), "Value": [_fmt(v) for v in metrics.values()]}),
                         width="stretch", hide_index=True, height=min(38 * (len(metrics) + 1), 460))
    with right:
        if manifest.get("params"):
            st.markdown("### Parameters")
            st.json(manifest["params"], expanded=1)
        if tables:
            st.caption("Saved tables: " + ", ".join(f"`{t}`" for t in tables))


def _render_predictions(pred: pd.DataFrame) -> None:
    """Q3: per-meeting error over time, fit, residual shape and the spread of a and b."""
    p = pred.copy()
    p["meeting"] = pd.to_datetime(p["meeting"])
    p["gap"] = p["target"] - p["mu"] if "gap" not in p.columns else p["gap"]
    n_seeds = p["seed"].nunique() if "seed" in p.columns else 1

    per_meeting = p.groupby("meeting").agg(mse=("gap", lambda g: float(np.mean(np.square(g)))),
                                           n=("gap", "size")).reset_index()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Test meetings", f"{len(per_meeting)}")
    with c2:
        metric_card("Rows", f"{len(p):,}", help=f"{n_seeds} seed(s)")
    with c3:
        metric_card("Mean per-meeting MSE", f"{per_meeting['mse'].mean():.3f}")
    with c4:
        worst = per_meeting.loc[per_meeting["mse"].idxmax()]
        metric_card("Worst meeting", f"{worst['meeting']:%Y-%m-%d}", help=f"MSE {worst['mse']:.2f}")

    fig = go.Figure(go.Bar(x=per_meeting["meeting"], y=per_meeting["mse"], marker_color=ACCENT,
                           hovertemplate="%{x|%Y-%m-%d}: MSE %{y:.2f}<extra></extra>"))
    fig.add_hline(y=per_meeting["mse"].mean(), line=dict(color=WARNING, dash="dash", width=1.5),
                  annotation_text="mean", annotation_position="top left")
    fig.update_layout(title="Squared error per test meeting", height=320, yaxis_title="MSE", bargap=0.15)
    show_chart(fig)

    sample = p.sample(min(SAMPLE, len(p)), random_state=0)
    left, right = st.columns(2, gap="medium")
    with left:
        fig = scatter(sample, x="mu", y="target", title="Actual against predicted return",
                      xaxis_title="Predicted (%)", yaxis_title="Actual (%)", height=380, opacity=0.25,
                      add_regression=True)
        fig.update_traces(marker=dict(size=4), selector=dict(mode="markers"))
        show_chart(fig)
    with right:
        fig = distribution(sample, "gap", title="Prediction error (actual minus predicted)",
                           xaxis_title="Error (%)", height=380, show_kde=True)
        fig.update_layout(showlegend=False)
        show_chart(fig)

    if {"a_hat", "b_hat"} <= set(p.columns):
        left, right = st.columns(2, gap="medium")
        for col, name, title in ((left, "a_hat", "Estimated a (shock-free move, %)"),
                                 (right, "b_hat", "Estimated b (sensitivity, % per 10bp)")):
            with col:
                fig = distribution(sample, name, title=title, xaxis_title=None, height=300, show_kde=True)
                fig.update_layout(showlegend=False)
                show_chart(fig)


def _render_curves(curves: pd.DataFrame) -> None:
    """Training and validation loss by epoch, one line per fold and seed."""
    c = curves.copy()
    keys = [k for k in ("fold", "seed") if k in c.columns]
    fig = go.Figure()
    for i, (key, g) in enumerate(c.groupby(keys) if keys else [((), c)]):
        label = ", ".join(f"{k} {v}" for k, v in zip(keys, key if isinstance(key, tuple) else (key,)))
        fig.add_trace(go.Scatter(x=g["epoch"], y=g["val_loss"], mode="lines", name=f"val · {label}",
                                 line=dict(width=2), legendgroup=label))
        if "train_loss" in g:
            fig.add_trace(go.Scatter(x=g["epoch"], y=g["train_loss"], mode="lines", name=f"train · {label}",
                                     line=dict(width=1, dash="dot"), opacity=0.6, legendgroup=label,
                                     showlegend=False))
    fig.update_layout(title="Loss by epoch (solid: validation, dotted: training)", height=420,
                      xaxis_title="Epoch", yaxis_title="Loss", hovermode="x unified")
    show_chart(fig)


def _render_residuals(residuals_df: pd.DataFrame) -> None:
    """DML: residual shape and residuals over time."""
    resid_col = next((c for c in ["y_resid", "residual", "resid", "epsilon", "e"] if c in residuals_df.columns),
                     None) or residuals_df.select_dtypes(include=[np.number]).columns[0]
    residuals = residuals_df[resid_col].dropna()
    left, right = st.columns(2, gap="medium")
    with left:
        fig = distribution(residuals_df, resid_col, title=f"Residual distribution ({resid_col})",
                           height=340, show_kde=True)
        fig.update_layout(showlegend=False)
        show_chart(fig)
    with right:
        show_chart(qq_plot(residuals, title="Residual Q-Q plot", height=340))

    date_col = next((c for c in ["announcement_date", "date", "meeting"] if c in residuals_df.columns), None)
    if date_col:
        d = residuals_df.assign(**{date_col: pd.to_datetime(residuals_df[date_col])})
        colors = [NEGATIVE if v < 0 else ACCENT for v in d[resid_col]]
        fig = go.Figure(go.Bar(x=d[date_col], y=d[resid_col], marker_color=colors,
                               hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra></extra>"))
        fig.update_layout(title="Residuals by meeting", height=320, bargap=0.2)
        show_chart(fig)


def _render_estimates(estimates_df: pd.DataFrame) -> None:
    """Point estimates with confidence intervals."""
    est = "theta" if "theta" in estimates_df.columns else (
        "estimate" if "estimate" in estimates_df.columns else estimates_df.select_dtypes(np.number).columns[0])
    label = next((c for c in ["estimator", "name", "term"] if c in estimates_df.columns), estimates_df.columns[0])
    fig = forest_plot(
        estimates_df, estimate=est,
        low="ci_low" if "ci_low" in estimates_df.columns else None,
        high="ci_high" if "ci_high" in estimates_df.columns else None,
        label=label, title="Estimates with 95% intervals", height=120 + 48 * len(estimates_df),
    )
    show_chart(fig)
    st.dataframe(estimates_df, width="stretch", hide_index=True)


def _render_influence(influence_df: pd.DataFrame) -> None:
    """How much each meeting moves the estimate when it is left out."""
    delta_col = next((c for c in ["delta_loo", "delta", "theta_delta", "loo_delta", "influence"]
                      if c in influence_df.columns), None) or influence_df.select_dtypes(np.number).columns[0]
    label_col = next((c for c in ["announcement_date", "meeting", "date"] if c in influence_df.columns),
                     influence_df.columns[0])
    d = influence_df.assign(_label=influence_df[label_col].astype(str).str[:10]).sort_values(delta_col)
    fig = bar_chart(d, x="_label", y=delta_col, orientation="h", title="Change in the estimate when a meeting is left out",
                    height=max(300, 26 * len(d) + 80), text_fmt="+.3f", color_scale="sign", sort=False)
    fig.update_layout(yaxis_title=None, xaxis_title="Leave-one-out change")
    show_chart(fig)


def _render_walk_forward(folds_df: pd.DataFrame) -> None:
    """Each fold's training and test window on one timeline."""
    f = folds_df.drop_duplicates("fold") if "fold" in folds_df.columns else folds_df
    if not {"train_start", "train_end", "test_start", "test_end"} <= set(f.columns):
        st.dataframe(folds_df, width="stretch", hide_index=True)
        return
    fig = go.Figure()
    for _, r in f.iterrows():
        name = f"fold {r['fold']}" if "fold" in f else ""
        for start, end, color, part in ((r["train_start"], r["train_end"], REFERENCE, "train"),
                                        (r["test_start"], r["test_end"], ACCENT, "test")):
            fig.add_trace(go.Scatter(x=[start, end], y=[name, name], mode="lines", line=dict(color=color, width=14),
                                     name=part, legendgroup=part, showlegend=bool(_ == f.index[0]),
                                     hovertemplate=f"{name} {part}: {start} to {end}<extra></extra>"))
    fig.update_layout(title="Walk-forward windows (grey: training, blue: test)", height=110 + 38 * len(f),
                      yaxis=dict(autorange="reversed", showgrid=False), xaxis=dict(type="date"))
    show_chart(fig)
    st.dataframe(folds_df, width="stretch", hide_index=True)
