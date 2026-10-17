"""Results Browser page — browse and inspect all saved model runs.

Provides a comprehensive view of all runs across all projects, with
filtering, sorting, and detailed inspection of metrics, tables, and
figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fedviz.components.layout import show_chart

from fedviz.components.filters import (
    categorical_filter,
    multi_select_filter,
    numeric_range_filter,
)
from fedviz.components.layout import (
    columns,
    metric_card,
    page_header,
    section_header,
    tabs,
)
from fedviz.components.tables import data_table, metrics_table
from fedviz.data import VizData
from fedviz.export import chart_export_menu, export_df_csv, export_df_json
from fedviz.interactive import (
    comparison_overlay,
    cross_filter_panel,
    filter_df_by_state,
    selection_info_panel,
)
from fedviz.session import get_state, init_session_state, set_state


def render(data: VizData | None = None) -> None:
    """Render the Results Browser page."""
    init_session_state()
    data = data or VizData()
    page_header(
        "Results Browser",
        "Every saved run across the DML, encoder and CDE projects: what was run, with what data, and what it found.",
        icon="",
    )

    runs_df = data.list_results()
    if runs_df.empty:
        st.info("No saved runs found. Run a model first to see results.")
        return

    with st.sidebar:
        st.markdown("**Filters**")
        projects = multi_select_filter(runs_df, "project", "Projects")
        questions = multi_select_filter(runs_df, "question", "Questions")
        roles = multi_select_filter(runs_df, "role", "Roles")

    filtered = runs_df.copy()
    if projects:
        filtered = filtered[filtered["project"].isin(projects)]
    if questions:
        filtered = filtered[filtered["question"].isin(questions)]
    if roles:
        filtered = filtered[filtered["role"].isin(roles)]

    tab_list, tab_detail, tab_compare = tabs([
        "Runs", "Run detail", "Compare",
    ])

    with tab_list:
        _render_run_list(filtered)

    with tab_detail:
        _render_run_detail(filtered, data)

    with tab_compare:
        _render_compare(filtered, data)


def _run_label(row) -> str:
    created = str(row.get("created", ""))[:16].replace("T", " ")
    return f"{row['name']}  ·  {row['project']}  ·  {created}"


def _render_run_list(runs_df: pd.DataFrame) -> None:
    """Summary cards, then every run; click a row to open it in Run Detail."""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Runs", f"{len(runs_df):,}")
    with c2:
        metric_card("Projects", f"{runs_df['project'].nunique()}")
    with c3:
        metric_card("Primary runs", f"{int((runs_df['role'] == 'primary').sum())}",
                    help="One primary spec per active question; everything else adds to the testing count")
    with c4:
        latest = pd.to_datetime(runs_df["created"], errors="coerce").max()
        metric_card("Latest", "—" if pd.isna(latest) else f"{latest:%b %d, %Y}")

    query = st.text_input("Search runs", placeholder="name contains…", label_visibility="collapsed")
    shown = runs_df
    if query:
        shown = runs_df[runs_df["name"].str.contains(query, case=False, na=False)]

    view = shown[["name", "project", "question", "role", "created", "n_metrics"]].copy()
    view["created"] = pd.to_datetime(view["created"], errors="coerce")
    event = st.dataframe(
        view,
        width="stretch",
        hide_index=True,
        height=460,
        on_select="rerun",
        selection_mode="single-row",
        key="fv_run_table",
        column_config={
            "name": st.column_config.TextColumn("Run", width="large"),
            "project": st.column_config.TextColumn("Project", width="small"),
            "question": st.column_config.TextColumn("Question", width="small"),
            "role": st.column_config.TextColumn("Role", width="small"),
            "created": st.column_config.DatetimeColumn("Created", format="MMM D, YYYY HH:mm"),
            "n_metrics": st.column_config.NumberColumn("Metrics", width="small"),
        },
    )
    rows = getattr(getattr(event, "selection", None), "rows", None) or []
    if rows:
        st.session_state["fv_selected_run"] = shown.iloc[rows[0]]["run_dir"]
        st.caption(f"Selected **{shown.iloc[rows[0]]['name']}**. Open the Run Detail tab to inspect it.")


def _render_run_detail(runs_df: pd.DataFrame, data: VizData) -> None:
    """Everything saved with one run: metrics, parameters, data fingerprints, tables, figures."""
    if runs_df.empty:
        st.info("No runs to display.")
        return

    dirs = runs_df["run_dir"].tolist()
    preselected = st.session_state.get("fv_selected_run")
    index = dirs.index(preselected) if preselected in dirs else 0
    labels = {r["run_dir"]: _run_label(r) for r in runs_df.to_dict("records")}
    selected_run_dir = st.selectbox("Run", dirs, index=index, format_func=labels.get)
    row = runs_df.set_index("run_dir").loc[selected_run_dir]

    run_data = data.load_run(selected_run_dir)
    manifest = run_data.get("manifest", {})
    tables = run_data.get("tables", {})
    large = run_data.get("large_tables", {})
    figures = run_data.get("figures", {})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Project", row["project"])
    with c2:
        metric_card("Question", manifest.get("question", "—"))
    with c3:
        metric_card("Role", manifest.get("role", "—"))
    with c4:
        created = pd.to_datetime(manifest.get("created"), errors="coerce")
        metric_card("Created", "—" if pd.isna(created) else f"{created:%b %d, %H:%M}")
    st.caption(f"`{manifest.get('run_id', '—')}`")
    if manifest.get("notes"):
        st.info(manifest["notes"], icon=":material/notes:")

    left, right = st.columns([3, 2], gap="large")
    with left:
        metrics = manifest.get("metrics") or {}
        if metrics:
            st.markdown("### Metrics")
            m = pd.DataFrame({"Metric": list(metrics), "Value": [metrics[k] for k in metrics]})
            numeric = pd.to_numeric(m["Value"], errors="coerce")
            m["Value"] = [f"{v:,.4g}" if pd.notna(v) else str(raw) for v, raw in zip(numeric, m["Value"])]
            st.dataframe(m, width="stretch", hide_index=True, height=min(38 * (len(m) + 1), 420))
    with right:
        if manifest.get("params"):
            st.markdown("### Parameters")
            st.json(manifest["params"], expanded=1)
        if manifest.get("datasets"):
            with st.expander("Data fingerprints", icon=":material/fingerprint:"):
                ds = manifest["datasets"]
                if isinstance(ds, dict):
                    st.dataframe(pd.DataFrame({"Dataset": list(ds), "Fingerprint": [str(v) for v in ds.values()]}),
                                 width="stretch", hide_index=True)

    names = list(tables) + [f"{k} (large)" for k in large]
    if names:
        st.markdown("### Tables")
        selected_table = st.segmented_control("Table", names, default=names[0], label_visibility="collapsed")
        if selected_table:
            if selected_table.endswith(" (large)"):
                key = selected_table.removesuffix(" (large)")
                path = large[key]
                st.caption(f"{key} is large; showing the first 5,000 rows.")
                df = pd.read_csv(path, nrows=5000)
            else:
                key, df = selected_table, tables[selected_table]
            data_table(df, height=400)
            col1, col2 = st.columns(2)
            with col1:
                export_df_csv(df, f"{key}.csv")
            with col2:
                export_df_json(df, f"{key}.json")

    if figures:
        st.markdown("### Figures")
        fig_names = list(figures.keys())
        selected_fig = st.selectbox("View figure", fig_names)
        if selected_fig:
            fig_json = figures[selected_fig]
            if isinstance(fig_json, dict):
                fig = go.Figure(fig_json)
                show_chart(fig)
                chart_export_menu(fig, prefix=selected_fig)


def _render_compare(runs_df: pd.DataFrame, data: VizData) -> None:
    """Render comparison of multiple runs."""
    if len(runs_df) < 2:
        st.info("Need at least 2 runs to compare.")
        return

    run_options = [(_run_label(r), r["run_dir"]) for r in runs_df.to_dict("records")]

    labels = [o[0] for o in run_options]
    selected_labels = st.multiselect("Runs to compare", labels, default=labels[:2])

    if len(selected_labels) < 2:
        st.info("Select at least 2 runs to compare.")
        return

    selected_dirs = []
    for label, run_dir in run_options:
        if label in selected_labels:
            selected_dirs.append(run_dir)

    all_metrics = {}
    for run_dir in selected_dirs:
        run_data = data.load_run(run_dir)
        manifest = run_data.get("manifest", {})
        name = f"{manifest.get('project', '?')}/{manifest.get('name', '?')}"
        all_metrics[name] = manifest.get("metrics", {})

    if all_metrics:
        all_keys = set()
        for m in all_metrics.values():
            all_keys.update(m.keys())

        comparison_records = []
        for key in sorted(all_keys):
            record = {"Metric": key}
            for name, metrics in all_metrics.items():
                record[name] = metrics.get(key, "—")
            comparison_records.append(record)

        comparison_df = pd.DataFrame(comparison_records)
        st.dataframe(comparison_df, width="stretch", hide_index=True)

        numeric_metrics = []
        for key in sorted(all_keys):
            values = []
            for metrics in all_metrics.values():
                v = metrics.get(key)
                if isinstance(v, (int, float)):
                    values.append(v)
            if len(values) == len(all_metrics):
                numeric_metrics.append(key)

        if numeric_metrics:
            fig = go.Figure(data=[
                go.Bar(
                    name=name,
                    x=numeric_metrics,
                    y=[metrics.get(m, 0) for m in numeric_metrics],
                )
                for name, metrics in all_metrics.items()
            ]).update_layout(
                barmode="group",
                title="Metrics Comparison",
                height=500,
            )
            show_chart(fig)
