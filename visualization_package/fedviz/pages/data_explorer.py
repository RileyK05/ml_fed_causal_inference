"""Data Explorer page — every catalog dataset: what it is, what it covers, what is in it.

Coverage and row counts come from DuckDB metadata queries, so nothing large is loaded just
to describe it. The dataset viewer loads small datasets in full and previews large ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fedviz.components.charts import distribution, time_series
from fedviz.components.filters import multi_select_filter
from fedviz.components.layout import guarded, metric_card, page_header, show_chart, tabs
from fedviz.components.tables import data_table
from fedviz.data import VizData
from fedviz.export import chart_export_menu, export_df_csv, export_df_json
from fedviz.session import get_state, init_session_state, set_state
from fedviz.themes import CATEGORICAL_PALETTE

FULL_LOAD_ROWS = 500_000   # above this, the viewer shows a preview
PREVIEW_ROWS = 200_000
LAYER_COLORS = {"raw": CATEGORICAL_PALETTE[0], "interim": CATEGORICAL_PALETTE[1], "processed": CATEGORICAL_PALETTE[2]}


@st.cache_data(ttl=600, show_spinner=False)
def _coverage() -> pd.DataFrame:
    return VizData().quick_coverage()


def render(data: VizData | None = None) -> None:
    """Render the Data Explorer page."""
    init_session_state()
    data = data or VizData()
    page_header(
        "Data Explorer",
        "Every dataset in the shared catalog: coverage, contents, and a SQL console over all of them.",
    )

    catalog = data.catalog
    if catalog.empty:
        st.warning("No dataset catalog found.")
        return

    with st.sidebar:
        st.markdown("**Dataset filters**")
        layers = multi_select_filter(catalog, "layer", "Layers")
        grains = multi_select_filter(catalog, "grain", "Grains")

    filtered = catalog.copy()
    if layers:
        filtered = filtered[filtered["layer"].isin(layers)]
    if grains:
        filtered = filtered[filtered["grain"].isin(grains)]

    tab_catalog, tab_coverage, tab_dataset, tab_sql = tabs(["Catalog", "Coverage", "Dataset", "SQL console"])
    with tab_catalog:
        guarded(_render_catalog, filtered, label="Catalog")
    with tab_coverage:
        guarded(_render_coverage, filtered, label="Coverage")
    with tab_dataset:
        guarded(_render_dataset_view, filtered, data, label="Dataset")
    with tab_sql:
        guarded(_render_sql_console, data, label="SQL console")


def _render_catalog(catalog: pd.DataFrame) -> None:
    """Counts, then the catalog itself."""
    counts = catalog["layer"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Datasets", f"{len(catalog)}")
    for col, layer in zip((c2, c3, c4), ("raw", "interim", "processed")):
        with col:
            metric_card(layer.capitalize(), f"{int(counts.get(layer, 0))}")
    st.dataframe(
        catalog[["name", "layer", "grain", "questions", "description", "source", "date_col", "key_col"]],
        width="stretch",
        hide_index=True,
        height=520,
        column_config={
            "name": st.column_config.TextColumn("Dataset", width="medium"),
            "layer": st.column_config.TextColumn("Layer", width="small"),
            "grain": st.column_config.TextColumn("Grain", width="small"),
            "questions": st.column_config.TextColumn("Questions", width="small"),
            "description": st.column_config.TextColumn("Description", width="large"),
            "source": st.column_config.TextColumn("Source", width="medium"),
            "date_col": st.column_config.TextColumn("Date column", width="small"),
            "key_col": st.column_config.TextColumn("Key", width="small"),
        },
    )


def _render_coverage(catalog: pd.DataFrame) -> None:
    """When each dataset starts and ends, and how big it is."""
    with st.spinner("Reading dataset metadata…"):
        cov = _coverage()
    cov = cov[cov["dataset"].isin(catalog["name"])]
    present = cov[cov["exists"] & cov["start"].notna()].sort_values("start")
    if present.empty:
        st.info("No readable datasets match the filters.")
        return

    fig = go.Figure()
    for layer, g in present.groupby("layer", sort=False):
        for _, r in g.iterrows():
            fig.add_trace(go.Scatter(
                x=[r["start"], r["end"]], y=[r["dataset"], r["dataset"]], mode="lines",
                line=dict(color=LAYER_COLORS.get(layer, CATEGORICAL_PALETTE[3]), width=10),
                name=layer, legendgroup=layer, showlegend=bool(r.name == g.index[0]),
                hovertemplate=(f"{r['dataset']}<br>{r['start']:%Y-%m-%d} to {r['end']:%Y-%m-%d}"
                               f"<br>{int(r['rows']):,} rows, {int(r['columns'])} columns<extra></extra>"),
            ))
    fig.update_layout(title="Date coverage by dataset", height=110 + 24 * len(present),
                      yaxis=dict(autorange="reversed", showgrid=False, type="category"),
                      xaxis=dict(type="date"), hovermode="closest")
    show_chart(fig)

    missing = cov[~cov["exists"]]
    if len(missing):
        st.caption("Not present locally: " + ", ".join(f"`{d}`" for d in missing["dataset"]))
    st.dataframe(
        cov.sort_values("rows", ascending=False),
        width="stretch",
        hide_index=True,
        column_config={
            "dataset": "Dataset", "layer": "Layer", "grain": "Grain", "exists": "Present",
            "rows": st.column_config.NumberColumn("Rows", format="%,d"),
            "columns": st.column_config.NumberColumn("Columns"),
            "start": st.column_config.DateColumn("Start"),
            "end": st.column_config.DateColumn("End"),
        },
    )


def _render_dataset_view(catalog: pd.DataFrame, data: VizData) -> None:
    """One dataset: size, preview, summary statistics, a distribution and a time series."""
    names = catalog["name"].tolist()
    if not names:
        st.info("No datasets available.")
        return
    default = names.index("etf_returns") if "etf_returns" in names else 0
    selected = st.selectbox("Dataset", names, index=default)
    info = catalog.set_index("name").loc[selected]
    if info["description"]:
        st.caption(info["description"])

    n_rows = data.row_count(selected)
    with st.spinner(f"Loading {selected}…"):
        if n_rows is not None and n_rows > FULL_LOAD_ROWS:
            df = data.preview(selected, PREVIEW_ROWS)
            st.info(f"{selected} has {n_rows:,} rows; showing the first {PREVIEW_ROWS:,}. "
                    "Use the SQL console for the rest.", icon=":material/info:")
        else:
            df = data.load(selected)
    if df.empty:
        st.warning(f"Dataset {selected} is empty.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Rows", f"{(n_rows or len(df)):,}")
    with c2:
        metric_card("Columns", str(len(df.columns)))
    with c3:
        metric_card("Missing cells", f"{df.isna().mean().mean():.1%}", help="Share of the loaded rows")
    with c4:
        metric_card("Memory", f"{df.memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB", help="Of the loaded rows")

    data_table(df, height=300)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    tab_stats, tab_dist, tab_ts = tabs(["Summary", "Distribution", "Time series"])
    with tab_stats:
        if numeric_cols:
            st.dataframe(df[numeric_cols].describe().T, width="stretch")
    with tab_dist:
        if numeric_cols:
            col = st.selectbox("Column", numeric_cols, key="fv_dist_col")
            fig = distribution(df, col, title=f"Distribution of {col}", height=380, show_kde=True)
            fig.update_layout(showlegend=False)
            show_chart(fig)
            chart_export_menu(fig, df[[col]].dropna(), prefix=f"{col}_dist")
    with tab_ts:
        date_col = info["date_col"] if info["date_col"] in df.columns else None
        if date_col and numeric_cols:
            cols = st.multiselect("Columns", numeric_cols, default=numeric_cols[:1], key="fv_ts_cols")
            if cols:
                d = df[[date_col, *cols]].copy()
                d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
                if d[date_col].duplicated().any():  # panel data: one line per column, averaged per date
                    d = d.groupby(date_col, as_index=False)[cols].mean()
                    st.caption("Several rows per date: showing the per-date average.")
                show_chart(time_series(d.sort_values(date_col), x=date_col, y=cols, height=400,
                                       title=", ".join(cols)))
        else:
            st.info("This dataset has no date column to plot against.")


def _render_sql_console(data: VizData) -> None:
    """Read-only SQL over every catalog dataset (DuckDB views with the catalog names)."""
    st.caption("Every dataset is a view with its catalog name. The connection is read-only.")
    query = st.text_area("SQL", value=get_state("sql_query", "SELECT * FROM etf_returns LIMIT 100"),
                         height=120, label_visibility="collapsed")
    set_state("sql_query", query)

    if st.button("Run query", type="primary", icon=":material/play_arrow:"):
        with st.spinner("Running…"):
            try:
                result = data.sql(query)
                set_state("sql_results", result)
                st.toast(f"{len(result):,} rows", icon=":material/check:")
            except Exception as e:
                set_state("sql_results", None)
                st.error(f"Query failed: {e}")

    sql_results = get_state("sql_results")
    if sql_results is not None:
        data_table(sql_results, height=400)
        col1, col2 = st.columns(2)
        with col1:
            export_df_csv(sql_results, "query_results.csv")
        with col2:
            export_df_json(sql_results, "query_results.json")

    picked = st.pills("Start from a table", data.catalog["name"].tolist(), key="fv_sql_table")
    if picked and picked != st.session_state.get("_fv_sql_last"):
        st.session_state["_fv_sql_last"] = picked
        set_state("sql_query", f"SELECT * FROM {picked} LIMIT 100")
        st.rerun()
