"""Project home: the five questions, where each stands, and how far the data reaches."""
import _common as c  # noqa: I001  (sets up the import path)

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fedci.questions import specs
from fedci.results import ResultStore

c.page("Home", ":material/home:")

TARGET_START = pd.Timestamp("1998-01-01")  # docs/question.md: extend returns to 1998 (~230 meetings)
TARGET_MEETINGS = 230

st.title("FOMC causal inference")
st.caption("Q1 establishes the aggregate baseline; Q3 predicts firm responses; Q4 tests prediction gaps; "
           "Q5 evaluates trading usefulness. Q2 is parked. Spec: docs/question.md")
st.markdown("> The complexity belongs in representation learning, not in the causal claim.")

meetings = c.data.meetings()
cov = c.coverage()
store = ResultStore()
runs = store.summary()

k1, k2, k3, k4 = st.columns(4)
k1.metric(f"Scheduled meetings in data (target ~{TARGET_MEETINGS})", len(meetings))
k2.metric("Datasets registered", int(cov.exists.sum()), f"{int((~cov.exists).sum())} missing" if (~cov.exists).any() else None)
k3.metric("Saved runs", len(runs))
k4.metric("Primary-spec runs", int((runs.role == "primary").sum()) if len(runs) else 0)

# ---- Questions board ----
st.subheader("Questions")
qs = specs()
cols = st.columns(len(qs))
for col, spec in zip(cols, qs.values()):
    n_runs = int((runs.question == spec.id).sum()) if len(runs) else 0
    with col.container(border=True):
        st.markdown(f"**{spec.id} - {spec.title}**")
        st.caption(spec.question)
        st.markdown(f":material/flag: {spec.status}  \n:material/science: {n_runs} runs")
        if spec.depends_on:
            st.caption("Needs " + ", ".join(spec.depends_on))
        if spec.data_needed:
            st.caption(f":material/database: {len(spec.data_needed)} data pulls pending")

st.code(
    "Q1: treatment + aggregate baseline --> Q3: expected firm response\n"
    "Q3: out-of-sample prediction gap   --> Q4: future-return test --> Q5: strategy\n"
    "Ordinary-day benchmark            --> independent Q3/Q4 comparison\n"
    "Q2: channels                      --> parked",
    language=None,
)

# ---- Data reach: each dataset's date span against the 1998 target ----
st.subheader("Data reach")
st.caption("Each bar spans a dataset's first to last date. Only the calendar multiplies the treatment: "
           "anything starting after the dashed line limits Q1, Q4b and Q5.")
have = cov[cov.exists].copy()
have["start"], have["end"] = pd.to_datetime(have.start), pd.to_datetime(have.end)
have = have.assign(layer_i=have.layer.map(c.LAYER_ORDER.index)).sort_values(["layer_i", "start"])
fig = go.Figure()
for i, layer in enumerate(c.LAYER_ORDER):
    d = have[have.layer == layer]
    if d.empty:
        continue
    fig.add_bar(
        name=layer, y=d.dataset, x=(d.end - d.start).dt.days * 86_400_000, base=d.start, orientation="h",
        marker=dict(color=c.series_colors()[i], cornerradius=4), width=0.6,
        customdata=d[["start", "end", "rows"]].astype(str).values,
        hovertemplate="%{y}<br>%{customdata[0]} to %{customdata[1]}<br>%{customdata[2]} rows<extra>" + layer + "</extra>",
    )
fig.add_vline(x=TARGET_START, line=dict(dash="dash", width=1, color=c.INK[c.mode()]["secondary"]))
fig.add_annotation(x=TARGET_START, y=1, yref="paper", text="1998 target", showarrow=False,
                   xanchor="left", yanchor="bottom", font=dict(color=c.INK[c.mode()]["secondary"]))
fig.update_xaxes(type="date")
fig.update_yaxes(autorange="reversed")
c.show(c.style(fig, height=96 + 28 * len(have), left=210).update_layout(hovermode="closest", bargap=0.3))

with st.expander("Coverage table"):
    st.dataframe(cov, hide_index=True, width="stretch")
