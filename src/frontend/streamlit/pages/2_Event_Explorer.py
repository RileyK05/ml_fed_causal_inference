"""Look at any meeting in event time: its surprise, the day-0 cross-section, and the path
around it. A viewer over fedci.data.event_panel -- no estimation happens here."""
import _common as c  # noqa: I001

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

c.page("Event explorer", ":material/event:")
st.title("Event explorer")

meet = c.data.meetings()
mps = c.load("mps_surprises").set_index("Date")
stmt = c.load("usmpd_statements").set_index("Date")
rets = c.load("etf_returns")
tickers = [t for t in rets.columns if t != "date"]
covered = meet[meet.announcement_date.between(rets.date.min(), rets.date.max())]

st.caption(f"{len(meet)} scheduled meetings in the calendar; {len(covered)} fall inside the returns data "
           f"({rets.date.min().date()} to {rets.date.max().date()}).")

date = st.selectbox("Meeting", covered.announcement_date[::-1],
                    format_func=lambda d: f"{d.date()}  -  {meet.set_index('announcement_date').notes.get(d, '')}")

# ---- the surprise ----
st.subheader("Surprise")
cols = st.columns(6)
for col, (label, src, field) in zip(cols, [
    ("MP1", stmt, "MP1"), ("MP2", stmt, "MP2"), ("STMT", mps, "STMT"),
    ("PC", mps, "PC"), ("ME", mps, "ME"), ("SEP meeting", stmt, "SEP"),
]):
    v = src[field].get(date) if field in src else None
    col.metric(label, "n/a" if v is None or pd.isna(v) else (("yes" if v else "no") if field == "SEP" else f"{v:+.4f}"))
st.caption("MP1/MP2 in percentage points (fed funds futures). STMT/PC/ME: Acosta et al. (2025) surprises, "
           "normalized to move the 1-year yield one-for-one.")

# ---- day-0 cross-section ----
st.subheader("Announcement-day returns")
day0 = c.data.event_panel("etf_returns", events=[date])
day0 = day0[day0.rel_day == 0].sort_values("value")
pos, neg = c.DIVERGING[c.mode()]
fig = go.Figure(go.Bar(
    x=day0.value, y=day0.series, orientation="h",
    marker=dict(color=[pos if v >= 0 else neg for v in day0.value], cornerradius=4),
    hovertemplate="%{y}: %{x:.2%}<extra></extra>",
))
fig.update_xaxes(tickformat=".1%")
c.show(c.style(fig, height=380, legend=False, left=56).update_layout(hovermode="closest"))

# ---- path around the meeting ----
st.subheader("Cumulative return around the meeting")
a, b, t = st.columns([1, 1, 3])
pre = a.number_input("Trading days before", 0, 30, 5)
post = b.number_input("Trading days after", 0, 60, 20)
pick = t.multiselect("Tickers (max 8)", tickers, default=[x for x in ["SPY", "XLF", "XLU", "XLK"] if x in tickers],
                     max_selections=8)
if pick:
    p = c.data.event_panel("etf_returns", pre=pre, post=post, columns=pick, events=[date])
    p = p.sort_values(["series", "rel_day"])
    p["cum"] = p.groupby("series")["value"].transform(lambda s: (1 + s.fillna(0)).cumprod() - 1)
    fig = px.line(p, x="rel_day", y="cum", color="series", category_orders={"series": pick},
                  color_discrete_sequence=c.series_colors(), custom_data=["date"])
    fig.update_traces(hovertemplate="%{customdata[0]|%Y-%m-%d}: %{y:.2%}")
    fig.add_vline(x=0, line=dict(dash="dash", width=1, color=c.INK[c.mode()]["secondary"]))
    fig.update_yaxes(tickformat=".1%", title=None)
    fig.update_xaxes(title="trading days relative to announcement")
    c.show(c.style(fig, legend=len(pick) > 1))
    st.caption("Cumulated from the first day shown; a missing return counts as zero only for this picture.")

# ---- across meetings ----
st.subheader("All covered meetings")
m1, m2 = st.columns(2)
shock = m1.selectbox("Surprise measure", ["STMT", "MP1", "MP2", "ME", "PC"])
tick = m2.selectbox("Ticker", tickers, index=tickers.index("SPY") if "SPY" in tickers else 0)
panel = c.data.event_panel("etf_returns", columns=[tick])
panel = panel[panel.rel_day == 0][["announcement_date", "value"]]
src = mps if shock in mps else stmt
panel["shock"] = panel.announcement_date.map(src[shock])
panel["highlight"] = panel.announcement_date.eq(date).map({True: "selected meeting", False: "other meetings"})
fig = px.scatter(panel, x="shock", y="value", color="highlight",
                 color_discrete_map={"other meetings": c.series_colors()[0], "selected meeting": c.series_colors()[1]},
                 custom_data=["announcement_date"])
fig.update_traces(marker=dict(size=9, line=dict(width=2, color="rgba(0,0,0,0)")),
                  hovertemplate="%{customdata[0]|%Y-%m-%d}<br>" + shock + " %{x:+.4f}<br>return %{y:.2%}<extra></extra>")
fig.update_yaxes(tickformat=".1%", title=f"{tick} return, day 0")
fig.update_xaxes(title=shock)
c.show(c.style(fig).update_layout(hovermode="closest"))
st.caption(f"n = {panel.shock.notna().sum()} meetings. Descriptive only; estimation belongs in the Q1 pipeline.")
