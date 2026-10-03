"""Browse any registered dataset: metadata, integrity, preview, and a time-series view."""
import _common as c  # noqa: I001

import plotly.express as px
import streamlit as st

c.page("Data catalog", ":material/database:")
st.title("Data catalog")
st.caption("Everything registered in data/catalog.yaml. Add a dataset there and it appears here, "
           "in fedcore.data.load(), and as a SQL view.")

cat = c.data.catalog()
layer = st.segmented_control("Layer", ["all", *c.LAYER_ORDER], default="all")
names = [n for n, d in cat.items() if layer in (None, "all") or d.layer == layer]
name = st.selectbox("Dataset", names, format_func=lambda n: f"{n}  ({cat[n].layer}, {cat[n].grain})")
ds = cat[name]

left, right = st.columns([2, 1])
with left:
    st.markdown(f"**{ds.name}** - {ds.description}")
    st.caption(f"Source: {ds.source}  \nFile: data/{ds.path}  \nKey: {', '.join(ds.key)}  \n"
               f"Used by: {', '.join(ds.questions) or 'none'}")
with right:
    problems = c.data.check(ds)
    if problems:
        st.error("Integrity check failed:\n\n" + "\n".join(f"- {p}" for p in problems), icon=":material/error:")
    else:
        st.success("Integrity check passed: file present, dates parse, key unique", icon=":material/check_circle:")

if not ds.exists:
    st.stop()

df = c.load(name)
d = df[ds.date_column]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", f"{len(df):,}")
m2.metric("Columns", df.shape[1])
m3.metric("From", str(d.min().date()))
m4.metric("To", str(d.max().date()))

tab_view, tab_missing, tab_table = st.tabs(["Chart", "Missing values", "Table"])

numeric = [col for col in df.select_dtypes("number").columns if col != ds.date_column]
with tab_view:
    if not numeric:
        st.info("No numeric columns to chart.")
    else:
        pick = st.multiselect("Series", numeric, default=numeric[: min(3, len(numeric))])
        if pick:
            long = df.melt(id_vars=ds.date_column, value_vars=pick, var_name="series")
            if len(pick) <= len(c.series_colors()):
                fig = px.line(long, x=ds.date_column, y="value", color="series",
                              color_discrete_sequence=c.series_colors(),
                              markers=ds.grain != "daily")
                c.show(c.style(fig, legend=len(pick) > 1))
            else:  # more series than categorical slots -> small multiples, one hue
                fig = px.line(long, x=ds.date_column, y="value", facet_col="series", facet_col_wrap=4,
                              color_discrete_sequence=c.series_colors()[:1])
                fig.update_yaxes(matches=None, showticklabels=True)
                fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
                c.show(c.style(fig, height=180 * ((len(pick) + 3) // 4), legend=False))
            st.caption("Gaps are real: nothing is filled or interpolated.")

with tab_missing:
    miss = df.isna().sum().rename("missing").to_frame()
    miss["share"] = (miss.missing / len(df)).round(4)
    st.dataframe(miss[miss.missing > 0].sort_values("missing", ascending=False), width="stretch")
    if not (miss.missing > 0).any():
        st.caption("No missing values.")

with tab_table:
    st.dataframe(df, hide_index=True, width="stretch", height=480)
    st.download_button("Download CSV", df.to_csv(index=False), file_name=f"{name}.csv", mime="text/csv")
