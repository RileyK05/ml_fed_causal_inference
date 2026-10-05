"""Ad-hoc SQL over every catalog dataset (DuckDB over the catalog files; reads only core/data, cannot write)."""
import _common as c  # noqa: I001

import streamlit as st

c.page("SQL console", ":material/terminal:")
st.title("SQL console")
st.caption("Each catalog dataset is a view with the same name. Same engine as fedcore.data.query().")

EXAMPLES = {
    "Biggest statement surprises": "SELECT Date, STMT, PC, ME\nFROM mps_surprises\nORDER BY ABS(STMT) DESC\nLIMIT 15",
    "Surprise vs same-day sector returns": (
        "SELECT m.announcement_date, s.STMT, r.SPY, r.XLF, r.XLU\n"
        "FROM fomc_meetings m\n"
        "JOIN mps_surprises s ON s.Date = m.announcement_date\n"
        "JOIN etf_returns  r ON r.date = m.announcement_date\n"
        "ORDER BY 1"
    ),
    "Unscheduled actions (separate treatment)": "SELECT Date, MP1, MP2\nFROM usmpd_statements\nWHERE Unscheduled = 1\nORDER BY Date",
    "Meetings per year": "SELECT YEAR(Date) AS year, COUNT(*) AS meetings\nFROM usmpd_statements\nWHERE Unscheduled = 0\nGROUP BY 1 ORDER BY 1",
}

side, main = st.columns([1, 3])
with side:
    st.markdown("**Tables**")
    for t in c.data.tables():
        with st.expander(t):
            st.caption(c.data.get(t).description)
            cols = c.data.query(f'DESCRIBE "{t}"')[["column_name", "column_type"]]
            st.dataframe(cols, hide_index=True, width="stretch")

with main:
    ex = st.selectbox("Start from an example", ["(blank)", *EXAMPLES])
    sql = st.text_area("SQL", EXAMPLES.get(ex, ""), height=180)
    if st.button("Run", type="primary", icon=":material/play_arrow:") and sql.strip():
        try:
            out = c.data.query(sql)
        except Exception as e:  # show the DuckDB message, don't crash the page
            st.error(str(e))
        else:
            st.caption(f"{len(out):,} rows")
            st.dataframe(out, hide_index=True, width="stretch", height=480)
            st.download_button("Download CSV", out.to_csv(index=False), file_name="query.csv", mime="text/csv")
