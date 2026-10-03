"""One question at a time: the spec, whether its data is ready, and its runs so far."""
import _common as c  # noqa: I001

import streamlit as st

from fedcore.questions import specs
from fedci.config import RESULTS
from fedcore.results import ResultStore

c.page("Questions", ":material/help:")
st.title("Questions")

qs = specs()
qid = st.segmented_control("Question", list(qs), default="Q1") or "Q1"
spec = qs[qid]

st.subheader(f"{spec.id} - {spec.title}")
st.markdown(f"**{spec.question}**")
st.caption(f"Status: {spec.status}  -  Owner: {spec.owner}  -  "
           f"Code: src/backend/fedci/questions/{spec.key}/  -  Run: `fedci run {spec.id.lower()}`")

a, b = st.columns(2)
with a:
    st.markdown("**Estimand**")
    st.write(spec.estimand)
    st.markdown("**Currency (effective n)**")
    st.write(spec.currency)
with b:
    st.markdown("**Method**")
    st.write(spec.method)
    if spec.depends_on:
        st.markdown("**Depends on**")
        st.write(", ".join(spec.depends_on))
    if spec.gate:
        st.warning(spec.gate, icon=":material/lock:")

for n in spec.notes:
    st.info(n, icon=":material/priority_high:")

st.subheader("Data readiness")
cov = c.coverage().set_index("dataset")
ready = cov.loc[[d for d in spec.datasets if d in cov.index], ["layer", "grain", "rows", "start", "end", "missing_share"]]
st.dataframe(ready, width="stretch")
if spec.data_needed:
    st.markdown("**Not in the catalog yet**")
    for item in spec.data_needed:
        st.markdown(f"- :material/pending: {item}")

st.subheader("Runs")
runs = ResultStore(RESULTS).summary(spec.id)
if runs.empty:
    st.caption(f"No runs yet. Implement run() in src/backend/fedci/questions/{spec.key}/pipeline.py.")
else:
    st.caption(f"{len(runs)} runs - {int((runs.role == 'primary').sum())} primary, "
               f"{int((runs.role == 'robustness').sum())} robustness. Every extra specification is part "
               "of the multiple-testing surface.")
    st.dataframe(runs, hide_index=True, width="stretch")
