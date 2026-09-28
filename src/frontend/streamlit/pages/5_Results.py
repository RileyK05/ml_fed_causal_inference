"""Browse saved runs: manifest, metrics, tables, figures -- exactly as the pipeline saved them."""
import _common as c  # noqa: I001

import streamlit as st

from fedci.results import ROLES, ResultStore

c.page("Results", ":material/lab_profile:")
st.title("Results")

store = ResultStore()
all_runs = store.runs()
if not all_runs:
    st.info("No runs saved yet. Pipelines save through ResultStore.save(); see src/backend/fedci/results/store.py.",
            icon=":material/info:")
    st.stop()

f1, f2 = st.columns(2)
q = f1.multiselect("Question", sorted({r.manifest["question"] for r in all_runs}))
roles = f2.multiselect("Role", list(ROLES))
runs = [r for r in all_runs if (not q or r.manifest["question"] in q) and (not roles or r.manifest["role"] in roles)]
if not runs:
    st.caption("No runs match the filters.")
    st.stop()

run = st.selectbox("Run", runs, format_func=lambda r: f"{r.manifest['question']}  {r.run_id}  ({r.manifest['role']})")
m = run.manifest

st.subheader(m["name"])
st.caption(f"{m['question']} - {m['role']} - {m['created']} - {run.path}")
if m.get("notes"):
    st.write(m["notes"])

if m["metrics"]:
    cols = st.columns(min(4, len(m["metrics"])))
    for i, (k, v) in enumerate(m["metrics"].items()):
        cols[i % len(cols)].metric(k, f"{v:.4g}" if isinstance(v, float) else str(v))

for f in m["figures"]:
    c.show(c.style(run.figure(f)))

for t in m["tables"]:
    st.markdown(f"**{t}**")
    st.dataframe(run.table(t), hide_index=True, width="stretch")

with st.expander("Params and data fingerprints"):
    st.json({"params": m["params"], "datasets": m["datasets"], "env": m.get("env", {})})
