"""Pipeline for this question. See spec.py for what it must answer and fedci/questions/base.py
for the contract. Run with:  fedci run q5"""
from fedci.data import load, meetings, event_panel  # noqa: F401
from fedci.results import ResultStore
from fedci.questions.q5_strategy.spec import SPEC


def run(store: ResultStore, **params):
    """Compute and save one run:

        store.save(SPEC, name="...", params=params, datasets=[...],
                   metrics={...}, tables={...}, figures={...}, notes="...")
    """
    raise NotImplementedError(f"{SPEC.id} pipeline not built yet")
