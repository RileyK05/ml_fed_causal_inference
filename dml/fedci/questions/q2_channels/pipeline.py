"""Pipeline for Q2 (parked). Definition: fedcore/questions/q2_channels.py.
Run with:  fedci run q2"""
from fedcore.data import load, meetings, event_panel  # noqa: F401
from fedcore.results import ResultStore
from fedcore.questions.q2_channels import SPEC


def run(store: ResultStore, **params):
    """Compute and save one run:

        store.save(SPEC, name="...", params=params, datasets=[...],
                   metrics={...}, tables={...}, figures={...}, notes="...")
    """
    raise NotImplementedError(f"{SPEC.id} pipeline not built yet")
