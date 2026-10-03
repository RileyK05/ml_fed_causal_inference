"""Pipelines for the questions this project answers: Q1 (built) and Q2 (parked stub).
Question definitions live in fedcore.questions."""
from importlib import import_module

from fedcore.questions import specs

PACKAGES = {"Q1": "q1_total_effect", "Q2": "q2_channels"}


def pipeline(qid: str):
    """The pipeline module for a question id, e.g. pipeline("Q1").run(store)."""
    qid = qid.upper()
    if qid not in PACKAGES:
        raise KeyError(f"{qid} is not a DML question; the dml project implements {list(PACKAGES)}")
    return import_module(f"fedci.questions.{PACKAGES[qid]}.pipeline")


__all__ = ["PACKAGES", "pipeline", "specs"]
