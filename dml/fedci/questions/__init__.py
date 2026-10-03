"""Registry of the five project questions (docs/question.md)."""
from importlib import import_module

from fedci.questions.base import STATUSES, QuestionSpec

PACKAGES = ["q1_total_effect", "q2_channels", "q3_susceptibility", "q4_gap", "q5_strategy"]


def specs() -> dict[str, QuestionSpec]:
    """{"Q1": QuestionSpec, ...} in question order."""
    out = {}
    for pkg in PACKAGES:
        spec = import_module(f"fedci.questions.{pkg}.spec").SPEC
        out[spec.id] = spec
    return out


def pipeline(qid: str):
    """The pipeline module for a question id, e.g. pipeline("Q1").run(store)."""
    spec = specs()[qid.upper()]
    return import_module(f"fedci.questions.{spec.key}.pipeline")


__all__ = ["QuestionSpec", "STATUSES", "specs", "pipeline", "PACKAGES"]
