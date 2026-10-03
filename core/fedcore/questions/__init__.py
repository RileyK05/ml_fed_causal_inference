"""The five project questions (docs/question.md). Definitions only; each project implements
the pipelines for its own questions (Q1 in dml/, Q3 in encoder/ and cde/)."""
from importlib import import_module

from fedcore.questions.base import STATUSES, QuestionSpec

MODULES = ["q1_total_effect", "q2_channels", "q3_susceptibility", "q4_gap", "q5_strategy"]


def specs() -> dict[str, QuestionSpec]:
    """{"Q1": QuestionSpec, ...} in question order."""
    out = {}
    for mod in MODULES:
        spec = import_module(f"fedcore.questions.{mod}").SPEC
        out[spec.id] = spec
    return out


__all__ = ["QuestionSpec", "STATUSES", "specs", "MODULES"]
