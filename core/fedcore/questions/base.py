"""QuestionSpec: what a question asks, its estimand and its currency (effective n).

Definitions live here (fedcore/questions/qN_<slug>.py). Pipelines live in the project that
answers the question and follow one contract: run(store, **params) computes the answer and
saves it as a Run via a ResultStore rooted in that project's results/ folder. Inputs come from
fedcore.data; nothing else is written anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field

STATUSES = ("not started", "in progress", "blocked", "done")


@dataclass(frozen=True)
class QuestionSpec:
    id: str                         # "Q1"
    slug: str                       # "total_effect" -- also the results/ folder name
    title: str                      # short name
    question: str                   # the question, verbatim from docs/question.md
    estimand: str
    currency: str                   # effective sample size: what n really is
    method: str
    owner: str
    status: str = "not started"
    depends_on: tuple[str, ...] = ()
    datasets: tuple[str, ...] = ()  # catalog names this question reads today
    data_needed: tuple[str, ...] = ()  # data not in the catalog yet (docs/question.md "Data moves")
    gate: str = ""                  # condition that must hold before this question is built
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"{self.id}: status must be one of {STATUSES}")

    @property
    def key(self) -> str:
        return f"{self.id.lower()}_{self.slug}"
