"""The contract every question package follows.

A question package (fedci/questions/qN_<slug>/) contains:
    spec.py      SPEC: QuestionSpec -- what is asked, the estimand, the currency (effective n)
    pipeline.py  run(store, **params) -- computes the answer and saves it as a Run via the
                 ResultStore. Everything it needs comes from fedci.data; everything it
                 produces goes to the store. Nothing else is written anywhere.

Keep modelling code for a question inside its package (add modules freely: features.py,
benchmark.py, models/, ...). Code shared by 2+ questions moves to fedci/eval or fedci/data.
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
