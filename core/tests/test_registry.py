import pytest

from fedcore import data
from fedcore.questions import specs


def test_five_questions_in_order():
    assert list(specs()) == ["Q1", "Q2", "Q3", "Q4", "Q5"]


@pytest.mark.parametrize("qid", ["Q1", "Q2", "Q3", "Q4", "Q5"])
def test_spec_references_are_valid(qid):
    spec = specs()[qid]
    assert set(spec.datasets) <= set(data.catalog())
    assert set(spec.depends_on) <= set(specs())
