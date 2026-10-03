import pytest

from fedci.questions import PACKAGES, pipeline
from fedcore.results import ResultStore


def test_dml_owns_q1_and_q2():
    assert list(PACKAGES) == ["Q1", "Q2"]
    with pytest.raises(KeyError):
        pipeline("Q3")


def test_parked_q2_is_a_stub(tmp_path):
    with pytest.raises(NotImplementedError):
        pipeline("Q2").run(ResultStore(tmp_path))


@pytest.mark.parametrize("outcome", ["usmpd_sp500", "etf_day0"])
def test_q1_pipeline_saves_runs(outcome, tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("statsmodels")
    store = ResultStore(tmp_path)
    run = pipeline("Q1").run(store, outcome=outcome, n_boot=20, min_train=12, test_size=6)
    assert run.manifest["role"] == "primary"
    assert run.manifest["metrics"]["n_meetings"] > 0
    assert run.manifest["params"]["test_size"] == 6
