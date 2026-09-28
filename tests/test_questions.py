import pytest

from fedci import data
from fedci.questions import pipeline, specs
from fedci.results import ResultStore

PARKED = ["Q2", "Q3", "Q4", "Q5"]  # not built yet; Q1 is (tests below)


def test_five_questions_in_order():
    assert list(specs()) == ["Q1", "Q2", "Q3", "Q4", "Q5"]


@pytest.mark.parametrize("qid", ["Q1", "Q2", "Q3", "Q4", "Q5"])
def test_spec_references_are_valid(qid):
    spec = specs()[qid]
    assert set(spec.datasets) <= set(data.catalog())
    assert set(spec.depends_on) <= set(specs())


@pytest.mark.parametrize("qid", PARKED)
def test_parked_pipelines_are_stubs_until_built(qid, tmp_path):
    with pytest.raises(NotImplementedError):
        pipeline(qid).run(ResultStore(tmp_path))


@pytest.mark.parametrize("outcome", ["usmpd_sp500", "etf_day0"])
def test_q1_pipeline_saves_runs(outcome, tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("statsmodels")
    store = ResultStore(tmp_path)
    run = pipeline("Q1").run(store, outcome=outcome, n_boot=20, min_train=12, test_size=6)
    assert run.manifest["role"] == "primary"
    assert run.manifest["metrics"]["n_meetings"] > 0
    assert run.manifest["params"]["test_size"] == 6
    (loaded,) = store.runs("Q1")
    assert set(loaded.manifest["tables"]) == {"estimates", "audit", "folds", "residuals", "influence"}
    assert loaded.table("estimates")["theta"].notna().all()
