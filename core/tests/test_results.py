import pandas as pd
import plotly.graph_objects as go
import pytest

from fedcore.questions import specs
from fedcore.results import ResultStore


def test_save_and_reload_roundtrip(tmp_path):
    store = ResultStore(tmp_path)
    spec = specs()["Q1"]
    table = pd.DataFrame({"a": [1, 2], "b": [0.5, 0.25]})
    fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
    run = store.save(spec, "Baseline OLS", role="primary", params={"window": "day0"},
                     metrics={"beta": -0.8}, datasets=["event_study_table"],
                     tables={"coefs": table}, figures={"scatter": fig}, notes="test")
    assert run.path.parent.name == "q1_total_effect"
    (loaded,) = store.runs("Q1")
    assert loaded.manifest["metrics"] == {"beta": -0.8}
    assert len(loaded.manifest["datasets"]["event_study_table"]) == 12
    pd.testing.assert_frame_equal(loaded.table("coefs"), table)
    assert loaded.figure("scatter").data[0].y == (3, 4)


def test_runs_are_append_only(tmp_path):
    store = ResultStore(tmp_path)
    spec = specs()["Q2"]
    a = store.save(spec, "same")
    b = store.save(spec, "same")
    assert a.path != b.path and len(store.runs()) == 2


def test_role_is_validated(tmp_path):
    with pytest.raises(ValueError):
        ResultStore(tmp_path).save(specs()["Q1"], "x", role="final")
