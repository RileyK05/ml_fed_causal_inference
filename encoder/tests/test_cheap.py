"""Gradient checks (B1-B4) and the cheap-test runner, on the synthetic stand-in."""
from __future__ import annotations

import pytest

from fedcore.q3.cheap import REGISTRY, Config, synthetic_data
from fedcore.results import ResultStore

from fedenc import cheap
from fedenc.cheap_gradients import b1, b3, b4

SMALL = Config(dev_meetings=40, inner_min_train=20, inner_test=5, outer_min_train=40, outer_test=5,
               val_meetings=8, n_boot=50, n_perm=5, min_obs=20, horizons=(21,), r1_effects=(0.0, 1.0),
               r2_min_meetings=20, r8_min_meetings=8, synthetic_firms=30, synthetic_meetings=40)


@pytest.fixture(scope="module")
def data():
    return synthetic_data(n_firms=30, n_meetings=60, seed=0)


def test_rescaling_and_dial_leave_predictions_unchanged(data):
    assert b3(data, SMALL).metrics["prediction_gap"] < 1e-4
    m = b4(data, SMALL).metrics
    assert m["max_prediction_gap"] < 1e-5
    assert m["a_head_grad_constant"]


def test_path_gradients_by_shock_bin(data):
    t = b1(data, SMALL).tables["by_shock_bin"]
    assert t["bin"].iloc[0] == "all"
    assert (t["trunk_a_norm"] > 0).all() and (t["trunk_b_norm"] >= 0).all()


def test_overnight_batch_covers_every_registered_check():
    ids = cheap.select(batch="overnight")
    assert set(ids) == set(REGISTRY) and len(ids) == 32
    assert cheap.select(ids="r1, a2") == ["A2", "R1"]
    with pytest.raises(SystemExit):
        cheap.select(ids="Z9")


def test_runner_saves_each_check_and_a_batch_summary(tmp_path, data):
    store = ResultStore(tmp_path)
    table = cheap.run(["A1", "B2", "S1"], cfg=SMALL, store=store, data=data, synthetic=True)
    assert (table["status"] == "ok").all(), table.to_string()
    names = sorted(r.manifest["name"] for r in store.runs())
    assert names == ["cheap-A1", "cheap-B2", "cheap-S1", "cheap-batch"]
    assert all(r.manifest["role"] == "exploratory" for r in store.runs())


def test_runner_records_failures_and_keeps_going(tmp_path, data, monkeypatch):
    def boom(*_):
        raise ValueError("planted failure")

    monkeypatch.setitem(REGISTRY, "A1", REGISTRY["A1"].__class__("A1", "boom", "descriptive", boom))
    table = cheap.run(["A1", "A3"], cfg=SMALL, store=ResultStore(tmp_path), data=data, synthetic=True)
    assert table.set_index("check").loc["A1", "status"] == "failed"
    assert table.set_index("check").loc["A3", "status"] == "ok"


def test_dry_run_loads_nothing(capsys):
    cheap.main(["run", "--batch", "overnight", "--dry-run"])
    out = capsys.readouterr().out
    assert "checks:" in out and "fitted checks" in out
