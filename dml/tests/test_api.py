"""API tests for the Q1 playground and the read-only runs browser (docs/plans/frontend-playground.md)."""
from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from fedci.api import create_app  # noqa: E402

pytest.importorskip("sklearn")
pytest.importorskip("statsmodels")


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def test_config(client):
    r = client.get("/api/q1/config")
    assert r.status_code == 200
    body = r.json()
    assert len(body["features"]["usmpd_sp500"]) == 6
    assert body["tickers"]
    assert body["defaults"]["outcome"] == "usmpd_sp500"


def test_estimate_returns_full_response(client):
    r = client.post("/api/q1/estimate", json=dict(
        outcome="usmpd_sp500", treatment="STMT", features=["y1_level", "st_prev"],
        nuisance="ridge", min_train=30, test_size=30, n_boot=50, se="HC1"))
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"params", "metrics", "estimates", "audit", "folds",
                         "residuals", "influence", "figures"}
    assert body["metrics"]["theta"] == body["metrics"]["theta"]  # finite
    assert len(body["residuals"]) == body["metrics"]["n_meetings"]
    assert len(body["estimates"]) == 4
    assert set(body["figures"]) == {"residualized_fit", "influence", "coefficients"}


def test_feature_validation(client):
    base = dict(outcome="usmpd_sp500", treatment="STMT", min_train=30, test_size=30,
                n_boot=20, se="HC1")
    assert client.post("/api/q1/estimate", json=dict(base, features=["nope"])).status_code == 422
    assert client.post("/api/q1/estimate", json=dict(base, features=[])).status_code == 422
    r = client.post("/api/q1/estimate", json=dict(base, features=["y1_level"]))
    assert r.status_code == 200


def test_hac_se_label(client):
    r = client.post("/api/q1/estimate", json=dict(
        outcome="usmpd_sp500", treatment="STMT", features=["y1_level", "st_prev"],
        min_train=30, test_size=30, n_boot=0, se="HAC", hac_lag=2))
    assert r.status_code == 200
    assert r.json()["metrics"]["se_type"] == "HAC(2)"


def test_no_bootstrap_row_when_disabled(client):
    r = client.post("/api/q1/estimate", json=dict(
        outcome="usmpd_sp500", treatment="STMT", features=["y1_level", "st_prev"],
        min_train=30, test_size=30, n_boot=0, se="HC1"))
    assert r.status_code == 200
    assert len(r.json()["estimates"]) == 3


def test_bad_fold_geometry_is_422(client):
    r = client.post("/api/q1/estimate", json=dict(
        outcome="usmpd_sp500", treatment="STMT", features=["y1_level", "st_prev"],
        min_train=30, test_size=30, n_boot=0, embargo=999))
    assert r.status_code == 422
    assert "min_train" in r.json()["detail"]


def test_runs_list(client):
    r = client.get("/api/runs")
    assert r.status_code == 200
    assert isinstance(r.json()["runs"], list)
    assert len(r.json()["runs"]) >= 1


def test_run_detail(client):
    runs = client.get("/api/runs").json()["runs"]
    first = runs[0]
    r = client.get(f"/api/runs/{first['question']}/{first['run_id']}")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"manifest", "tables", "figures"}
    assert body["figures"]["residualized_fit"]["data"]
    assert body["figures"]["residualized_fit"]["layout"]


def test_export_static_figures(client):
    body = dict(outcome="usmpd_sp500", treatment="STMT", features=["y1_level", "st_prev"],
                min_train=30, test_size=30, n_boot=0)
    for kind in ("residualized_fit", "influence", "coefficients"):
        r = client.post("/api/q1/export", json=dict(body, figure=kind, format="png"))
        assert r.status_code == 200, kind
        assert r.headers["content-type"] == "image/png"
        assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
    r = client.post("/api/q1/export", json=dict(body, figure="coefficients", format="svg"))
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/svg+xml"
    assert b"<svg" in r.content[:2000]
    assert client.post("/api/q1/export", json=dict(body, figure="nope", format="png")).status_code == 422
    assert client.post("/api/q1/export", json=dict(body, figure="coefficients", format="pdf")).status_code == 422


def test_run_export(client):
    first = client.get("/api/runs").json()["runs"][0]
    for kind in ("residualized_fit", "influence", "coefficients"):
        r = client.get(f"/api/runs/{first['question']}/{first['run_id']}/export/{kind}?format=png")
        assert r.status_code == 200, kind
        assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
    r = client.get(f"/api/runs/{first['question']}/{first['run_id']}/export/coefficients?format=svg")
    assert r.status_code == 200 and b"<svg" in r.content[:2000]
