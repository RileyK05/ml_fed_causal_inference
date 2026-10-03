"""Q1 routes: playground config and on-demand DML estimation.

Estimates are ephemeral: nothing here writes to results/ (only `fedci run q1` saves
runs). Fits are cached per request body so knob flips that were already run return
instantly; rf nuisance fits can take ~1 minute uncached.
"""
from __future__ import annotations

import functools
from collections import OrderedDict

from fastapi import APIRouter, HTTPException, Response

from fedci.api.export import render_export
from fedci.api.figures import coefficients_figure, influence_figure
from fedci.api.schemas import EstimateRequest, EstimateResponse, ExportRequest
from fedci.api.serialize import figure_json, records
from fedci.eval.dml import NUISANCES
from fedci.questions.q1_total_effect import pipeline
from fedci.questions.q1_total_effect.samples import FEATURE_INFO, OUTCOMES, SURPRISES

router = APIRouter()

_CACHE: OrderedDict[str, EstimateResponse] = OrderedDict()
_CACHE_MAX = 32


@functools.lru_cache(maxsize=1)
def _tickers() -> list[str]:
    from fedcore.data import load
    return sorted(c[4:] for c in load("event_study_table").columns if c.startswith("ret_"))


def _sample_columns(outcome: str, treatment: str, outcome_ticker: str) -> list[str]:
    from fedci.questions.q1_total_effect.samples import etf_day0_sample, usmpd_window_sample
    sample = (usmpd_window_sample(treatment) if outcome == "usmpd_sp500"
              else etf_day0_sample(treatment, outcome_ticker))
    return [c for c in sample.columns if c not in ("announcement_date", "y", "s")]


@router.get("/q1/config")
def q1_config() -> dict:
    try:
        tickers = _tickers()
    except Exception as exc:
        raise HTTPException(500, f"failed to load data: {exc}")
    return dict(
        surprises=list(SURPRISES), outcomes=list(OUTCOMES), nuisances=list(NUISANCES),
        se_types=["HC1", "HAC"],
        defaults=dict(outcome="usmpd_sp500", treatment="STMT", outcome_ticker="SPY", nuisance="ridge",
                      alpha=None, min_train=None, test_size=None, step=None, embargo=0,
                      se="HC1", hac_lag=4, n_boot=400, seed=0, level=0.95),
        features={k: list(v) for k, v in FEATURE_INFO.items()},
        feature_info={k: dict(v) for k, v in FEATURE_INFO.items()},
        tickers=tickers,
    )


@router.post("/q1/estimate", response_model=EstimateResponse)
def q1_estimate(req: EstimateRequest) -> EstimateResponse:
    return _estimate_cached(req)


def _estimate_cached(req: EstimateRequest) -> EstimateResponse:
    key = req.model_dump_json()
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    try:
        controls = _sample_columns(req.outcome, req.treatment, req.outcome_ticker)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"failed to load sample: {exc}")
    bad = [f for f in req.features if f not in controls]
    if bad:
        raise HTTPException(422, f"unknown features {bad}; allowed: {controls}")
    try:
        e = pipeline.estimate(**req.model_dump())
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    resp = EstimateResponse(
        params=e["resolved"], metrics=e["metrics"],
        estimates=records(e["estimates"]), audit=records(e["audit"]), folds=records(e["folds"]),
        residuals=records(e["residuals"]), influence=records(e["influence"]),
        figures=dict(residualized_fit=figure_json(e["figure"]),
                     influence=figure_json(influence_figure(e["influence"])),
                     coefficients=figure_json(coefficients_figure(e["estimates"], req.level))),
    )
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.popitem(last=False)
    _CACHE[key] = resp
    return resp


@router.post("/q1/export")
def q1_export(req: ExportRequest) -> Response:
    payload = req.model_dump()
    fmt = payload.pop("format")
    figure = payload.pop("figure")
    resp = _estimate_cached(EstimateRequest(**payload))
    try:
        data = render_export(figure, metrics=resp.metrics, estimates=resp.estimates,
                             residuals=resp.residuals, influence=resp.influence,
                             fmt=fmt, level=req.level)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return Response(content=data, media_type="image/png" if fmt == "png" else "image/svg+xml")
