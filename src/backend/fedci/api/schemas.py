"""Request/response models for the Q1 estimation playground (docs/question.md Q1)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EstimateRequest(BaseModel):
    outcome: Literal["usmpd_sp500", "etf_day0"] = "usmpd_sp500"
    treatment: Literal["STMT", "MP1", "ME"] = "STMT"
    outcome_ticker: str = "SPY"
    features: list[str] = Field(min_length=1)
    nuisance: Literal["ridge", "enet", "rf"] = "ridge"
    alpha: float | None = None
    min_train: int | None = Field(default=None, ge=2)
    test_size: int | None = Field(default=None, ge=1)
    step: int | None = Field(default=None, ge=1)
    embargo: int = Field(default=0, ge=0)
    se: Literal["HC1", "HAC"] = "HC1"
    hac_lag: int = Field(default=4, ge=1)
    n_boot: int = Field(default=400, ge=0)
    seed: int = 0
    level: float = Field(default=0.95, gt=0.5, lt=1.0)


class EstimateResponse(BaseModel):
    params: dict
    metrics: dict
    estimates: list[dict]
    audit: list[dict]
    folds: list[dict]
    residuals: list[dict]
    influence: list[dict]
    figures: dict[str, dict]


class ExportRequest(EstimateRequest):
    figure: Literal["residualized_fit", "influence", "coefficients"] = "coefficients"
    format: Literal["png", "svg"] = "png"
