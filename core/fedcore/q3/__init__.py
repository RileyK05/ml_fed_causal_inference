"""Shared Q3 contract used by the encoder and Neural CDE projects.

Torch is optional for the rest of fedcore: import this package only from Q3 code.
"""
from fedcore.q3.baselines import BASELINES
from fedcore.q3.contracts import CHANNELS, L, HistoryEncoder, Q3Panel, SummaryEncoder
from fedcore.q3.evaluate import EvalResult, compare, run_arms, run_walk_forward, save
from fedcore.q3.features import summary_feature_names, summary_features
from fedcore.q3.heads import ShockHead
from fedcore.q3.model import ResponseModel
from fedcore.q3.synthetic import (
    CONTEXT_NAMES,
    FUNDAMENTAL_NAMES,
    make_pretrain_windows,
    make_synthetic_panel,
    sequence_pattern,
    synthetic_components,
)
from fedcore.q3.train import (
    ChannelScaler,
    FitResult,
    fit,
    per_meeting_mse,
    predict,
    resolve_device,
    validation_meetings,
)

__all__ = [
    "BASELINES",
    "CHANNELS",
    "CONTEXT_NAMES",
    "ChannelScaler",
    "EvalResult",
    "FUNDAMENTAL_NAMES",
    "FitResult",
    "HistoryEncoder",
    "L",
    "Q3Panel",
    "ResponseModel",
    "ShockHead",
    "SummaryEncoder",
    "compare",
    "fit",
    "make_pretrain_windows",
    "make_synthetic_panel",
    "per_meeting_mse",
    "predict",
    "run_arms",
    "run_walk_forward",
    "save",
    "sequence_pattern",
    "summary_feature_names",
    "summary_features",
    "synthetic_components",
    "validation_meetings",
]
