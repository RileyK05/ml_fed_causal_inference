"""Portable encoder + scaler checkpoints, written exclusively through ResultStore.

Tensor values are JSON arrays in a CSV table: no pickle, temporary cache, raw
WRDS observations or second artifact writer is needed. The temporary head is discarded.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fedcore.q3.contracts import CHANNELS
from fedcore.questions import specs
from fedenc.encoder_transformer import PatchTransformerEncoder
from fedenc.pretrain_return import OBJECTIVES

SCHEMA = "fedenc-history-v1"


def _digest(state, scaler):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().numpy()
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    for name in ("hist_median", "hist_iqr"):
        digest.update(np.asarray(scaler[name], dtype=np.float64).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class EncoderCheckpoint:
    path: Path
    state_dict: dict
    scaler: dict
    config: dict
    params: dict
    fingerprints: dict
    digest: str

    @property
    def available_through(self):
        return pd.Timestamp(self.params["available_through"])

    def encoder(self):
        encoder = PatchTransformerEncoder(**self.config)
        encoder.load_state_dict(self.state_dict, strict=True)
        return encoder

    def require_before(self, date):
        if self.available_through >= pd.Timestamp(date):
            raise ValueError(
                f"checkpoint available through {self.available_through.date()} must precede "
                f"validation/prediction date {pd.Timestamp(date).date()}"
            )


def save_checkpoint(result, store, *, datasets=(), role="exploratory", name=None):
    """Append one Q3 pretraining run; save no temporary prediction head."""
    state = result.state_dict
    tensor_table = pd.DataFrame([
        {"name": key, "shape": json.dumps(list(value.shape)),
         "values": json.dumps(value.detach().cpu().reshape(-1).tolist(), allow_nan=False)}
        for key, value in state.items()
    ])
    scaler_table = pd.DataFrame({
        "channel": CHANNELS, "median": result.scaler["hist_median"], "iqr": result.scaler["hist_iqr"],
    })
    params = {**result.params, "checkpoint_schema": SCHEMA,
              "encoder_config": result.encoder.get_config(), "channels": list(CHANNELS),
              "checkpoint_digest": _digest(state, result.scaler)}
    return store.save(
        specs()["Q3"], name or f"pretrain-{params['objective']}", role=role,
        params=params, metrics={"best_epoch": result.best_epoch, "best_val_loss": result.best_val_loss},
        datasets=list(datasets), tables={"encoder-state": tensor_table, "history-scaler": scaler_table,
                                       "curves": result.curves},
        notes="History-only pretraining. Head discarded. Encoder and history scaler retained. "
              "Checkpoint availability includes pretraining validation labels. No Fed sensitivity claim.",
    )


def load_checkpoint(path) -> EncoderCheckpoint:
    path = Path(path)
    if path.name == "manifest.json":
        path = path.parent
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    params = manifest["params"]
    if manifest.get("question") != "Q3" or params.get("checkpoint_schema") != SCHEMA:
        raise ValueError("not a supported Q3 encoder checkpoint")
    if params.get("objective") not in OBJECTIVES or params.get("channels") != list(CHANNELS):
        raise ValueError("checkpoint objective/channels do not match this encoder")
    if params.get("return_units") != "percent":
        raise ValueError("checkpoint must use percent returns")
    end = pd.Timestamp(params["available_through"])
    if pd.isna(end) or end >= pd.Timestamp(params["cutoff"]):
        raise ValueError("invalid checkpoint availability/cutoff")
    for key in ("train_target_end", "val_target_end"):
        if pd.Timestamp(params[key]) > end:
            raise ValueError("checkpoint availability excludes consumed labels")
    config = params["encoder_config"]
    prototype = PatchTransformerEncoder(**config)
    expected = prototype.state_dict()
    state = {}
    table = pd.read_csv(path / "tables" / "encoder-state.csv")
    for row in table.itertuples(index=False):
        if row.name in state or row.name not in expected:
            raise ValueError("unexpected/duplicate checkpoint tensor")
        shape = json.loads(row.shape)
        if shape != list(expected[row.name].shape):
            raise ValueError(f"checkpoint tensor shape mismatch: {row.name}")
        value = torch.tensor(json.loads(row.values), dtype=expected[row.name].dtype).reshape(shape)
        if not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite checkpoint tensor")
        state[row.name] = value
    prototype.load_state_dict(state, strict=True)
    scale = pd.read_csv(path / "tables" / "history-scaler.csv", float_precision="round_trip")
    if scale.channel.tolist() != list(CHANNELS):
        raise ValueError("checkpoint scaler channel order mismatch")
    scaler = {"hist_median": scale['median'].to_numpy(dtype=np.float64),
              "hist_iqr": scale['iqr'].to_numpy(dtype=np.float64)}
    if not all(np.isfinite(v).all() for v in scaler.values()) or (scaler["hist_iqr"] <= 0).any():
        raise ValueError("invalid checkpoint scaler")
    digest = _digest(state, scaler)
    if digest != params.get("checkpoint_digest"):
        raise ValueError("checkpoint weights/scaler failed digest validation")
    return EncoderCheckpoint(path, state, scaler, config, params, manifest["datasets"], digest)
