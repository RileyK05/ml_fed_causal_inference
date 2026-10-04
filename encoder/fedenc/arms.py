"""Q3 patch-transformer arms.

ARMS follow the shared arm protocol (``fit(panel, **kwargs)`` / ``predict``), so they
plug into ``run_arms`` in ``fedcore.q3``. ``init`` picks scratch vs pre-trained weights;
``adapt`` picks how much of the encoder fine-tunes (full / frozen / top block only).
On real data a pre-trained checkpoint may only serve test folds that start after its
corpus ends.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fedcore.q3 import Q3Panel, ResponseModel, make_pretrain_windows
from fedcore.q3.train import fit as train_fit
from fedcore.q3.train import predict as train_predict
from fedcore.q3.train import seed_everything

from fedenc.encoder_transformer import PatchTransformerEncoder
from fedenc.pretrain import PretrainResult, pretrain

PRETRAIN_SEED = 7777
PRETRAIN_WINDOWS = 20000
PRETRAIN_EPOCHS = 30
PRETRAIN_LR = 1e-3

# Training budget for every transformer arm.
TRAIN_BUDGET = {"max_epochs": 40, "patience": 10}

_PRETRAIN_CACHE: dict[tuple, PretrainResult] = {}


def get_pretrained(
    n_windows: int = PRETRAIN_WINDOWS,
    seed: int = PRETRAIN_SEED,
    epochs: int = PRETRAIN_EPOCHS,
    lr: float = PRETRAIN_LR,
) -> PretrainResult:
    """Pre-train once and cache. Deterministic: same parameters, same checkpoint.

    Worker processes share the checkpoint through a scratch file in the system
    temp directory (``FEDENC_PRETRAIN_CACHE`` overrides the path); compute cache only.
    """
    key = (int(n_windows), int(seed), int(epochs), float(lr))
    if key in _PRETRAIN_CACHE:
        return _PRETRAIN_CACHE[key]
    default = Path(tempfile.gettempdir()) / f"fedenc_pretrain_{key[0]}_{key[1]}_{key[2]}_{key[3]:g}.pt"
    cache = Path(os.environ.get("FEDENC_PRETRAIN_CACHE") or default)
    if cache.exists():
        payload = torch.load(cache, weights_only=False)
        if payload.get("key") == key:
            _PRETRAIN_CACHE[key] = payload["result"]
            return _PRETRAIN_CACHE[key]
    windows, mask = make_pretrain_windows(n_windows, seed=seed)
    result = pretrain(PatchTransformerEncoder(), windows, mask, epochs=epochs, lr=lr, seed=seed)
    try:
        torch.save({"key": key, "result": result}, cache)
    except OSError:
        pass
    _PRETRAIN_CACHE[key] = result
    return result


class TxArm:
    """One transformer arm.

    ``init``    "scratch" | "pretrained"
    ``adapt``   "full" (all weights) | "frozen" (freeze_encoder) | "top" (unfreeze top block only)
    ``history_only``  zero fundamentals and context at fit and predict (E4 ablation)
    """

    def __init__(
        self,
        seed: int = 0,
        init: str = "scratch",
        adapt: str = "full",
        history_only: bool = False,
        encoder_kwargs: dict | None = None,
        pretrain_kwargs: dict | None = None,
        **train_kwargs,
    ):
        if init not in ("scratch", "pretrained"):
            raise ValueError(f"unknown init {init!r}")
        if adapt not in ("full", "frozen", "top"):
            raise ValueError(f"unknown adapt {adapt!r}")
        self.seed = int(seed)
        self.init = init
        self.adapt = adapt
        self.history_only = bool(history_only)
        self.encoder_kwargs = dict(encoder_kwargs or {})
        self.pretrain_kwargs = dict(pretrain_kwargs or {})
        self.train_kwargs = dict(train_kwargs)
        self.model: nn.Module | None = None
        self.scaler = None
        self.fit_result = None

    def _prep(self, panel: Q3Panel) -> Q3Panel:
        if not self.history_only:
            return panel
        zero = np.zeros_like(panel.fundamentals)
        zctx = np.zeros_like(panel.context)
        return Q3Panel(
            meeting=panel.meeting,
            firm_id=panel.firm_id,
            history=panel.history,
            history_mask=panel.history_mask,
            fundamentals=zero,
            fundamentals_mask=panel.fundamentals_mask,
            context=zctx,
            shock=panel.shock,
            target=panel.target,
            a_true=panel.a_true,
            b_true=panel.b_true,
        )

    def _encoder(self) -> PatchTransformerEncoder:
        enc = PatchTransformerEncoder(**self.encoder_kwargs)
        if self.init == "pretrained":
            enc.load_state_dict(get_pretrained(**self.pretrain_kwargs).state_dict)
            if self.adapt == "top":
                enc.freeze([enc.proj, enc.pos, *enc.blocks.layers[:-1]])
        return enc

    def fit(self, panel: Q3Panel, **kwargs) -> None:
        kw = dict(self.train_kwargs)
        kw.update(kwargs)
        seed = int(kw.pop("seed", self.seed))
        seed_everything(seed)
        if self.adapt == "frozen":
            kw.setdefault("freeze_encoder", True)
        if self.init == "pretrained" and self.adapt == "full":
            kw.setdefault("encoder_lr", float(kw.get("lr", 1e-3)) / 10.0)
        encoder = self._encoder()
        self.model = ResponseModel(
            encoder, n_fund=panel.fundamentals.shape[1], n_ctx=panel.context.shape[1]
        )
        self.fit_result = train_fit(self.model, self._prep(panel), seed=seed, **kw)
        self.scaler = self.fit_result.scaler

    def predict(self, panel: Q3Panel):
        if self.model is None or self.scaler is None:
            raise RuntimeError("fit the arm before predict")
        return train_predict(self.model, self._prep(panel), self.scaler)


ARMS = {
    "tx_scratch": lambda seed: TxArm(seed=seed, init="scratch", **TRAIN_BUDGET),
    "tx_pre_frozen": lambda seed: TxArm(seed=seed, init="pretrained", adapt="frozen", **TRAIN_BUDGET),
    "tx_pre_top": lambda seed: TxArm(seed=seed, init="pretrained", adapt="top", **TRAIN_BUDGET),
    "tx_pre_full": lambda seed: TxArm(seed=seed, init="pretrained", adapt="full", **TRAIN_BUDGET),
}
