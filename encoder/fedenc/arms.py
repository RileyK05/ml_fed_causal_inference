"""Q3 patch-transformer arms.

ARMS follow the shared arm protocol (``fit(panel, **kwargs)`` / ``predict``), so they
plug into ``run_arms`` in ``fedcore.q3``. ``init`` picks scratch vs pre-trained weights;
``adapt`` picks how much of the encoder fine-tunes (full / frozen / top block only).
On real data a pre-trained checkpoint may only serve test folds that start after its
corpus ends.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from fedcore.q3 import Q3Panel, ResponseModel, make_pretrain_windows
from fedcore.q3.train import fit as train_fit
from fedcore.q3.train import predict as train_predict
from fedcore.q3.train import resolve_device, seed_everything, validation_meetings

from fedenc.encoder_transformer import PatchTransformerEncoder
from fedenc.pretrain import PretrainResult, pretrain
from fedenc.checkpoint import EncoderCheckpoint, load_checkpoint
from fedenc.train import fit_transferred

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
    device: str | None = None,
) -> PretrainResult:
    """Legacy synthetic diagnostic, explicit calls only; cache in memory, never write files.

    This is training and requires chat authorization outside tiny unit-test smoke fits.
    Real and synthetic TxArms now load explicit saved checkpoints instead of calling this.
    """
    key = (int(n_windows), int(seed), int(epochs), float(lr))
    if key in _PRETRAIN_CACHE:
        return _PRETRAIN_CACHE[key]
    seed_everything(seed)
    windows, mask = make_pretrain_windows(n_windows, seed=seed)
    result = pretrain(
        PatchTransformerEncoder(), windows, mask, epochs=epochs, lr=lr, seed=seed,
        device=resolve_device(device),
    )
    result.encoder.cpu()  # a GPU-trained cache must load on a CPU-only machine
    _PRETRAIN_CACHE[key] = result
    return result


class TxArm:
    """One transformer arm.

    ``init``    "scratch" | "pretrained"
    ``adapt``   "full" (all weights) | "frozen" (freeze_encoder) | "top" (unfreeze top block only)
    ``history_only``  zero fundamentals and context at fit and predict (E4 ablation)
    ``device``  None = CUDA when available, else CPU
    """

    def __init__(
        self,
        seed: int = 0,
        init: str = "scratch",
        adapt: str = "full",
        history_only: bool = False,
        encoder_kwargs: dict | None = None,
        pretrain_kwargs: dict | None = None,
        checkpoint: str | Path | EncoderCheckpoint | None = None,
        device: str | None = None,
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
        if self.pretrain_kwargs:
            raise ValueError("pretrain_kwargs no longer starts training inside arms; pass a checkpoint")
        self.checkpoint = checkpoint
        self.loaded_checkpoint: EncoderCheckpoint | None = None
        self.device = device
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
        if self.init == "pretrained" and self.checkpoint is None:
            raise ValueError("pretrained arms require an explicit pre-cutoff checkpoint; no automatic pretraining")
        if self.checkpoint is not None:
            checkpoint = self.checkpoint if isinstance(self.checkpoint, EncoderCheckpoint) else load_checkpoint(self.checkpoint)
            self.loaded_checkpoint = checkpoint
            if any(checkpoint.config.get(k) != v for k, v in self.encoder_kwargs.items()):
                raise ValueError("encoder_kwargs must match checkpoint architecture")
            # Scratch uses the same ordinary-day scaler/architecture, but random
            # weights. This isolates initialization from normalization changes.
            enc = checkpoint.encoder() if self.init == "pretrained" else PatchTransformerEncoder(**checkpoint.config)
            if self.init == "pretrained" and self.adapt == "top":
                enc.freeze([enc.proj, enc.pos, *enc.blocks.layers[:-1]])
        else:
            enc = PatchTransformerEncoder(**self.encoder_kwargs)
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
        if self.loaded_checkpoint is not None:
            # Pretraining validation also consumes labels, so precede the *inner*
            # early-stopping block, not just the outer test. Check before any fit.
            val_m = validation_meetings(panel, kw.get("val_frac", 0.15))
            self.loaded_checkpoint.require_before(val_m.min())
        self.model = ResponseModel(
            encoder, n_fund=panel.fundamentals.shape[1], n_ctx=panel.context.shape[1]
        ).to(resolve_device(self.device))
        if self.loaded_checkpoint is None:
            self.fit_result = train_fit(self.model, self._prep(panel), seed=seed, **kw)
        else:
            self.fit_result = fit_transferred(
                self.model, self._prep(panel), self.loaded_checkpoint.scaler, seed=seed, **kw
            )
        self.scaler = self.fit_result.scaler

    def predict(self, panel: Q3Panel):
        if self.model is None or self.scaler is None:
            raise RuntimeError("fit the arm before predict")
        if self.loaded_checkpoint is not None:
            self.loaded_checkpoint.require_before(panel.meetings().min())
        return train_predict(self.model, self._prep(panel), self.scaler)


ARMS = {
    "tx_scratch": lambda seed: TxArm(seed=seed, init="scratch", **TRAIN_BUDGET),
    "tx_pre_frozen": lambda seed: TxArm(seed=seed, init="pretrained", adapt="frozen", **TRAIN_BUDGET),
    "tx_pre_top": lambda seed: TxArm(seed=seed, init="pretrained", adapt="top", **TRAIN_BUDGET),
    "tx_pre_full": lambda seed: TxArm(seed=seed, init="pretrained", adapt="full", **TRAIN_BUDGET),
}


def make_arms(checkpoint, *, train_kwargs=None):
    """Bind an explicit checkpoint to the transfer arms; preserve its architecture for scratch."""
    cp = checkpoint if isinstance(checkpoint, EncoderCheckpoint) else load_checkpoint(checkpoint)
    budget = {**TRAIN_BUDGET, **(train_kwargs or {})}
    return {
        "tx_scratch": lambda seed: TxArm(seed=seed, checkpoint=cp, **budget),
        "tx_pre_frozen": lambda seed: TxArm(seed=seed, init="pretrained", adapt="frozen", checkpoint=cp, **budget),
        "tx_pre_top": lambda seed: TxArm(seed=seed, init="pretrained", adapt="top", checkpoint=cp, **budget),
        "tx_pre_full": lambda seed: TxArm(seed=seed, init="pretrained", adapt="full", checkpoint=cp, **budget),
    }
