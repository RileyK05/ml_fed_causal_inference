"""Arm definitions for the Neural CDE project.

Primary arms (``ARMS``): ``sig_ridge`` (the linear CDE) and ``cde_logsig`` (the neural
CDE driven by log-signature windows). The full-length ``cde`` arm is kept for the small
panel; at 252 RK4 steps per row it is too slow for full-panel CPU training, so the
log-signature arm is the default neural arm.

The CDE arms plug into the shared ``ResponseModel`` and trainer, so they are scored by
the same ``run_arms`` path as the baselines.
"""
from __future__ import annotations

import time

from fedcore.q3 import ResponseModel
from fedcore.q3.contracts import Q3Panel
from fedcore.q3.train import fit as train_fit
from fedcore.q3.train import predict as train_predict
from fedcore.q3.train import seed_everything

from fedcde.encoder_cde import CDELogSigEncoder, NeuralCDEEncoder
from fedcde.signature import SigRidge


class CDEModel:
    """``ResponseModel`` with a CDE encoder, trained by the shared trainer.

    The encoder is built inside ``fit`` after ``seed_everything`` so its initialisation is
    reproducible for a given seed, exactly like the summary-network baseline.
    """

    def __init__(self, seed: int = 0, encoder_cls=None, encoder_kwargs=None, **train_kwargs):
        self.seed = int(seed)
        self.encoder_cls = encoder_cls
        self.encoder_kwargs = dict(encoder_kwargs or {})
        self.train_kwargs = train_kwargs
        self.model = None
        self.scaler = None
        self.fit_result = None
        self.fit_seconds_ = 0.0

    def fit(self, panel: Q3Panel, **kwargs) -> None:
        kw = dict(self.train_kwargs)
        kw.update(kwargs)
        seed = int(kw.pop("seed", self.seed))
        seed_everything(seed)
        encoder = self.encoder_cls(**self.encoder_kwargs)
        self.model = ResponseModel(
            encoder,
            n_fund=panel.fundamentals.shape[1],
            n_ctx=panel.context.shape[1],
        )
        start = time.perf_counter()
        self.fit_result = train_fit(self.model, panel, seed=seed, **kw)
        self.fit_seconds_ = time.perf_counter() - start
        self.scaler = self.fit_result.scaler
        n_ep = len(self.fit_result.train_loss)
        print(
            f"[cde fit] seed={seed} epochs={n_ep} best={self.fit_result.best_epoch} "
            f"{self.fit_seconds_:.0f}s",
            flush=True,
        )

    def predict(self, panel: Q3Panel):
        if self.model is None or self.scaler is None:
            raise RuntimeError("fit before predict")
        return train_predict(self.model, panel, self.scaler)


def _cde_factory(encoder_cls, **enc_kw):
    def factory(seed: int) -> CDEModel:
        return CDEModel(seed=seed, encoder_cls=encoder_cls, encoder_kwargs=enc_kw)

    return factory


ARMS = {
    "sig_ridge": lambda seed: SigRidge(seed=seed),
    "cde_logsig": _cde_factory(CDELogSigEncoder, window=252, window_size=10),
}
