"""
TimesFM wrapper -- Google's time-series foundation model (PyTorch variant).

Package:        timesfm (pip install timesfm)
HuggingFace:    google/timesfm-2.0-500m-pytorch, google/timesfm-1.0-200m-pytorch, ...

API variants
------------
The library has gone through two versions of its public surface:

* Legacy:  `timesfm.TimesFm(hparams=..., checkpoint=...)` then
           `model.forecast(inputs=[np.array, ...], freq=[0, ...])`
* 2.5+:    `timesfm.TimesFM_2p5_200M_torch.from_pretrained(...)` then
           `model.compile(ForecastConfig(...))` and `model.forecast(...)`

We try the modern API first and fall back to legacy.

Inputs are **lists of 1-D numpy arrays** (context per sample), output is the
point forecast tensor `(B, horizon)` and, if requested, quantile forecasts
`(B, horizon, n_quantiles)`.

Embedding: TimesFM does not ship a native embedding call.  We fall back to a
forward hook on the final transformer block via `embed_via_hidden_states`.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from .base import (
    BaseFoundationModel,
    EmbeddingResult,
    ForecastResult,
)


_SIZE_TO_HF = {
    "small": "google/timesfm-1.0-200m-pytorch",
    "base":  "google/timesfm-1.0-200m-pytorch",   # 200M is default; use large for 500M
    "large": "google/timesfm-2.0-500m-pytorch",
}
_HF_TO_LAYERS = {
    "google/timesfm-1.0-200m-pytorch": 20,
    "google/timesfm-2.0-500m-pytorch": 50,
}


class TimesFMWrapper(BaseFoundationModel):
    name = "timesfm"
    supports_forecast = True
    supports_denoise = False
    supports_embed = False   # no native embed; could fall back via hooks

    def __init__(self) -> None:
        super().__init__()
        self._forecast_fn = None                       # closure of whichever API loaded
        self._context_len = 512
        self._horizon = 128

    def load(
        self,
        device: str = "cuda",
        model_size: str = "base",
        *,
        hf_id: Optional[str] = None,
        context_len: int = 512,
        horizon: int = 128,
        **_unused,
    ) -> None:
        try:
            import timesfm
        except ImportError as e:
            raise ImportError(
                "timesfm is required for TimesFMWrapper. "
                "Install: pip install timesfm"
            ) from e

        self.device = device
        self._context_len = context_len
        self._horizon = horizon
        hf = hf_id or _SIZE_TO_HF.get(model_size, _SIZE_TO_HF["base"])

        # Try the modern (2.5+) API first.
        try:
            from_pretrained = getattr(timesfm, "TimesFM_2p5_200M_torch", None)
            if from_pretrained is not None and hasattr(from_pretrained, "from_pretrained"):
                m = from_pretrained.from_pretrained(hf)
                cfg = timesfm.ForecastConfig(
                    max_context=context_len,
                    max_horizon=horizon,
                    normalize_inputs=True,
                )
                m.compile(cfg)
                self.model = m
                self._forecast_fn = lambda inp: m.forecast(
                    horizon=horizon, inputs=inp,
                )
                return
        except Exception:
            pass

        # Legacy API: TimesFm(hparams=..., checkpoint=...).
        n_layers = _HF_TO_LAYERS.get(hf, 20)
        hparams = timesfm.TimesFmHparams(
            backend=("gpu" if "cuda" in device else "cpu"),
            per_core_batch_size=32,
            horizon_len=horizon,
            context_len=context_len,
            num_layers=n_layers,
        )
        ckpt = timesfm.TimesFmCheckpoint(huggingface_repo_id=hf)
        m = timesfm.TimesFm(hparams=hparams, checkpoint=ckpt)
        self.model = m
        self._forecast_fn = lambda inp: m.forecast(inputs=inp, freq=[0] * len(inp))

    # ── Forecasting ───────────────────────────────────────────────────────

    @torch.no_grad()
    def forecast(self, context: np.ndarray, horizon: int) -> ForecastResult:
        if self._forecast_fn is None:
            raise RuntimeError("Call .load() before forecast().")
        # TimesFM takes a list of 1-D numpy arrays.
        B = context.shape[0]
        inputs = [context[i].astype(np.float32) for i in range(B)]
        out = self._forecast_fn(inputs)
        # Output contract differs across versions:
        #  - legacy: (point_forecast, quantile_forecast) tuple, both np arrays
        #  - modern: (point_forecast, quantile_forecast) tuple as well
        if isinstance(out, tuple):
            point, quants = out[0], out[1]
        else:
            point, quants = out, None
        point = np.asarray(point, dtype=np.float32)[:, :horizon]
        samples = None
        if quants is not None:
            quants = np.asarray(quants, dtype=np.float32)
            # Quantile tensor layout: (B, H, Q); treat quantiles as samples.
            if quants.ndim == 3:
                samples = np.transpose(quants, (0, 2, 1))[:, :, :horizon]
        return ForecastResult(point_forecast=point, samples=samples)

    # Embedding: default raises NotImplementedError; users can wire up
    # embed_via_hidden_states() per model revision if needed.
