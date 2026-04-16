"""
Lag-Llama wrapper -- ServiceNow/ML4TS' decoder-only probabilistic forecaster.

Installation: pip install -e 'git+https://github.com/time-series-foundation-models/lag-llama.git#egg=lag-llama'
Checkpoint:   huggingface-cli download time-series-foundation-models/Lag-Llama lag-llama.ckpt

Lag-Llama uses GluonTS-style estimators; we call
`LagLlamaEstimator(ckpt_path=..., prediction_length=..., context_length=...)`
then `.create_predictor()`.  Inference is identical in shape to MOIRAI.

Embedding: Lag-Llama is auto-regressive and does not export a pooled
representation by default.  We leave supports_embed=False for now;
hidden-state hooks are feasible but require version-specific layer access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .base import (
    BaseFoundationModel,
    ForecastResult,
)


class LagLlamaWrapper(BaseFoundationModel):
    name = "lagllama"
    supports_forecast = True
    supports_denoise = False
    supports_embed = False

    def __init__(self) -> None:
        super().__init__()
        self._predictor = None
        self._num_samples = 20

    def load(
        self,
        device: str = "cuda",
        model_size: str = "base",
        *,
        ckpt_path: Optional[str] = None,
        context_len: int = 512,
        horizon: int = 128,
        num_samples: int = 10,        # lowered from 20: GluonTS iterator is slow per sample
        **_unused,
    ) -> None:
        try:
            from lag_llama.gluon.estimator import LagLlamaEstimator
        except ImportError as e:
            raise ImportError(
                "lag_llama is required for LagLlamaWrapper. "
                "Install: pip install -e "
                "'git+https://github.com/time-series-foundation-models/lag-llama.git#egg=lag-llama'"
            ) from e
        self.device = device
        self._num_samples = num_samples

        # Default path: look in HF cache, shared NFS checkpoints, or current dir
        if ckpt_path is None:
            candidates = [
                Path.home() / ".cache" / "huggingface" / "hub" / "models--time-series-foundation-models--Lag-Llama" / "snapshots",
                Path("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/checkpoints/lag-llama/lag-llama.ckpt"),
                Path("checkpoints/lag-llama/lag-llama.ckpt"),
                Path("lag-llama.ckpt"),
            ]
            ckpt_path = None
            for c in candidates:
                if c.exists():
                    if c.is_dir():
                        ckpts = list(c.glob("*/lag-llama.ckpt"))
                        if ckpts:
                            ckpt_path = str(ckpts[0])
                            break
                    else:
                        ckpt_path = str(c)
                        break
            if ckpt_path is None:
                raise FileNotFoundError(
                    "Lag-Llama checkpoint not found. Download via:\n"
                    "  huggingface-cli download time-series-foundation-models/Lag-Llama lag-llama.ckpt\n"
                    "and pass ckpt_path=... on load()."
                )

        ckpt_info = torch.load(ckpt_path, map_location=device, weights_only=False)
        estimator_args = ckpt_info["hyper_parameters"]["model_kwargs"]

        estimator = LagLlamaEstimator(
            ckpt_path=ckpt_path,
            prediction_length=horizon,
            context_length=context_len,
            # Reuse architecture hyperparameters from the checkpoint
            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            scaling=estimator_args["scaling"],
            time_feat=estimator_args["time_feat"],
            num_parallel_samples=num_samples,
            device=torch.device(device),
        )
        self._predictor = estimator.create_predictor(
            estimator.create_transformation(),
            estimator.create_lightning_module(),
        )
        self.model = estimator

    # ── Forecasting ───────────────────────────────────────────────────────

    # Class-level batch counter for coarse progress logging (no per-sample spam)
    _batch_counter = 0

    @torch.no_grad()
    def forecast(self, context: np.ndarray, horizon: int) -> ForecastResult:
        if self._predictor is None:
            raise RuntimeError("Call .load() before forecast().")
        ds = self._to_gluonts(context, freq="s")
        it = self._predictor.predict(ds, num_samples=self._num_samples)
        samples_list = []
        for fc in it:
            samples_list.append(np.asarray(fc.samples, dtype=np.float32))
        samples = np.stack(samples_list, axis=0)[:, :, :horizon]
        point = np.median(samples, axis=1).astype(np.float32)

        LagLlamaWrapper._batch_counter += 1
        if LagLlamaWrapper._batch_counter % 10 == 0:
            import time
            print(f"    [lagllama] forecast batch {LagLlamaWrapper._batch_counter} "
                  f"(batch_size={context.shape[0]}) — "
                  f"{time.strftime('%H:%M:%S')}", flush=True)
        return ForecastResult(point_forecast=point, samples=samples)
