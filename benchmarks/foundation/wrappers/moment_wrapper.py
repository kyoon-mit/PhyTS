"""
MOMENT wrapper -- the only model in the suite with native support for all
three tasks (forecasting, reconstruction, embedding) via a single
`MOMENTPipeline.from_pretrained(..., model_kwargs={'task_name': ...})` call.

Package:        momentfm
HuggingFace:    AutonLab/MOMENT-1-small | small | base | large

MOMENT requires the input length to be a multiple of its patch size (8 for
the base model).  Since the toy dataset has L=640, which is divisible by 8
and 16, no padding is needed.  For forecasting, the horizon must also match
the head's configured `forecast_horizon`, which is set at load time.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .base import (
    BaseFoundationModel,
    DenoiseResult,
    EmbeddingResult,
    ForecastResult,
)


_SIZE_TO_HF = {
    "small": "AutonLab/MOMENT-1-small",
    "base":  "AutonLab/MOMENT-1-base",
    "large": "AutonLab/MOMENT-1-large",
}


class MomentWrapper(BaseFoundationModel):
    name = "moment"
    supports_forecast = True
    supports_denoise = True    # native reconstruction head
    supports_embed = True      # native embedding head

    def __init__(self) -> None:
        super().__init__()
        self.forecast_model: Optional[torch.nn.Module] = None
        self.reconstruct_model: Optional[torch.nn.Module] = None
        self.embed_model: Optional[torch.nn.Module] = None
        self._hf_id: str = ""
        self._forecast_horizon: int = 0
        self._seq_len: int = 512

    # ── Loading ───────────────────────────────────────────────────────────

    def load(
        self,
        device: str = "cuda",
        model_size: str = "base",
        *,
        forecast_horizon: int = 128,
        seq_len: int = 512,
        **_unused,
    ) -> None:
        """Load all three task heads of MOMENT.

        MOMENT's implementation requires picking the task at construction
        time.  We instantiate three pipelines (one per task) so the evaluator
        can switch without reloading.  GPU memory cost is modest since the
        backbone weights are shared across instances in CUDA cache.
        """
        try:
            from momentfm import MOMENTPipeline
        except ImportError as e:
            raise ImportError(
                "momentfm is required for MomentWrapper. "
                "Install via: pip install momentfm"
            ) from e

        if model_size not in _SIZE_TO_HF:
            raise ValueError(
                f"Unknown MOMENT size '{model_size}'. "
                f"Choose one of: {list(_SIZE_TO_HF)}"
            )
        self._hf_id = _SIZE_TO_HF[model_size]
        self._forecast_horizon = forecast_horizon
        self._seq_len = seq_len
        self.device = device

        self.forecast_model = MOMENTPipeline.from_pretrained(
            self._hf_id,
            model_kwargs={
                "task_name": "forecasting",
                "forecast_horizon": forecast_horizon,
                "head_dropout": 0.1,
                "weight_decay": 0.0,
                "freeze_encoder": True,
                "freeze_embedder": True,
                "freeze_head": False,
            },
        )
        self.forecast_model.init()
        self.forecast_model.to(device).eval()

        self.reconstruct_model = MOMENTPipeline.from_pretrained(
            self._hf_id,
            model_kwargs={"task_name": "reconstruction"},
        )
        self.reconstruct_model.init()
        self.reconstruct_model.to(device).eval()

        self.embed_model = MOMENTPipeline.from_pretrained(
            self._hf_id,
            model_kwargs={"task_name": "embedding"},
        )
        self.embed_model.init()
        self.embed_model.to(device).eval()

        # For the BaseFoundationModel.unload() helper
        self.model = self.forecast_model

    def unload(self) -> None:
        self.forecast_model = None
        self.reconstruct_model = None
        self.embed_model = None
        super().unload()

    # ── Forecasting (native) ──────────────────────────────────────────────

    @torch.no_grad()
    def forecast(self, context: np.ndarray, horizon: int) -> ForecastResult:
        if self.forecast_model is None:
            raise RuntimeError("Call .load() before forecast().")
        if horizon != self._forecast_horizon:
            raise ValueError(
                f"MOMENT's forecast head was built for horizon="
                f"{self._forecast_horizon}; got horizon={horizon}. "
                "Reload the model with a matching forecast_horizon."
            )
        B, C = context.shape
        # MOMENT expects (B, n_channels=1, L=seq_len).  Pad/truncate to seq_len.
        x = self._prep_input(context, target_len=self._seq_len)            # (B, 1, seq_len)
        mask = torch.ones(B, self._seq_len, dtype=torch.float32, device=self.device)
        out = self.forecast_model(x_enc=x, input_mask=mask)
        fcst = out.forecast                                                 # (B, 1, H)
        fcst = fcst.squeeze(1).cpu().numpy().astype(np.float32)             # (B, H)
        return ForecastResult(point_forecast=fcst, samples=None)

    # ── Denoising via native reconstruction ───────────────────────────────

    @torch.no_grad()
    def denoise(self, noisy: np.ndarray) -> DenoiseResult:
        if self.reconstruct_model is None:
            raise RuntimeError("Call .load() before denoise().")
        B, L = noisy.shape
        x = self._prep_input(noisy, target_len=self._seq_len)               # (B, 1, seq_len)
        mask = torch.ones(B, self._seq_len, dtype=torch.float32, device=self.device)
        out = self.reconstruct_model(x_enc=x, input_mask=mask)
        rec = out.reconstruction.squeeze(1).cpu().numpy().astype(np.float32)  # (B, seq_len)
        # Trim/pad back to L
        if rec.shape[-1] >= L:
            rec = rec[:, :L]
        else:
            pad = np.zeros((B, L - rec.shape[-1]), dtype=np.float32)
            rec = np.concatenate([rec, pad], axis=-1)
        return DenoiseResult(denoised=rec)

    # ── Embedding (native) ────────────────────────────────────────────────

    @torch.no_grad()
    def embed(self, signal: np.ndarray) -> EmbeddingResult:
        if self.embed_model is None:
            raise RuntimeError("Call .load() before embed().")
        B, L = signal.shape
        x = self._prep_input(signal, target_len=self._seq_len)              # (B, 1, seq_len)
        mask = torch.ones(B, self._seq_len, dtype=torch.float32, device=self.device)
        out = self.embed_model(x_enc=x, input_mask=mask)
        # MOMENT returns .embeddings of shape (B, d_model) after its pooling head.
        emb = out.embeddings.cpu().numpy().astype(np.float32)
        if emb.ndim == 3:  # (B, T, d) -> mean pool
            emb = emb.mean(axis=1)
        return EmbeddingResult(embeddings=emb)

    # ── Fine-tuning parameter selection ───────────────────────────────────

    def get_finetune_params(self) -> list[torch.nn.Parameter]:
        """Only the forecasting head is trainable for fine-tuning.

        MOMENT was pre-trained with a reconstruction objective; to avoid
        destroying the representation, we freeze the encoder and embedder
        and only unfreeze the task-specific head.  This matches the MOMENT
        authors' recommended fine-tuning protocol.
        """
        if self.forecast_model is None:
            raise RuntimeError("Call .load() before get_finetune_params().")
        return [p for p in self.forecast_model.parameters() if p.requires_grad]

    # ── Helpers ───────────────────────────────────────────────────────────

    def _prep_input(self, x: np.ndarray, target_len: int) -> torch.Tensor:
        """(B, L) numpy -> (B, 1, target_len) float tensor on device.

        Pads with zeros on the right or truncates from the right as needed,
        so the last sample of the input lines up with position L-1 in the
        output.  MOMENT's patch embedder requires target_len to be a multiple
        of the patch size (typically 8 for MOMENT-1).
        """
        B, L = x.shape
        if L == target_len:
            arr = x
        elif L < target_len:
            arr = np.concatenate(
                [x, np.zeros((B, target_len - L), dtype=x.dtype)], axis=-1
            )
        else:
            arr = x[:, -target_len:]
        t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(1).to(self.device)
        return t
