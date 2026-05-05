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
    EmbeddingResult,
    ForecastResult,
)


def _patch_gluonts_freq_names() -> None:
    """Work around pandas 2.2+ frequency rename ("Q"->"QE", "M"->"ME", etc.)
    breaking gluonts' `norm_freq_str`, which LagLlamaEstimator calls on load.
    Idempotent; safe to call multiple times.
    """
    import gluonts.time_feature._base as _base
    import gluonts.time_feature.lag as _lag

    if getattr(_base, "_kepler_patched", False):
        return
    _orig = _base.norm_freq_str

    def _patched(freq_str: str) -> str:
        s = _orig(freq_str)
        # pandas 2.2 renamed: QE->Q, ME->M, YE->A, BQE->Q, BME->M, ...
        return {
            "QE": "Q", "BQE": "Q",
            "ME": "M", "BME": "M", "SME": "M",
            "YE": "A", "Y": "A", "BYE": "A",
            "h": "H", "min": "T", "s": "S",
        }.get(s, s)

    _base.norm_freq_str = _patched
    _lag.norm_freq_str = _patched
    _base._kepler_patched = True


class LagLlamaWrapper(BaseFoundationModel):
    name = "lagllama"
    supports_forecast = True
    supports_denoise = False
    supports_embed = True      # via forward hook on transformer block

    def __init__(self) -> None:
        super().__init__()
        self._predictor = None
        self._lightning_module = None
        self._num_samples = 20
        self._context_len = 512
        self._horizon = 128

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
        _patch_gluonts_freq_names()
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
        self._context_len = context_len
        self._horizon = horizon

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
        lm = estimator.create_lightning_module()
        self._predictor = estimator.create_predictor(
            estimator.create_transformation(),
            lm,
        )
        self._lightning_module = lm
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

    # ── Embedding via final LayerNorm hook ────────────────────────────────

    @torch.no_grad()
    def embed(self, signal: np.ndarray, *, pool: str = "last") -> EmbeddingResult:
        """Hook `model.transformer.ln_f` once during the initial forward of a
        short forecast call, then pool the context-only hidden states.

        Lag-Llama batches B*num_samples together for probabilistic sampling;
        we pick the first sample branch since all num_samples copies share the
        same context pass.
        """
        if self._lightning_module is None:
            raise RuntimeError("Call .load() before embed().")

        B_in = signal.shape[0]
        captured: list[torch.Tensor] = []

        def _hook(_m, _inp, out):
            if captured:                      # only keep the first capture
                return
            h = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(h):
                captured.append(h.detach())

        target = self._lightning_module.model.transformer.ln_f
        handle = target.register_forward_hook(_hook)
        try:
            # Use a tiny horizon to minimise autoregressive cost; the first
            # ln_f call processes just the context before any generation.
            self.forecast(signal, horizon=1)
        finally:
            handle.remove()

        if not captured:
            raise RuntimeError("Lag-Llama ln_f hook captured no output.")
        h = captured[0]                       # (B*num_samples, T, d)
        BN, T, d = h.shape
        n = BN // B_in
        h = h.view(B_in, n, T, d)[:, 0]       # pick first sample branch → (B, T, d)
        emb = self._pool_time(h, pool=pool)
        return EmbeddingResult(embeddings=emb.cpu().numpy().astype(np.float32))

    # ── Grad-flowing embedding for fine-tuning ───────────────────────────

    def embed_torch(self, signal: np.ndarray, *, pool: str = "last") -> torch.Tensor:
        """Same as embed() but returns a torch tensor with grads enabled.

        Lag-Llama's GluonTS predictor wraps inference in inference mode, so
        we can't just unwrap the @torch.no_grad on forecast(). Instead we
        capture the LagLlamaModel's pre-forward inputs during a no-grad
        forecast pass, then re-run the model with grads enabled and hook
        ``transformer.ln_f`` (without detaching) to grab its output.
        Cost: one extra full model forward per batch.

        LoRA modules in ``model.transformer.h.<i>.attn.{q,k,v,o}_proj``
        receive gradient flow; layers earlier than the attention stack
        stay frozen, which matches the LoRA fine-tuning intent.
        """
        if self._lightning_module is None:
            raise RuntimeError("Call .load() before embed_torch().")

        B_in = signal.shape[0]
        captured_args: list[tuple] = []

        def _pre_hook(_m, args, kwargs):
            captured_args.append((args, dict(kwargs)))

        pre_handle = self._lightning_module.model.register_forward_pre_hook(
            _pre_hook, with_kwargs=True,
        )
        try:
            self.forecast(signal, horizon=1)
        finally:
            pre_handle.remove()

        if not captured_args:
            raise RuntimeError("Lag-Llama model pre-hook captured nothing.")

        args, kwargs = captured_args[0]

        # Re-run the model with grads enabled; ln_f hook captures (B*N, T, d).
        captured_h: list[torch.Tensor] = []

        def _ln_hook(_m, _inp, out):
            if captured_h:
                return
            h = out[0] if isinstance(out, tuple) else out
            captured_h.append(h)

        h_handle = self._lightning_module.model.transformer.ln_f.register_forward_hook(_ln_hook)
        try:
            with torch.enable_grad():
                _ = self._lightning_module.model(*args, **kwargs)
        finally:
            h_handle.remove()

        if not captured_h:
            raise RuntimeError("Lag-Llama ln_f re-hook captured nothing.")
        h = captured_h[0]                       # (B*num_samples, T, d)
        BN, T, d = h.shape
        n = BN // B_in
        h = h.view(B_in, n, T, d)[:, 0]
        return self._pool_time(h, pool=pool)

    def get_finetune_backbone(self) -> torch.nn.Module:
        if self._lightning_module is None:
            raise RuntimeError("Call .load() before get_finetune_backbone().")
        return self._lightning_module.model
