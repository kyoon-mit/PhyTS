"""Denoise the TIDMAD test file with one model, for the official denoising score (Table 2).

Every 10th one-second chunk of tidmad_test_file.h5 is cut into 100,000-sample windows (the
training window length) and denoised. The output keeps the official scorer's file layout:
denoised time_series_ch1, untouched reference time_series_ch2, and signal_frequency per chunk.

Supervised models are rebuilt from their training config and best checkpoint. Foundation models
are zero-shot and see short chunks that fit their context: MOMENT reconstructs each chunk with
its own head, the others forecast each half of a chunk from the other half.

Usage:
  python benchmarks/TIDMAD/denoise.py --model raw|conv|s4d|linoss
  benchmarks/foundation/.venv/bin/python benchmarks/TIDMAD/denoise.py --model moment|chronos|...
Writes results/TIDMAD/<model>/tidmad_test_file.h5 (fm_<model> for foundation models).
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

_REPO = Path(__file__).resolve().parents[2]
for p in (str(_REPO / "src"), str(_REPO / "benchmarks")):
    if p not in sys.path:
        sys.path.insert(0, p)

WINDOW = 100_000          # samples per model input
INJECTION_GRID_HZ = np.array([*range(1_100, 10_000, 100), *range(10_000, 100_000, 1_000),
                              *range(100_000, 1_000_000, 10_000),
                              *range(1_000_000, 4_900_001, 100_000)], dtype=np.float64)
SUPERVISED = ("conv", "s4d", "linoss")
# zero-shot model: (wrapper module, wrapper class, model size, chunk length, chunks per call)
FOUNDATION = {
    "moment": ("moment_wrapper", "MomentWrapper", "base", 512, 512),
    "chronos": ("chronos_wrapper", "ChronosWrapper", "tiny", 128, 512),
    "granite_ttm": ("granite_ttm_wrapper", "GraniteTTMWrapper", "base", 512, 512),
    "moirai": ("moirai_wrapper", "MoiraiWrapper", "base", 512, 64),
}


def injection_frequency(ch2: np.ndarray, fs: float = 1e7) -> float:
    """Injection frequency of a chunk: the reference spectrum's peak, snapped to the injection grid."""
    spectrum = np.abs(np.fft.rfft(ch2.astype(np.float32)))
    peak = (np.argmax(spectrum[1:]) + 1) * fs / len(ch2)
    return float(INJECTION_GRID_HZ[np.argmin(np.abs(INJECTION_GRID_HZ - peak))])


def load_supervised(model: str, device: str):
    """Trained model from its config and best checkpoint: (B, L) -> (B, L)."""
    from jsonargparse import ArgumentParser
    from lightning import LightningModule

    name = f"tidmad_{model}_denoising"
    cfg = yaml.safe_load((_REPO / f"configs/TIDMAD/train_{name}.yaml").read_text())
    parser = ArgumentParser()
    parser.add_subclass_arguments(LightningModule, "model", fail_untyped=False)  # as LightningCLI
    module = parser.instantiate_classes(parser.parse_object({"model": cfg["model"]})).model

    ckpt_path = _REPO / f"checkpoints/{name}/best.ckpt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    module.on_load_checkpoint(ckpt)              # restores the JAX weights of LinOSS
    module.load_state_dict(ckpt["state_dict"])
    module.eval().to(device)

    @torch.no_grad()
    def denoise(x: np.ndarray) -> np.ndarray:
        return module(torch.from_numpy(x).to(device)).float().cpu().numpy()

    return denoise


def load_foundation(model: str, device: str):
    """Zero-shot foundation model: (B, L) -> (B, L), denoised in chunks of the model's context.

    100,000 is not a multiple of every chunk length; the last few samples of each window are
    passed through unchanged.
    """
    module, cls, size, chunk, per_call = FOUNDATION[model]
    wrapper = getattr(importlib.import_module(f"foundation.wrappers.{module}"), cls)()
    wrapper.load(device=device, model_size=size, seq_len=chunk,
                 context_len=chunk // 2, horizon=chunk // 2)

    def denoise(x: np.ndarray) -> np.ndarray:
        n = x.shape[1] // chunk * chunk
        chunks = x[:, :n].reshape(-1, chunk)
        out = x.copy()
        out[:, :n] = np.concatenate([wrapper.denoise(chunks[i:i + per_call]).denoised
                                     for i in range(0, len(chunks), per_call)]).reshape(len(x), n)
        return out

    return denoise


def main():
    """Denoise every --stride-th chunk of the test file with --model."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("raw",) + SUPERVISED + tuple(FOUNDATION),
                    help="raw: no denoising (baseline row)")
    ap.add_argument("--test_file", default="data/TIDMAD/original/tidmad_test_file.h5", type=Path)
    ap.add_argument("--scale", default="data/TIDMAD/preprocessed/scale.npy", type=Path)
    ap.add_argument("--out_root", default="results/TIDMAD", type=Path)
    ap.add_argument("--stride", default=10, type=int, help="denoise every Nth one-second chunk")
    ap.add_argument("--batch_size", default=10, type=int, help="windows per model call")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.model == "raw":
        denoise, run_name = (lambda x: x), "raw"
    elif args.model in SUPERVISED:
        denoise, run_name = load_supervised(args.model, args.device), args.model
    else:
        denoise, run_name = load_foundation(args.model, args.device), f"fm_{args.model}"
    scale = float(np.load(args.scale))           # DAQ counts -> mV, as in training

    out = args.out_root / run_name / args.test_file.name
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.test_file, "r") as src, h5py.File(out, "w") as dst:
        n_chunks, chunk_len = src["time_series_ch1"].shape
        picked = np.arange(0, n_chunks, args.stride)
        dst.attrs["sample_rate_hz"] = src.attrs["sample_rate_hz"]
        dst.attrs["n_chunks"] = len(picked)
        dst["signal_frequency"] = [[injection_frequency(src["time_series_ch2"][c])] for c in picked]
        ch1 = dst.create_dataset("time_series_ch1", (len(picked), chunk_len), "float32")
        ch2 = dst.create_dataset("time_series_ch2", (len(picked), chunk_len), "float32")
        for i, c in enumerate(picked):
            windows = (src["time_series_ch1"][c].astype(np.float32) * scale).reshape(-1, WINDOW)
            ch1[i] = np.concatenate([denoise(windows[j:j + args.batch_size])
                                     for j in range(0, len(windows), args.batch_size)]).ravel()
            ch2[i] = src["time_series_ch2"][c].astype(np.float32) * scale
            print(f"chunk {c} ({i + 1}/{len(picked)})", flush=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
