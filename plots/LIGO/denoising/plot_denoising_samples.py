"""
Plot per-SNR-bin denoising samples for a trained LinOSS LIGO denoising checkpoint.

Usage (from repo root):
    python plots/LIGO/denoising/plot_denoising_samples.py \
        --checkpoint checkpoints/ligo_linoss_denoising/checkpoint_epoch_0099_step_1234_metric_0.001234.eqx \
        [--config configs/LIGO/train_ligo_linoss_denoising.yaml] \
        [--data /fast/barmstrong/LIGO/merged/test.h5] \
        [--n-per-bin 3] \
        [--snr-bins 5 10 20 50] \
        [--out-dir plots/LIGO/denoising/figures] \
        [--seed 42]

Each bin produces one figure: rows = examples, cols = H1 / L1.
Three traces per panel: noisy input (gray), clean signal (green), denoised (red dashed).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from models.linoss import LinOSS
from models.utils.jax.training import jax_inference
from models.utils.jax.utils import tensor_to_jax
from tasks.LIGO.ligo_denoising_jax import LinOSSLIGODenoising

DETECTORS = ["H1", "L1"]
PALETTE = {
    "noisy": dict(color="silver", lw=0.8, label="Noisy input"),
    "clean": dict(color="tab:green", lw=1.3, label="Clean signal"),
    "denoised": dict(color="tab:red", lw=1.3, ls="--", label="Denoised"),
}


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────


def build_task(cfg: dict, ckpt_path: Path) -> LinOSSLIGODenoising:
    """Instantiate task module from config and load checkpoint weights."""
    m = cfg["model"]["init_args"]
    inner_model = LinOSS(**m["model"]["init_args"])

    # Construct the task (this sets up jax_model / jax_model_state).
    # load_from_checkpoint calls load_model(), which tries (model, state) first
    # then falls back to model-only if the shapes don't match.
    task = LinOSSLIGODenoising(
        model=inner_model,
        lr=m.get("lr", 1e-3),
        clip_grad_norm=m.get("clip_grad_norm"),
        load_from_checkpoint=str(ckpt_path),
        seed=m.get("seed", 0),
    )

    return task


def _load_3tuple(task: LinOSSLIGODenoising, ckpt_path: Path, lr: float, clip: float | None):
    """Load checkpoint saved as (model, state, opt_state)."""
    jax_opt = optax.chain(
        optax.clip_by_global_norm(clip) if clip is not None else optax.identity(),
        optax.adamw(lr),
    )
    filter_spec = jax.tree_util.tree_map(eqx.is_inexact_array, task.jax_model)
    diff_model, _ = eqx.partition(task.jax_model, filter_spec)
    opt_state = jax_opt.init(diff_model)

    loaded_model, loaded_state, _ = eqx.tree_deserialise_leaves(
        ckpt_path, (task.jax_model, task.jax_model_state, opt_state)
    )
    task.jax_model = loaded_model
    task.jax_model_state = loaded_state


def infer(task: LinOSSLIGODenoising, X_np: np.ndarray) -> np.ndarray:
    """Run inference on a batch.

    Args:
        X_np: (B, n_ifos, L) — whitened injected strain
    Returns:
        (B, L, n_ifos) — denoised output
    """
    x_torch = torch.as_tensor(X_np, dtype=torch.float32)
    x_jax = tensor_to_jax(x_torch)
    x_jax = jnp.transpose(x_jax, (0, 2, 1))  # (B, n_ifos, L) → (B, L, n_ifos)
    B = x_jax.shape[0]
    keys = jax.random.split(jax.random.key(0), B)
    out = jax_inference(
        model=task.jax_model,
        x=x_jax,
        state=task.jax_model_state,
        key=keys,
    )
    return np.asarray(out)  # (B, L, n_ifos)


# ──────────────────────────────────────────────────────────────────────────────
# Data collection
# ──────────────────────────────────────────────────────────────────────────────


def _snr_bin_labels(edges: list[float]) -> list[str]:
    labels = [f"SNR {edges[i]:.0f}–{edges[i + 1]:.0f}" for i in range(len(edges) - 1)]
    labels.append(f"SNR ≥{edges[-1]:.0f}")
    return labels


def collect_by_snr_bin(
    h5_path: str,
    data_cfg: dict,
    bin_edges: list[float],
    n_per_bin: int,
    max_scan: int,
    rng: np.random.Generator,
) -> dict[str, list[dict]]:
    """Scan the HDF5 test file and return n_per_bin samples for each SNR bin.

    Returns:
        {label: [{"X": (n_ifos, L), "y": (n_ifos, L), "snr": float}]}
    """
    da = data_cfg["init_args"]
    sr = da["strain_frequency"]
    start = int(da["window_begin"] * sr)
    stop = int(da["window_end"] * sr)
    step = da.get("downsample_factor", 1)
    x_key = da.get("injected_data_key", "whitened_injected")
    y_key = da.get("clean_data_key", "whitened_signal")

    edges = sorted(bin_edges)
    labels = _snr_bin_labels(edges)
    bins: dict[str, list[dict]] = {lb: [] for lb in labels}
    counts = {lb: 0 for lb in labels}

    with h5py.File(h5_path, "r") as f:
        n_total = f[x_key].shape[0]
        indices = rng.permutation(min(n_total, max_scan)).tolist()

        for idx in indices:
            if all(v >= n_per_bin for v in counts.values()):
                break

            snr = float(f["snr"][idx])
            if np.isnan(snr) or snr < edges[0]:
                continue

            # np.searchsorted: bin_idx = position of snr in edges array, clamped
            bin_idx = min(int(np.searchsorted(edges, snr, side="right")) - 1, len(labels) - 1)
            lb = labels[bin_idx]

            if counts[lb] >= n_per_bin:
                continue

            X = f[x_key][idx, :, start:stop:step].copy()  # (n_ifos, L)
            y = f[y_key][idx, :, start:stop:step].copy()  # (n_ifos, L)
            bins[lb].append({"X": X, "y": y, "snr": snr})
            counts[lb] += 1

    return bins


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────


def plot_snr_bin(
    label: str,
    samples: list[dict],
    task: LinOSSLIGODenoising,
    sample_rate: float,
    out_dir: Path,
) -> None:
    if not samples:
        print(f"  [skip] no samples for {label}")
        return

    B = len(samples)
    X_batch = np.stack([s["X"] for s in samples])  # (B, n_ifos, L)
    print(f"  Running inference on {B} samples …")
    denoised = infer(task, X_batch)  # (B, L, n_ifos)

    L = X_batch.shape[-1]
    t = np.arange(L) / sample_rate  # seconds within window

    fig, axes = plt.subplots(B, 2, figsize=(13, 3.2 * B), sharex=True, squeeze=False)
    fig.suptitle(f"LIGO LinOSS Denoising — {label}", fontsize=13, fontweight="bold")

    for row, (sample, row_axes) in enumerate(zip(samples, axes)):
        snr = sample["snr"]
        for col, (det, ax) in enumerate(zip(DETECTORS, row_axes)):
            noisy = sample["X"][col]  # (L,)
            clean = sample["y"][col]  # (L,)
            pred = denoised[row, :, col]  # (L,)

            ax.plot(t, noisy, **PALETTE["noisy"])
            ax.plot(t, clean, **PALETTE["clean"])
            ax.plot(t, pred, **PALETTE["denoised"])

            ax.set_title(f"{det}  |  SNR = {snr:.1f}", fontsize=10)
            ax.set_ylabel("Whitened strain", fontsize=8)
            ax.grid(True, ls=":", alpha=0.4)
            if row == 0 and col == 0:
                ax.legend(loc="upper right", fontsize=8, framealpha=0.5)

    for ax in axes[-1]:
        ax.set_xlabel(f"Time [s]  (window {t[0]:.1f}–{t[-1]:.1f} s)")

    plt.tight_layout()
    slug = label.replace(" ", "_").replace("≥", "ge").replace("–", "-")
    out = out_dir / f"denoising_{slug}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot per-SNR-bin denoising samples for a trained LinOSS LIGO model."
    )
    ap.add_argument("--checkpoint", required=True, help=".eqx checkpoint file")
    ap.add_argument(
        "--config",
        default="configs/LIGO/train_ligo_linoss_denoising.yaml",
        help="Training YAML config (relative to repo root or absolute)",
    )
    ap.add_argument("--data", default=None, help="Override the test HDF5 path from config")
    ap.add_argument("--n-per-bin", type=int, default=3, help="Samples to plot per SNR bin")
    ap.add_argument(
        "--snr-bins",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 20.0, 50.0],
        metavar="EDGE",
        help="SNR bin edges. N edges → N bins: [e0,e1), [e1,e2), …, [eN-1,∞)",
    )
    ap.add_argument(
        "--max-scan",
        type=int,
        default=10000,
        help="Max events to scan in the test file when collecting samples",
    )
    ap.add_argument(
        "--out-dir",
        default="plots/LIGO/denoising/figures",
        help="Output directory",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    data_cfg = cfg["data"]
    h5_path = args.data or data_cfg["init_args"]["test_file"]
    da = data_cfg["init_args"]
    sample_rate = da["strain_frequency"] / da.get("downsample_factor", 1)

    print(f"Config  : {cfg_path}")
    print(f"Data    : {h5_path}")
    print(f"Sample rate (effective): {sample_rate:.0f} Hz")
    print(
        f"Window  : {da['window_begin']}–{da['window_end']} s  "
        f"({int((da['window_end'] - da['window_begin']) * sample_rate)} samples)"
    )

    print(f"\nBuilding model …")
    task = build_task(cfg, Path(args.checkpoint))
    task.eval()

    print(f"\nScanning test file (up to {args.max_scan} events) …")
    bins_data = collect_by_snr_bin(
        h5_path,
        data_cfg,
        bin_edges=args.snr_bins,
        n_per_bin=args.n_per_bin,
        max_scan=args.max_scan,
        rng=rng,
    )

    print("\nGenerating plots …")
    for label, samples in bins_data.items():
        print(f"\n{label}: {len(samples)} sample(s)")
        plot_snr_bin(label, samples, task, sample_rate, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
