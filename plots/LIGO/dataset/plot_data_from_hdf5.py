"""
Two-panel plot (time slice + PSD) for LIGO BNS data
loaded from an HDF5 file with shape (N_events, 2, 16384):
  - axis 0: event index
  - axis 1: detector  (0 = H1, 1 = L1)
  - axis 2: samples   (16384 @ 16384 Hz  ->  1 s chunks)

Keys used:
  whitened_signal    -> pure injected waveform (signal)
  whitened_bkg       -> detector noise only
  whitened_injected  -> signal + noise          (measurement)
"""

from __future__ import annotations
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries

# Always write outputs next to this script (CWD may be read-only).
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 256.0
DETECTORS   = {0: "H1", 1: "L1"}
KEYS        = ("whitened_signal", "whitened_bkg", "whitened_injected")


# ---------------------------------------------------------------------------
# HDF5 loader
# ---------------------------------------------------------------------------
def load_event(h5path: str,
               event_idx: int | None = None,
               rng: np.random.Generator | None = None):
    """
    Load one event and return TimeSeries objects.

    Returns
    -------
    event_idx : int  -- which event was actually loaded
    data      : dict -- data[key][det] -> gwpy.TimeSeries
                        e.g. data["whitened_injected"]["H1"]
    meta      : dict -- scalar parameters (snr, chirp_mass, ...)
    """
    rng = rng or np.random.default_rng()
    with h5py.File(h5path, "r") as f:
        n_events = f["whitened_injected"].shape[0]
        if event_idx is None:
            event_idx = int(rng.integers(0, n_events))

        data: dict[str, dict[str, TimeSeries]] = {}
        for key in KEYS:
            arr = f[key][event_idx]            # shape (2, 16384)
            data[key] = {
                DETECTORS[d]: TimeSeries(arr[d],
                                         sample_rate=SAMPLE_RATE,
                                         name=f"{DETECTORS[d]} {key}",
                                         channel=f"{DETECTORS[d]}:{key}")
                for d in DETECTORS
            }

        meta = {k: float(f[k][event_idx])
                for k in ("snr", "chirp_mass", "distance",
                          "mass_1", "mass_2")
                if k in f}

    return event_idx, data, meta


# ---------------------------------------------------------------------------
# Plotter — accepts multiple TimeSeries
# ---------------------------------------------------------------------------
def plot_two_panel_style(ts_ch1_h1: TimeSeries,
                      ts_ch2_h1: TimeSeries | None = None,
                      ts_ch1_l1: TimeSeries | None = None,
                      ts_ch2_l1: TimeSeries | None = None,
                      slice_ms: float = 10.0,
                      slice_center: str = "start",          # not used, slice from 63s backward
                      freq_marker: float | None = None,
                      freq_label: str = "Feature of Interest",
                      ch1_label: str = "i strain",
                      ch2_label: str = "Pure signal x5",
                      savepath: str | None = None):
    """Two-panel stacked H1/L1 plot."""
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
    })
    C1, C2, CM = "tab:blue", "tab:orange", "tab:red"

    fig, axes = plt.subplots(2, 2, figsize=(13, 5), sharex="col")
    ax_t_h1, ax_f_h1 = axes[0]
    ax_t_l1, ax_f_l1 = axes[1]

    dur_h1 = ts_ch1_h1.duration.value
    slice_s = slice_ms / 1000.0
    t_end = 63
    t_start = 63 - slice_s

    def _slice(ts):
        s = ts.crop(t_start, t_end)
        return (s.times.value - 63), s.value

    x_h1, y_h1 = _slice(ts_ch1_h1)
    ax_t_h1.plot(x_h1, y_h1, color="lightgray", linewidth=0.9, label="H1 strain (whitened)")
    if ts_ch2_h1 is not None:
        x2_h1, y2_h1 = _slice(ts_ch2_h1)
        y2_h1_scaled = y2_h1 * 5
        ax_t_h1.plot(x2_h1, y2_h1_scaled, color=C1, linewidth=0.9, label="H1 signal (whitened) x5")
    ax_t_h1.set_title(f"Time Slice ({slice_ms/1000:.0f}s of 64s Data Chunk)")
    ax_t_h1.set_xlim(-slice_s, 0)
    ax_t_h1.legend(loc="upper left", frameon=True, fancybox=True, framealpha=0.3, edgecolor="gray", facecolor="white")
    ax_t_h1.grid(True, linestyle=":", alpha=0.4)

    if ts_ch1_l1 is not None:
        x_l1, y_l1 = _slice(ts_ch1_l1)
        ax_t_l1.plot(x_l1, y_l1, color="lightgray", linewidth=0.9, label="L1 strain (whitened)")
        if ts_ch2_l1 is not None:
            x2_l1, y2_l1 = _slice(ts_ch2_l1)
            y2_l1_scaled = y2_l1 * 5
            ax_t_l1.plot(x2_l1, y2_l1_scaled, color=C2, linewidth=0.9, label="L1 signal (whitened) x5")
    ax_t_l1.set_xlabel("Time from Coalescence [s]")
    ax_t_l1.set_title("")
    ax_t_l1.set_xlim(-slice_s, 0)
    ax_t_l1.legend(loc="upper left", frameon=True, fancybox=True, framealpha=0.3, edgecolor="gray", facecolor="white")
    ax_t_l1.grid(True, linestyle=":", alpha=0.4)

    ts_psd1_h1 = ts_ch1_h1.crop(1, 63)
    psd_h1 = ts_psd1_h1.psd()
    ax_f_h1.loglog(psd_h1.frequencies.value, psd_h1.value,
                   color="lightgray", linewidth=0.9, label="H1 strain (whitened)")
    if ts_ch2_h1 is not None:
        ts_psd2_h1 = ts_ch2_h1.crop(1, 63)
        psd2_h1 = ts_psd2_h1.psd()
        ax_f_h1.loglog(psd2_h1.frequencies.value, psd2_h1.value,
                       color=C1, linewidth=0.9, label="H1 signal (whitened)")
    if freq_marker is not None:
        ax_f_h1.axvline(freq_marker, color=CM, linestyle="--", linewidth=1.2,
                        label=f"{freq_label} [{freq_marker:.0f} Hz]")
    ax_f_h1.set_title("Frequency Domain")
    ax_f_h1.set_xlim(20, 100)
    ax_f_h1.legend(loc="lower right", frameon=True, fancybox=True, framealpha=0.3, edgecolor="gray", facecolor="white")
    ax_f_h1.grid(True, which="both", linestyle=":", alpha=0.4)

    if ts_ch1_l1 is not None:
        ts_psd1_l1 = ts_ch1_l1.crop(1, 63)
        psd_l1 = ts_psd1_l1.psd()
        ax_f_l1.loglog(psd_l1.frequencies.value, psd_l1.value,
                       color="lightgray", linewidth=0.9, label="L1 strain (whitened)")
        if ts_ch2_l1 is not None:
            ts_psd2_l1 = ts_ch2_l1.crop(1, 63)
            psd2_l1 = ts_psd2_l1.psd()
            ax_f_l1.loglog(psd2_l1.frequencies.value, psd2_l1.value,
                           color=C2, linewidth=0.9, label="L1 signal (whitened)")
    ax_f_l1.set_xlabel("Frequency [Hz]")
    ax_f_l1.set_title("")
    ax_f_l1.set_xlim(20, 100)
    ax_f_l1.legend(loc="lower right", frameon=True, fancybox=True, framealpha=0.3, edgecolor="gray", facecolor="white")
    ax_f_l1.grid(True, which="both", linestyle=":", alpha=0.4)

    # Align left and right y ranges
    left_ys = [y_h1]
    if ts_ch2_h1 is not None:
        left_ys.append(y2_h1_scaled)
    if ts_ch1_l1 is not None:
        left_ys.append(y_l1)
    if ts_ch2_l1 is not None:
        left_ys.append(y2_l1_scaled)
    left_all = np.concatenate(left_ys)
    left_ylim = (-np.max(np.abs(left_all)), np.max(np.abs(left_all)))
    ax_t_h1.set_ylim(left_ylim)
    ax_t_l1.set_ylim(left_ylim)

    right_ys = [psd_h1.value]
    if ts_ch2_h1 is not None:
        right_ys.append(psd2_h1.value)
    if ts_ch1_l1 is not None:
        right_ys.append(psd_l1.value)
    if ts_ch2_l1 is not None:
        right_ys.append(psd2_l1.value)
    right_all = np.concatenate(right_ys)
    right_ylim = (np.min(right_all), np.max(right_all))
    ax_f_h1.set_ylim(right_ylim)
    ax_f_l1.set_ylim(right_ylim)

    ax_t_h1.xaxis.set_tick_params(labelbottom=False)
    ax_f_h1.xaxis.set_tick_params(labelbottom=False)
    ax_f_h1.yaxis.set_tick_params(labelleft=True)
    ax_f_l1.yaxis.set_tick_params(labelleft=True)
    fig.text(0.015, 0.5, "Strain", va="center", ha="center", rotation="vertical")
    fig.text(0.50, 0.5, r"Power Spectral Density (strain$^2$/Hz)", va="center", ha="center", rotation="vertical")

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    H5_PATH = "<your file path goes here>"     # <- your file

    rng = np.random.default_rng()       # no seed -> truly random event each run
    idx, data, meta = load_event(H5_PATH, rng=rng)
    print(f"Loaded event #{idx}  meta={meta}")

    plot_two_panel_style(
        ts_ch1_h1    = data["whitened_injected"]["H1"],
        ts_ch2_h1    = data["whitened_signal"]["H1"],
        ts_ch1_l1    = data["whitened_injected"]["L1"],
        ts_ch2_l1    = data["whitened_signal"]["L1"],
        slice_ms      = 2000.0,
        savepath      = OUT_DIR / f"event{idx}_H1L1.pdf",
    )

    plt.show()