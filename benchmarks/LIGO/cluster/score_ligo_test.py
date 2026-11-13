"""Score LIGO test predictions: RMSE, R^2, NLL, calibration, RMSE by SNR quartile.

Default: every v2 run in results/LIGO/v2_* (supervised and zero-shot FM). Per run it scores
test_predictions.csv, or else the newest test_predictions_best_epNNN.csv snapshot (epoch shown
in the `ckpt` column). FM rows (v2_fm_*) have a constant logvar (log val MSE), so their
NLL / coverage columns only reflect a homoscedastic Gaussian. The mean-baseline row predicts
the train-set mean (taken from an FM metrics.json) for every test sample.

Usage: cd ~/TimeSeriesPhysics && uv run --no-sync python /lustre/home/barmstrong/phyts/cluster/score_ligo_test.py [--v1]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/barmstrong/TimeSeriesPhysics/results/LIGO")
V1_RUNS = ["s4d_gaussnll_regression", "conv1d_gaussnll_regression",
           "linoss_h512_p8_lr3e-4_gaussnll_regression", "linoss_h512_p16_lr3e-4_gaussnll_regression",
           "linoss_h512_p32_conv_lr3e-4_gaussnll_regression"]


def pick_csv(d: Path):
    if (d / "test_predictions.csv").exists():
        return d / "test_predictions.csv", "final"
    snaps = sorted(d.glob("test_predictions_best_ep*.csv"))
    if snaps:
        return snaps[-1], snaps[-1].stem.replace("test_predictions_", "")
    return None, None


v1 = "--v1" in sys.argv
dirs = [ROOT / r for r in V1_RUNS] if v1 else sorted(ROOT.glob("v2_*"))

rows, snr_rows, baseline = [], [], None
for d in dirs:
    f, ckpt = pick_csv(d)
    if f is None:
        print("missing:", d)
        continue
    is_fm = d.name.startswith("v2_fm_")
    if is_fm and baseline is None and (d / "metrics.json").exists():
        baseline = json.loads((d / "metrics.json").read_text())["mean_baseline_test_rmse"]
    t = pd.read_csv(f)
    y, mu, lv = (t[c].to_numpy() for c in ("true_chirp_mass", "pred_chirp_mass_mean", "pred_chirp_mass_logvar"))
    err = mu - y
    sd = np.exp(0.5 * lv)
    z = err / sd
    name = d.name.replace("_gaussnll_regression", "").replace("v2_", "")
    rows.append(dict(model=name, kind="FM zero-shot" if is_fm else ("v1" if v1 else "supervised"), ckpt=ckpt,
                     n=len(t), rmse=np.sqrt(np.mean(err**2)), mae=np.mean(np.abs(err)),
                     r2=1 - np.sum(err**2) / np.sum((y - y.mean()) ** 2),
                     nll=np.mean(0.5 * (lv + z**2)), cov68=np.mean(np.abs(z) < 1), cov95=np.mean(np.abs(z) < 1.96),
                     mean_pred_sd=sd.mean(), y_std=y.std()))
    for b, g in t.groupby(pd.qcut(t["snr"], 4), observed=True):
        snr_rows.append(dict(model=name, snr=str(b),
                             rmse=np.sqrt(np.mean((g["pred_chirp_mass_mean"] - g["true_chirp_mass"]) ** 2))))

pd.set_option("display.width", 250)
fmt = lambda x: f"{x:.4f}"
df = pd.DataFrame(rows)
if baseline is not None:
    y_std = df["y_std"].iloc[0]
    df = pd.concat([df, pd.DataFrame([dict(model="mean baseline (train mean)", kind="baseline", ckpt="-",
                                           rmse=baseline, r2=1 - baseline**2 / y_std**2)])], ignore_index=True)
df["n"] = df["n"].astype("Int64")
print(df.sort_values("rmse").to_string(index=False, float_format=fmt, na_rep="-"))
if snr_rows:
    print("\nTest RMSE by SNR quartile:")
    print(pd.DataFrame(snr_rows).pivot(index="model", columns="snr", values="rmse").to_string(float_format=fmt))
