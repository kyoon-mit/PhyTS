"""Ridge linear probe on the cached LIGO v2 FM embeddings (alternative to the MLP head in
fm_zeroshot_ligo.py, whose val MSE got worse from the first epochs on).

Embeddings are read from /fast/barmstrong/outputs/ligo_fm/<model>_<input>/emb (written by
fm_zeroshot_ligo.py). Features are standardised with train statistics, the ridge solution is
computed in closed form from streamed sufficient statistics (float64), the penalty alpha is
chosen on val MSE over a log grid, and the test split is scored with that alpha.

Outputs: results/LIGO/v2_fm_<model>_<input>_ridge/{metrics.json,test_predictions.csv,
val_predictions.csv}; logvar column = constant log val MSE (same convention as the MLP head).

Usage: fm_linear_probe.py [--models moment chronos ...] [--inputs 256hz 2048hz]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

EMB_ROOT = Path("/fast/barmstrong/outputs/ligo_fm")
RES_ROOT = Path(__file__).resolve().parents[2] / "results" / "LIGO"
MODELS = ["moment", "chronos", "timesfm", "timemoe", "moirai", "granite_ttm"]
ALPHAS = np.logspace(-3, 7, 41)  # relative to n_train (penalty on mean squared error scale)


def shards(out: Path, split: str):
    files = sorted((out / "emb").glob(f"{split}_[0-9][0-9][0-9][0-9].npy"))
    assert files, (out, split)
    return files


def stream(out, split, dev):
    for p in shards(out, split):
        yield torch.from_numpy(np.load(p)).to(dev, torch.float64)


def labels(out, split):
    d = np.load(out / f"labels_{split}.npz")
    return d["y"].astype(np.float64), d["snr"]


def metrics(y, p):
    e = p - y
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))),
                r2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)), n=int(len(y)))


def probe(model, inp, dev):
    t0 = time.time()
    out = EMB_ROOT / f"{model}_{inp}"
    ytr, _ = labels(out, "train")
    yva, snr_va = labels(out, "val")
    yte, snr_te = labels(out, "test")
    n = len(ytr)

    # pass 1: feature mean / std (non-finite entries ignored)
    s = ss = cnt = None
    for E in stream(out, "train", dev):
        ok = torch.isfinite(E)
        E = torch.where(ok, E, 0.0)
        if s is None:
            s, ss, cnt = (torch.zeros(E.shape[1], dtype=torch.float64, device=dev) for _ in range(3))
        s += E.sum(0); ss += (E * E).sum(0); cnt += ok.sum(0)
    mu = s / cnt
    sd = (ss / cnt - mu ** 2).clamp_min(0).sqrt() + 1e-6
    ym = float(ytr.mean())

    def z(E):
        Z = (E - mu) / sd
        return torch.where(torch.isfinite(Z), Z, 0.0)

    # pass 2: Z^T Z and Z^T (y - ym) on train
    d = len(mu)
    G = torch.zeros(d, d, dtype=torch.float64, device=dev)
    b = torch.zeros(d, dtype=torch.float64, device=dev)
    i = 0
    yc = torch.from_numpy(ytr - ym).to(dev)
    for E in stream(out, "train", dev):
        Z = z(E); k = len(Z)
        G += Z.T @ Z; b += Z.T @ yc[i:i + k]; i += k
    assert i == n
    G /= n; b /= n
    lam, V = torch.linalg.eigh(G)
    lam = lam.clamp_min(0)
    Vb = V.T @ b

    def proj(split):
        return torch.cat([z(E) @ V for E in stream(out, split, dev)])

    # val: predictions for every alpha at once
    Pva = proj("val")                                            # (n_val, d)
    A = torch.from_numpy(ALPHAS).to(dev)
    coef = Vb[:, None] / (lam[:, None] + A[None, :])             # (d, n_alpha)
    preds = Pva @ coef + ym                                      # (n_val, n_alpha)
    yv = torch.from_numpy(yva).to(dev)
    val_mse = ((preds - yv[:, None]) ** 2).mean(0).cpu().numpy()
    j = int(np.argmin(val_mse))
    pva = preds[:, j].cpu().numpy()
    del preds
    # Diagnostic (not the protocol): 5-fold CV ridge fitted on val itself, i.e. is there any
    # linearly decodable chirp-mass signal in the embeddings at the low-SNR test prior?
    # Pva = Zva @ V is an orthogonal rotation of the standardised features, so ridge is unchanged.
    folds = torch.arange(len(Pva), device=dev) % 5
    cv_sse = torch.zeros(len(ALPHAS), dtype=torch.float64, device=dev)
    for f in range(5):
        tr, te = folds != f, folds == f
        Pf, yf = Pva[tr], yv[tr]
        pm, ymf = Pf.mean(0), yf.mean()
        Pc = Pf - pm
        lf, Vf = torch.linalg.eigh(Pc.T @ Pc / len(Pc))
        bf = Vf.T @ (Pc.T @ (yf - ymf) / len(Pc))
        cf = Vf @ (bf[:, None] / (lf.clamp_min(0)[:, None] + A[None, :]))
        cv_sse += (((Pva[te] - pm) @ cf + ymf - yv[te, None]) ** 2).sum(0)
    cv_mse = (cv_sse / len(Pva)).cpu().numpy()
    jc = int(np.argmin(cv_mse))
    val_cv = dict(alpha=float(ALPHAS[jc]), rmse=float(np.sqrt(cv_mse[jc])),
                  r2=float(1 - cv_mse[jc] / yva.var()))
    train_r2_min_alpha = float(2 * (coef[:, 0] * Vb).sum() - (coef[:, 0] ** 2 * lam).sum()) / float(ytr.var())
    del Pva
    pte = (proj("test") @ coef[:, j] + ym).cpu().numpy()
    train_r2 = float(2 * (coef[:, j] * Vb).sum() - (coef[:, j] ** 2 * lam).sum()) / float(ytr.var())

    res = dict(model=model, input=inp, probe="ridge", embed_dim=d, n_train=n,
               alpha=float(ALPHAS[j]), alpha_at_grid_edge=j in (0, len(ALPHAS) - 1),
               val_mse_by_alpha={f"{a:.3g}": float(v) for a, v in zip(ALPHAS, val_mse)},
               train_r2=train_r2, train_r2_min_alpha=train_r2_min_alpha,
               val_cv_ridge_diag=val_cv, val=metrics(yva, pva), test=metrics(yte, pte),
               mean_baseline_test_rmse=float(np.sqrt(np.mean((yte - ym) ** 2))),
               elapsed_s=round(time.time() - t0, 1))
    rd = RES_ROOT / f"v2_fm_{model}_{inp}_ridge"
    rd.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(snr=snr_te, true_chirp_mass=yte, pred_chirp_mass_mean=pte,
                      pred_chirp_mass_logvar=np.full_like(pte, np.log(val_mse[j])))
                 ).to_csv(rd / "test_predictions.csv", index=False)
    pd.DataFrame(dict(snr=snr_va, true_chirp_mass=yva, pred_chirp_mass_mean=pva)
                 ).to_csv(rd / "val_predictions.csv", index=False)
    (rd / "metrics.json").write_text(json.dumps(res, indent=2))
    print(f"{model:12s} {inp:7s} alpha={ALPHAS[j]:.3g} train_r2={train_r2:.3f} "
          f"val {res['val']['rmse']:.4f} test {res['test']['rmse']:.4f} r2 {res['test']['r2']:.3f} "
          f"| diag: train_r2@min_alpha={train_r2_min_alpha:.3f} val-CV rmse={val_cv['rmse']:.4f} r2={val_cv['r2']:.3f} "
          f"({res['elapsed_s']:.0f} s)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--inputs", nargs="+", default=["256hz", "2048hz"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    for m in a.models:
        for inp in a.inputs:
            probe(m, inp, a.device)


if __name__ == "__main__":
    main()
