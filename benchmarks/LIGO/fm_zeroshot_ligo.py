"""Zero-shot foundation-model benchmark on LIGO (chirp-mass regression, paper Table 2).

Protocol (frozen backbone, same splits / window / jitter as the supervised runs)
--------------------------------------------------------------------------------
* Window: 59-63 s (4 s, 8192 samples per IFO at 2048 Hz) cut from the stored 58.75-63.25 s
  segment, shifted by a fixed per-sample offset in [-0.25, 0.25] s drawn exactly like
  LIGODataset's val/test shifts (seed 0 + {train: 0, val: 1, test: 2}).
* Each IFO channel is split into 16 non-overlapping 512-sample chunks (the common context limit
  of MOMENT / Chronos / TimesFM / TTM). Every chunk is embedded by the frozen model with the same
  token pooling as the TESS / Project 8 zero-shot rows: mean for MOMENT / Chronos / MOIRAI / TTM,
  last token for the causal decoders Time-MoE / TimesFM; MOIRAI is embedded as in a patch-32,
  horizon-128 forecast pass. Chunk embeddings are mean-pooled and H1 / L1 are concatenated.
* Head: closed-form Ridge(alpha=1) on z-scored features and target. Because the training split's
  SNR prior differs from val / test (Appendix A), the probe and its z-scoring statistics are fitted
  on the validation split and evaluated on the held-out test split.

Outputs: <results_root>/fm_<model>/{test_predictions.csv, val_predictions.csv, metrics.json};
test_predictions.csv uses the supervised schema (snr, true_chirp_mass, pred_chirp_mass_mean,
pred_chirp_mass_logvar), with logvar the constant log val MSE.

Embedding extraction is sharded and resumable (finished shards are skipped) and can be split over
several GPUs with --worker i --num_workers k; the last worker to finish fits the probe. With
--time_limit_h the script stops after that long and exits with code 85 (for batch-system requeue).
Embedding all three splits (1.9M samples) takes from ~1 GPU-hour (TTM) to ~40 GPU-hours
(Chronos, Time-MoE) on an H100.

Usage (foundation-model venv, see benchmarks/foundation):
  python benchmarks/LIGO/fm_zeroshot_ligo.py --model moment
  # smoke test
  python benchmarks/LIGO/fm_zeroshot_ligo.py --model moment --max_samples 512 --out_root /tmp/ligo_fm_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
for p in (str(_REPO / "src"), str(_REPO / "benchmarks")):
    if p not in sys.path:
        sys.path.insert(0, p)

REQUEUE_RC = 85
SPLITS = ("train", "val", "test")
SEG_LEN = 9216            # 58.75-63.25 s stored at 2048 Hz
WIN_START = 512           # 0.25 s into the segment -> 59 s
WIN_LEN = 8192            # 4 s
MAX_SHIFT = 512           # +-0.25 s
CHUNK_LEN = 512
RIDGE_ALPHA = 1.0
PROBE_FIT = "val"         # split the probe and its z-scoring are fitted on

MODELS = ("moment", "chronos", "timesfm", "timemoe", "moirai", "granite_ttm")
# Token pooling of the TESS / Project 8 zero-shot runs: encoders -> mean, causal decoders -> last.
POOL = {"moment": "mean", "chronos": "mean", "moirai": "mean", "granite_ttm": "mean",
        "timemoe": "last", "timesfm": "last"}
MOIRAI_HORIZON = 128


# ── model loading ──────────────────────────────────────────────────────────────

def load_wrapper(name: str, device: str):
    if name == "moment":
        from foundation.wrappers.moment_wrapper import MomentWrapper as W
        w = W(); w.load(device=device, model_size="base", seq_len=CHUNK_LEN)
    elif name == "chronos":
        from foundation.wrappers.chronos_wrapper import ChronosWrapper as W
        w = W(); w.load(device=device, model_size="base")
    elif name == "timesfm":
        from foundation.wrappers.timesfm_wrapper import TimesFMWrapper as W
        w = W(); w.load(device=device, model_size="base", context_len=CHUNK_LEN, horizon=128)
    elif name == "timemoe":
        from foundation.wrappers.timemoe_wrapper import TimeMoEWrapper as W
        w = W(); w.load(device=device, model_size="base")
    elif name == "moirai":
        from foundation.wrappers.moirai_wrapper import MoiraiWrapper as W
        w = W(); w.load(device=device, model_size="base", context_len=CHUNK_LEN)
        w._module.to(device).eval()
    elif name == "granite_ttm":
        from foundation.wrappers.granite_ttm_wrapper import GraniteTTMWrapper as W
        w = W(); w.load(device=device, model_size="base")
        if w._ctx_len != CHUNK_LEN:
            raise ValueError(f"TTM context {w._ctx_len} != chunk length {CHUNK_LEN}")
    else:
        raise ValueError(name)
    return w


def embed_kwargs(model: str) -> dict:
    kw = {}
    if POOL[model] != "mean":
        kw["pool"] = POOL[model]
    if model == "moirai":
        kw["horizon"] = MOIRAI_HORIZON
    return kw


# ── data / embedding ─────────────────────────────────────────────────────────

def split_shifts(split: str, n: int, jitter_seed: int = 0) -> np.ndarray:
    """Identical to LIGODataset.fixed_shifts."""
    rng = np.random.default_rng(jitter_seed + {"train": 0, "val": 1, "test": 2}[split])
    return rng.integers(-MAX_SHIFT, MAX_SHIFT + 1, size=n)


def read_windows(f: h5py.File, a: int, b: int, shifts: np.ndarray) -> np.ndarray:
    """Rows [a, b) -> (n, 2, 8192) float32 windows."""
    X = f["whitened_injected"][a:b]                               # (n, 2, 9216)
    assert X.shape[-1] == SEG_LEN, X.shape
    idx = WIN_START + shifts[a:b, None] + np.arange(WIN_LEN)[None, :]
    return np.take_along_axis(X, idx[:, None, :], axis=-1).astype(np.float32)


def embed_windows(wrapper, X: np.ndarray, batch: int, embed_kw: dict) -> np.ndarray:
    """(n, 2, L) -> (n, 2*d): per-chunk embeddings, mean over chunks, concat IFOs."""
    n, c, L = X.shape
    k = L // CHUNK_LEN
    flat = X.reshape(n * c * k, CHUNK_LEN)
    outs = []
    with torch.inference_mode():
        for i in range(0, len(flat), batch):
            outs.append(np.asarray(wrapper.embed(flat[i:i + batch], **embed_kw).embeddings, dtype=np.float32))
    return np.concatenate(outs).reshape(n, c, k, -1).mean(axis=2).reshape(n, -1)


def n_samples(args, split: str) -> int:
    with h5py.File(Path(args.data_dir) / f"{split}.h5", "r") as f:
        n = f["whitened_injected"].shape[0]
    return min(n, args.max_samples) if args.max_samples else n


def all_shards_done(args, out: Path) -> bool:
    for split in SPLITS:
        k = -(-n_samples(args, split) // args.shard_size)
        if not all((out / "emb" / f"{split}_{i:04d}.npy").exists() for i in range(k)):
            return False
        if not (out / f"labels_{split}.npz").exists():
            return False
    return True


def claim_head(out: Path) -> bool:
    """Exactly one worker fits the probe: the first to create the lock file."""
    try:
        fd = os.open(out / "head.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False


def extract(args, out: Path, t0: float) -> bool:
    """Embed all splits shard by shard. Returns False if the time limit was hit."""
    wrapper = None
    pool = ThreadPoolExecutor(max_workers=1)
    for split in SPLITS:
        with h5py.File(Path(args.data_dir) / f"{split}.h5", "r") as f:
            n = n_samples(args, split)
            shifts = split_shifts(split, f["whitened_injected"].shape[0])
            lab = out / f"labels_{split}.npz"
            if not lab.exists():
                tmp = lab.with_name(f"{lab.stem}.tmp{args.worker}.npz")   # atomic, per-worker name
                np.savez(tmp, y=f["chirp_mass"][:n].astype(np.float32),
                         snr=f["snr"][:n].astype(np.float32), shift=shifts[:n])
                os.replace(tmp, lab)
            bounds = [(a, min(a + args.shard_size, n)) for a in range(0, n, args.shard_size)]
            todo = [(i, a, b) for i, (a, b) in enumerate(bounds)
                    if i % args.num_workers == args.worker
                    and not (out / "emb" / f"{split}_{i:04d}.npy").exists()]
            print(f"[{split}] n={n} shards={len(bounds)} todo={len(todo)}", flush=True)
            if not todo:
                continue
            if wrapper is None:
                print(f"loading {args.model} ...", flush=True)
                wrapper = load_wrapper(args.model, args.device)
            fut = pool.submit(read_windows, f, todo[0][1], todo[0][2], shifts)
            for j, (i, a, b) in enumerate(todo):
                X = fut.result()
                if j + 1 < len(todo):
                    fut = pool.submit(read_windows, f, todo[j + 1][1], todo[j + 1][2], shifts)
                ts = time.time()
                E = embed_windows(wrapper, X, args.batch_size, embed_kwargs(args.model))
                if not np.isfinite(E).all():
                    bad = ~np.isfinite(E).all(1)
                    print(f"  WARNING shard {split}_{i:04d}: {bad.sum()} rows non-finite")
                dst = out / "emb" / f"{split}_{i:04d}.npy"
                tmp = dst.with_suffix(".tmp.npy")
                np.save(tmp, E)
                os.replace(tmp, dst)
                el = time.time() - t0
                print(f"  {split} shard {i + 1}/{len(bounds)} rows {a}-{b} dim={E.shape[1:]} "
                      f"{(b - a) / (time.time() - ts):.0f} samples/s  elapsed {el / 3600:.2f} h", flush=True)
                if args.time_limit_h and el > args.time_limit_h * 3600 \
                        and (j + 1 < len(todo) or split != SPLITS[-1]):
                    pool.shutdown(wait=True)
                    return False
    pool.shutdown()
    return True


# ── ridge probe ────────────────────────────────────────────────────────────────

def fit_and_eval_probe(args, out: Path, res_dir: Path):
    """sklearn-equivalent Ridge(alpha=1, fit_intercept=True) on features and target z-scored with the
    fit split's mean / std (+1e-6 / +1e-8), fitted on PROBE_FIT and scored on test. Solved in closed
    form from shard-streamed float64 sufficient statistics, so the embeddings never have to fit in
    memory at once."""
    dev = args.device

    def shard_files(split):
        return sorted((out / "emb").glob(f"{split}_[0-9][0-9][0-9][0-9].npy"))

    def load(p):
        return torch.from_numpy(np.load(p)).to(dev, torch.float64)

    # pass 1: per-feature mean / std over finite entries (ddof=0, like np.std)
    s = ss = cnt = None
    for p in shard_files(PROBE_FIT):
        E = load(p)
        fin = torch.isfinite(E)
        E = torch.where(fin, E, torch.zeros((), dtype=E.dtype, device=dev))
        if s is None:
            s, ss, cnt = (torch.zeros(E.shape[1], dtype=torch.float64, device=dev) for _ in range(3))
        s += E.sum(0); ss += (E * E).sum(0); cnt += fin.sum(0)
    mu = s / cnt
    sd = (ss / cnt - mu ** 2).clamp_min(0).sqrt() + 1e-6

    def z(E):  # non-finite entries -> 0 after z-scoring
        return torch.nan_to_num((E - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)

    lab = {sp: np.load(out / f"labels_{sp}.npz") for sp in SPLITS}
    yfit = lab[PROBE_FIT]["y"].astype(np.float64)
    ym, ys = float(yfit.mean()), float(yfit.std() + 1e-8)

    # pass 2: Z'Z, Z'y, column sums (sklearn centres X and y before solving)
    d = mu.numel()
    A = torch.zeros(d, d, dtype=torch.float64, device=dev)
    b = torch.zeros(d, dtype=torch.float64, device=dev)
    zs = torch.zeros(d, dtype=torch.float64, device=dev)
    yz = torch.from_numpy((yfit - ym) / ys).to(dev)
    i = 0
    for p in shard_files(PROBE_FIT):
        Z = z(load(p))
        yb = yz[i:i + len(Z)]; i += len(Z)
        A += Z.T @ Z; b += Z.T @ yb; zs += Z.sum(0)
    n = i
    assert n == len(yfit), (n, len(yfit))
    zm, yzm = zs / n, yz.mean()
    Ac = A - n * torch.outer(zm, zm)
    bc = b - n * zm * yzm
    w = torch.linalg.solve(Ac + RIDGE_ALPHA * torch.eye(d, dtype=torch.float64, device=dev), bc)
    c = yzm - zm @ w

    def predict(split):
        return np.concatenate([((z(load(p)) @ w + c) * ys + ym).cpu().numpy() for p in shard_files(split)])

    pva, pte = predict("val"), predict("test")
    yva, snr_va = lab["val"]["y"], lab["val"]["snr"]
    yte, snr_te = lab["test"]["y"], lab["test"]["snr"]
    assert len(pva) == len(yva) and len(pte) == len(yte)
    val_mse = float(np.mean((pva - yva) ** 2))

    def metrics(y, p):
        e = p - y
        return dict(rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))),
                    r2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)), n=int(len(y)))

    res = dict(model=args.model, pool=POOL[args.model], chunk_len=CHUNK_LEN,
               moirai_horizon=MOIRAI_HORIZON if args.model == "moirai" else None,
               head="ridge", ridge_alpha=RIDGE_ALPHA, probe_fit=PROBE_FIT, embed_dim=int(d), n_fit=int(n),
               val=metrics(yva, pva), test=metrics(yte, pte),
               mean_baseline_test_rmse=float(np.std(yte)))
    res_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(snr=snr_te, true_chirp_mass=yte, pred_chirp_mass_mean=pte,
                      pred_chirp_mass_logvar=np.full_like(pte, np.log(val_mse)))
                 ).to_csv(res_dir / "test_predictions.csv", index=False)
    pd.DataFrame(dict(snr=snr_va, true_chirp_mass=yva, pred_chirp_mass_mean=pva)
                 ).to_csv(res_dir / "val_predictions.csv", index=False)
    (res_dir / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--data_dir", default="data/LIGO/2048Hz")
    ap.add_argument("--out_root", default="outputs/LIGO/fm_embeddings", help="embedding cache")
    ap.add_argument("--results_root", default="results/LIGO")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=512, help="chunks per forward pass")
    ap.add_argument("--shard_size", type=int, default=20000, help="samples per embedding shard")
    ap.add_argument("--max_samples", type=int, default=None, help="per split (smoke tests)")
    ap.add_argument("--time_limit_h", type=float, default=0,
                    help=f"stop extraction after this many hours and exit {REQUEUE_RC} (0 = no limit)")
    ap.add_argument("--stage", choices=("all", "embed", "head"), default="all")
    ap.add_argument("--worker", type=int, default=0, help="this worker's index (multi-GPU split)")
    ap.add_argument("--num_workers", type=int, default=1,
                    help="split shards round-robin over this many jobs; the last to finish fits the probe")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    tag = f"{args.model}_2048hz"
    if POOL[args.model] != "mean":
        tag += f"_{POOL[args.model]}"
    if args.model == "moirai":
        tag += f"_h{MOIRAI_HORIZON}"
    out = Path(args.out_root) / tag
    (out / "emb").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.stage in ("all", "embed"):
        if not extract(args, out, t0):
            print(f"time limit {args.time_limit_h} h reached, exiting {REQUEUE_RC} for requeue", flush=True)
            sys.exit(REQUEUE_RC)
        print(f"embeddings complete ({(time.time() - t0) / 3600:.2f} h)", flush=True)
    if args.stage in ("all", "head"):
        if args.num_workers > 1:
            if not all_shards_done(args, out):
                print("other workers' shards still missing; leaving the probe to the last worker", flush=True)
                return
            if not claim_head(out):
                print("probe already claimed by another worker", flush=True)
                return
        res_dir = out / "results" if args.max_samples else Path(args.results_root) / f"fm_{args.model}"
        fit_and_eval_probe(args, out, res_dir)


if __name__ == "__main__":
    main()
