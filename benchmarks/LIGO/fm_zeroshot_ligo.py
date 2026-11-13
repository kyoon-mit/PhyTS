"""Zero-shot foundation-model benchmark on the LIGO 2048 Hz v2 dataset.

Protocol (frozen backbone, same splits / window / jitter as the supervised v2 runs)
---------------------------------------------------------------------------------
* Window: 59-63 s (4 s) cut from the stored 58.75-63.25 s segment, shifted by a fixed
  per-sample offset in [-0.25, 0.25] s drawn exactly like LIGODataset's val/test shifts
  (seed 0 + {train: 0, val: 1, test: 2}).  Train uses one fixed shift per sample too,
  since the embeddings are computed once.
* --input 2048hz : full resolution, 8192 samples per IFO.
  --input 256hz  : anti-aliased FIR decimation x8 (zero phase) -> 1024 samples per IFO
                   (the v1 / paper FM protocol).
* Each IFO channel is split into non-overlapping chunks of --chunk_len (512, the common
  context limit of MOMENT / Chronos / TimesFM / TTM), every chunk is embedded by the frozen
  model, chunk embeddings are mean-pooled, and H1 / L1 embeddings are concatenated.
  -> 2 chunks per IFO at 256 Hz, 16 chunks per IFO at 2048 Hz.
* Default head: an MLP (same as chronos_ligo.py) trained with MSE on standardised embeddings
  and standardised chirp mass, early-stopped on val MSE, evaluated on the full test split.
* --paper_protocol: the protocol behind the Kepler / TESS / Project 8 zero-shot FM rows
  (run_tess*.py, run_project8.py, evaluators/regression.py @ 82188e1):
    - token pooling per model inside each chunk: mean for MOMENT / Chronos / MOIRAI / TTM,
      last token for the causal decoders Time-MoE / TimesFM (then mean over chunks, concat IFOs,
      like run_project8.py's mean over windows and concat over I/Q channels);
    - MOIRAI embedded as in a patch-32, horizon-128 forecast pass (prediction-mask tokens present);
    - head: Ridge(alpha=1) on train-standardised features and target, fitted on train only.
  Embeddings go to <model>_<input>[_last][_h128]; results to v2_fm_<...>_probe.
  Output test_predictions.csv uses the supervised schema (snr, true_chirp_mass,
  pred_chirp_mass_mean, pred_chirp_mass_logvar); logvar is the constant log val MSE.

Embedding extraction is sharded and resumable: finished shards are skipped, and after
--time_limit_h hours the script exits with code 85 so condor can requeue it
(on_exit_remove = ExitCode =!= 85), like the supervised v2 training jobs.

Usage
-----
  cd ~/TimeSeriesPhysics
  benchmarks/foundation/.venv/bin/python benchmarks/LIGO/fm_zeroshot_ligo.py \
      --model chronos --input 256hz
  # smoke test
  ... --model moment --input 2048hz --max_samples 512 --out_root /fast/barmstrong/outputs/ligo_fm_smoke
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
import torch.nn as nn
from scipy.signal import decimate

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
for p in (str(_REPO / "src"), str(_REPO / "benchmarks")):
    if p not in sys.path:
        sys.path.insert(0, p)

REQUEUE_RC = 85
SPLITS = ("train", "val", "test")
FS = 2048
SEG_LEN = 9216            # 58.75-63.25 s stored
WIN_START = 512           # 0.25 s into the segment -> 59 s
WIN_LEN = 8192            # 4 s
MAX_SHIFT = 512           # +-0.25 s
DECIM = 8                 # 2048 -> 256 Hz
DECIM_PAD = 128           # reflect-pad / margin (2048 Hz samples) against filter edge effects

MODELS = ("moment", "chronos", "timesfm", "timemoe", "moirai", "granite_ttm")
# Per-model token pooling of the Kepler / TESS / Project 8 FM runs (run_tess.py / run_project8.py
# _DEFAULT_POOL): encoders -> mean, causal decoders -> last token.
PAPER_POOL = {"moment": "mean", "chronos": "mean", "moirai": "mean", "granite_ttm": "mean",
              "timemoe": "last", "timesfm": "last"}
# MoiraiWrapper.load default horizon in those runs; their embed() hooked a horizon-128 forecast pass.
PAPER_MOIRAI_HORIZON = 128


# ── model loading ──────────────────────────────────────────────────────────────

def load_wrapper(name: str, device: str, chunk_len: int):
    if name == "moment":
        from foundation.wrappers.moment_wrapper import MomentWrapper as W
        w = W(); w.load(device=device, model_size="base", seq_len=chunk_len)
    elif name == "chronos":
        from foundation.wrappers.chronos_wrapper import ChronosWrapper as W
        w = W(); w.load(device=device, model_size="base")
    elif name == "timesfm":
        from foundation.wrappers.timesfm_wrapper import TimesFMWrapper as W
        w = W(); w.load(device=device, model_size="base", context_len=chunk_len, horizon=128)
    elif name == "timemoe":
        from foundation.wrappers.timemoe_wrapper import TimeMoEWrapper as W
        w = W(); w.load(device=device, model_size="base")
    elif name == "moirai":
        from foundation.wrappers.moirai_wrapper import MoiraiWrapper as W
        w = W(); w.load(device=device, model_size="base", context_len=chunk_len)
        w._module.to(device).eval()
    elif name == "granite_ttm":
        from foundation.wrappers.granite_ttm_wrapper import GraniteTTMWrapper as W
        w = W(); w.load(device=device, model_size="base")
        if w._ctx_len != chunk_len:
            raise ValueError(f"TTM context {w._ctx_len} != chunk_len {chunk_len}")
    else:
        raise ValueError(name)
    return w


# ── data ───────────────────────────────────────────────────────────────────────

def split_shifts(split: str, n: int, jitter_seed: int = 0) -> np.ndarray:
    """Identical to LIGODataset.fixed_shifts."""
    rng = np.random.default_rng(jitter_seed + {"train": 0, "val": 1, "test": 2}[split])
    return rng.integers(-MAX_SHIFT, MAX_SHIFT + 1, size=n)


def read_windows(f: h5py.File, a: int, b: int, shifts: np.ndarray, mode: str) -> np.ndarray:
    """Rows [a, b) -> (n, 2, L) float32 window(s), L = 8192 (2048hz) or 1024 (256hz)."""
    X = f["whitened_injected"][a:b]                               # (n, 2, 9216)
    assert X.shape[-1] == SEG_LEN, X.shape
    sh = shifts[a:b]
    if mode == "2048hz":
        idx = WIN_START + sh[:, None] + np.arange(WIN_LEN)[None, :]   # (n, L)
        return np.take_along_axis(X, idx[:, None, :], axis=-1).astype(np.float32)
    # 256hz: take window +- DECIM_PAD margin (reflect-padded at the segment edges), decimate,
    # then crop the margin back off.
    Xp = np.pad(X, ((0, 0), (0, 0), (DECIM_PAD, DECIM_PAD)), mode="reflect")
    idx = WIN_START + sh[:, None] + np.arange(WIN_LEN + 2 * DECIM_PAD)[None, :]
    W = np.take_along_axis(Xp, idx[:, None, :], axis=-1)
    D = decimate(W, DECIM, ftype="fir", zero_phase=True, axis=-1)
    m = DECIM_PAD // DECIM
    return np.ascontiguousarray(D[..., m:m + WIN_LEN // DECIM], dtype=np.float32)


def embed_kwargs(args) -> dict:
    """Extra wrapper.embed() kwargs; empty for the default (mean-pooled, no MOIRAI horizon) runs."""
    kw = {}
    if args.pool != "mean":
        kw["pool"] = args.pool
    if args.model == "moirai" and args.moirai_horizon:
        kw["horizon"] = args.moirai_horizon
    return kw


def embed_windows(wrapper, X: np.ndarray, chunk_len: int, batch: int, bf16: bool = False,
                  keep_chunks: bool = False, embed_kw: dict | None = None) -> np.ndarray:
    """(n, 2, L) -> (n, 2*d): per-chunk embeddings, mean over chunks, concat IFOs.
    keep_chunks: return the per-chunk embeddings (n, 2, k, d) as float16 instead (no pooling)."""
    n, c, L = X.shape
    assert L % chunk_len == 0
    k = L // chunk_len
    flat = X.reshape(n * c * k, chunk_len)
    outs = []
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16)
    with torch.inference_mode(), amp:
        for i in range(0, len(flat), batch):
            outs.append(np.asarray(wrapper.embed(flat[i:i + batch], **(embed_kw or {})).embeddings,
                                   dtype=np.float32))
    E = np.concatenate(outs).reshape(n, c, k, -1)
    if keep_chunks:
        return E.astype(np.float16)
    return E.mean(axis=2).reshape(n, -1)


def all_shards_done(args, out: Path) -> bool:
    for split in SPLITS:
        with h5py.File(Path(args.data_dir) / f"{split}.h5", "r") as f:
            n = f["whitened_injected"].shape[0]
        if args.max_samples:
            n = min(n, args.max_samples)
        k = -(-n // args.shard_size)
        if not all((out / "emb" / f"{split}_{i:04d}.npy").exists() for i in range(k)):
            return False
        if not (out / f"labels_{split}.npz").exists():
            return False
    return True


def claim_head(out: Path, head: str = "mlp") -> bool:
    """Exactly one worker trains the head: first to create head.lock wins; the owner
    (same condor job id) may retry after an eviction."""
    me = os.environ.get("CONDOR_JOB_ID", f"pid{os.getpid()}").split(".")[0]
    lock = out / ("head.lock" if head == "mlp" else f"head_{head}.lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, me.encode()); os.close(fd)
        return True
    except FileExistsError:
        return lock.read_text().strip() == me


def extract(args, out: Path, t0: float) -> bool:
    """Embed all splits shard by shard. Returns False if the time limit was hit."""
    wrapper = None
    pool = ThreadPoolExecutor(max_workers=1)
    for split in SPLITS:
        path = Path(args.data_dir) / f"{split}.h5"
        with h5py.File(path, "r") as f:
            n = f["whitened_injected"].shape[0]
            if args.max_samples:
                n = min(n, args.max_samples)
            shifts = split_shifts(split, f["whitened_injected"].shape[0])
            lab = out / f"labels_{split}.npz"
            if not lab.exists():
                # atomic: an eviction can't leave a partial file; per-worker tmp name so parallel
                # workers don't rename each other's file away
                tmp = lab.with_name(f"{lab.stem}.tmp{args.worker}.npz")
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
                wrapper = load_wrapper(args.model, args.device, args.chunk_len)
            fut = pool.submit(read_windows, f, todo[0][1], todo[0][2], shifts, args.input)
            for j, (i, a, b) in enumerate(todo):
                X = fut.result()
                if j + 1 < len(todo):
                    fut = pool.submit(read_windows, f, todo[j + 1][1], todo[j + 1][2], shifts, args.input)
                ts = time.time()
                E = embed_windows(wrapper, X, args.chunk_len, args.batch_size, args.amp_bf16, args.keep_chunks,
                                  embed_kwargs(args))
                if not np.isfinite(E).all():
                    bad = ~np.isfinite(E).reshape(len(E), -1).all(1)
                    print(f"  WARNING shard {split}_{i:04d}: {bad.sum()} rows non-finite")
                dst = out / "emb" / f"{split}_{i:04d}.npy"
                tmp = dst.with_suffix(".tmp.npy")
                np.save(tmp, E)
                os.replace(tmp, dst)
                el = time.time() - t0
                print(f"  {split} shard {i + 1}/{len(bounds)} rows {a}-{b} dim={E.shape[1:]} "
                      f"{(b - a) / (time.time() - ts):.0f} samples/s  elapsed {el / 3600:.2f} h", flush=True)
                if el > args.time_limit_h * 3600 and (j + 1 < len(todo) or split != SPLITS[-1]):
                    pool.shutdown(wait=True)
                    return False
    pool.shutdown()
    return True


# ── head ───────────────────────────────────────────────────────────────────────

class EmbeddingMLP(nn.Module):
    def __init__(self, d_embed: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_embed, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_split(out: Path, split: str):
    files = sorted((out / "emb").glob(f"{split}_[0-9][0-9][0-9][0-9].npy"))
    E = np.concatenate([np.load(p) for p in files])
    lab = np.load(out / f"labels_{split}.npz")
    assert len(E) == len(lab["y"]), (split, len(E), len(lab["y"]))
    return E, lab["y"], lab["snr"]


def train_and_eval(args, out: Path, res_dir: Path):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = args.device
    Etr, ytr, _ = load_split(out, "train")
    Eva, yva, snr_va = load_split(out, "val")
    Ete, yte, snr_te = load_split(out, "test")
    # Non-finite embedding rows (should not happen) are zeroed after standardisation.
    mu = np.nanmean(Etr, axis=0, dtype=np.float64)
    sd = np.nanstd(Etr, axis=0, dtype=np.float64) + 1e-6
    ym, ys = float(ytr.mean()), float(ytr.std())

    def prep(E):
        Z = ((E - mu) / sd).astype(np.float32)
        Z[~np.isfinite(Z)] = 0.0
        return torch.from_numpy(Z).to(dev)

    Xtr, Xva, Xte = prep(Etr), prep(Eva), prep(Ete)
    del Etr, Eva, Ete
    Ytr = torch.from_numpy((ytr - ym) / ys).to(dev)
    Yva = torch.from_numpy((yva - ym) / ys).to(dev)

    head = EmbeddingMLP(Xtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.head_lr, weight_decay=0.01)
    crit = nn.MSELoss()

    def predict(X):
        head.eval()
        with torch.no_grad():
            return torch.cat([head(X[i:i + 65536]) for i in range(0, len(X), 65536)])

    best, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(args.head_epochs):
        head.train()
        perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(perm), args.head_batch_size):
            b = perm[i:i + args.head_batch_size]
            opt.zero_grad(set_to_none=True)
            crit(head(Xtr[b]), Ytr[b]).backward()
            opt.step()
        v = crit(predict(Xva), Yva).item()
        hist.append(v)
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in head.state_dict().items()}
        else:
            bad += 1
        print(f"  head epoch {ep + 1}: val_mse(std units)={v:.5f}  best={best:.5f}", flush=True)
        if bad >= args.head_patience:
            break
    head.load_state_dict(best_state)

    pva = predict(Xva).cpu().numpy() * ys + ym
    pte = predict(Xte).cpu().numpy() * ys + ym
    val_mse = float(np.mean((pva - yva) ** 2))

    def metrics(y, p):
        e = p - y
        return dict(rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))),
                    r2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)), n=int(len(y)))

    res = dict(model=args.model, input=args.input, chunk_len=args.chunk_len,
               embed_dim=int(Xtr.shape[1]), head_epochs_run=len(hist),
               best_epoch=int(np.argmin(hist)) + 1, n_train=int(len(ytr)),
               val=metrics(yva, pva), test=metrics(yte, pte),
               mean_baseline_test_rmse=float(np.sqrt(np.mean((yte - ym) ** 2))))
    res_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(snr=snr_te, true_chirp_mass=yte, pred_chirp_mass_mean=pte,
                      pred_chirp_mass_logvar=np.full_like(pte, np.log(val_mse)))
                 ).to_csv(res_dir / "test_predictions.csv", index=False)
    pd.DataFrame(dict(snr=snr_va, true_chirp_mass=yva, pred_chirp_mass_mean=pva)
                 ).to_csv(res_dir / "val_predictions.csv", index=False)
    (res_dir / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res


# ── ridge linear probe (paper protocol, --head ridge) ──────────────────────────────────────────

def train_and_eval_ridge(args, out: Path, res_dir: Path):
    """Linear probe of evaluators/regression.py (TESS) / run_project8.py, i.e. the head behind the
    other datasets' zero-shot FM rows: features standardised with train mean / std (+1e-6), target
    standardised with train mean / std, sklearn-Ridge(alpha, fit_intercept=True) fitted on train only
    (val is not used for fitting or model selection). Solved in closed form from shard-streamed
    float64 sufficient statistics, since the 720k x 2d train matrix does not fit in host memory."""
    dev = args.device

    def shard_files(split):
        return sorted((out / "emb").glob(f"{split}_[0-9][0-9][0-9][0-9].npy"))

    def load(p):
        return torch.from_numpy(np.load(p)).to(dev, torch.float64)

    # pass 1: per-feature mean / std over finite train entries (ddof=0, like np.std)
    s = ss = cnt = None
    fit = args.probe_fit   # split the probe and its standardisation statistics are fitted on
    for p in shard_files(fit):
        E = load(p)
        fin = torch.isfinite(E)
        E = torch.where(fin, E, torch.zeros((), dtype=E.dtype, device=dev))
        if s is None:
            s, ss, cnt = (torch.zeros(E.shape[1], dtype=torch.float64, device=dev) for _ in range(3))
        s += E.sum(0); ss += (E * E).sum(0); cnt += fin.sum(0)
    mu = s / cnt
    sd = (ss / cnt - mu ** 2).clamp_min(0).sqrt() + 1e-6

    def z(E):  # non-finite entries -> 0 after standardisation (same as the MLP head)
        Z = (E - mu) / sd
        return torch.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    lab = {sp: np.load(out / f"labels_{sp}.npz") for sp in SPLITS}
    ytr = lab[fit]["y"].astype(np.float64)
    ym, ys = float(ytr.mean()), float(ytr.std() + 1e-8)

    # pass 2: Z'Z, Z'y, column sums (sklearn centres X and y before solving)
    d = mu.numel()
    A = torch.zeros(d, d, dtype=torch.float64, device=dev)
    b = torch.zeros(d, dtype=torch.float64, device=dev)
    zs = torch.zeros(d, dtype=torch.float64, device=dev)
    yz = torch.from_numpy((ytr - ym) / ys).to(dev)
    i = 0
    for p in shard_files(fit):
        Z = z(load(p))
        yb = yz[i:i + len(Z)]; i += len(Z)
        A += Z.T @ Z; b += Z.T @ yb; zs += Z.sum(0)
    n = i
    assert n == len(ytr), (n, len(ytr))
    zm, yzm = zs / n, yz.mean()
    Ac = A - n * torch.outer(zm, zm)
    bc = b - n * zm * yzm
    w = torch.linalg.solve(Ac + args.ridge_alpha * torch.eye(d, dtype=torch.float64, device=dev), bc)
    c = yzm - zm @ w
    syy = ((yz - yzm) ** 2).sum()
    train_r2 = float(1 - (syy - 2 * (w @ bc) + w @ Ac @ w) / syy)   # SSR from centred statistics

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

    res = dict(model=args.model, input=args.input, chunk_len=args.chunk_len, head="ridge", probe_fit=fit,
               ridge_alpha=args.ridge_alpha, pool=args.pool,
               moirai_horizon=args.moirai_horizon if args.model == "moirai" else None,
               embed_dim=int(d), n_train=int(n), train_r2=train_r2,
               val=metrics(yva, pva), test=metrics(yte, pte),
               mean_baseline_test_rmse=float(np.sqrt(np.mean((yte - ym) ** 2))),
               test_mean_baseline_rmse=float(np.std(yte)))
    res_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(snr=snr_te, true_chirp_mass=yte, pred_chirp_mass_mean=pte,
                      pred_chirp_mass_logvar=np.full_like(pte, np.log(val_mse)))
                 ).to_csv(res_dir / "test_predictions.csv", index=False)
    pd.DataFrame(dict(snr=snr_va, true_chirp_mass=yva, pred_chirp_mass_mean=pva)
                 ).to_csv(res_dir / "val_predictions.csv", index=False)
    (res_dir / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res


# ── order-aware head on per-chunk embeddings (ablation, --keep_chunks) ─────────────────────────

class ChunkHead(nn.Module):
    """(B, c, k, d) standardised chunk embeddings -> chirp mass. Chunk order is kept because the
    chunk embeddings are concatenated by position (no pooling). 'proj' first maps every chunk with a
    shared Linear(d, p) + LayerNorm + GELU so the concat stays small for many chunks."""

    def __init__(self, c: int, k: int, d: int, mode: str, p: int, dropout: float = 0.1):
        super().__init__()
        if mode == "proj":
            self.proj = nn.Sequential(nn.Linear(d, p), nn.LayerNorm(p), nn.GELU())
            width = c * k * p
        else:
            self.proj = nn.Identity()
            width = c * k * d
        self.mlp = EmbeddingMLP(width, dropout)

    def forward(self, x):
        return self.mlp(self.proj(x).flatten(1))


def _load_chunks(out: Path, split: str):
    """Concatenate fp16 shards (n, c, k, d) into one preallocated host array."""
    files = sorted((out / "emb").glob(f"{split}_[0-9][0-9][0-9][0-9].npy"))
    shapes = [np.load(p, mmap_mode="r").shape for p in files]
    E = np.empty((sum(s[0] for s in shapes),) + shapes[0][1:], dtype=np.float16)
    i = 0
    for p, s in zip(files, shapes):
        E[i:i + s[0]] = np.load(p); i += s[0]
    lab = np.load(out / f"labels_{split}.npz")
    assert len(E) == len(lab["y"]), (split, len(E), len(lab["y"]))
    return E, lab["y"], lab["snr"]


def train_and_eval_chunks(args, out: Path, res_dir: Path):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = args.device
    Etr, ytr, _ = _load_chunks(out, "train")
    n, c, k, d = Etr.shape
    mode = args.chunk_head if args.chunk_head != "auto" else ("flat" if c * k <= 4 else "proj")
    print(f"chunk head: mode={mode} c={c} k={k} d={d}", flush=True)
    # per-feature (IFO, chunk position, dim) standardisation with train statistics, float64 sums
    s = np.zeros((c, k, d)); ss = np.zeros((c, k, d))
    for a in range(0, n, 50000):
        B = np.nan_to_num(Etr[a:a + 50000].astype(np.float64))
        s += B.sum(0); ss += (B * B).sum(0)
    mu_np = s / n
    sd_np = np.sqrt(np.maximum(ss / n - mu_np ** 2, 0)) + 1e-6
    mu = torch.from_numpy(mu_np).float().to(dev)
    sd = torch.from_numpy(sd_np).float().to(dev)
    ym, ys = float(ytr.mean()), float(ytr.std())

    def to_dev(B):
        Z = (torch.from_numpy(np.ascontiguousarray(B)).to(dev).float() - mu) / sd
        return torch.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    Eva, yva, snr_va = _load_chunks(out, "val")
    Ytr = torch.from_numpy((ytr - ym) / ys).float()
    Yva = torch.from_numpy((yva - ym) / ys).float().to(dev)

    head = ChunkHead(c, k, d, mode, args.chunk_proj_dim).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.head_lr, weight_decay=0.01)
    crit = nn.MSELoss()

    def predict(E):
        head.eval()
        with torch.no_grad():
            return torch.cat([head(to_dev(E[i:i + 8192])) for i in range(0, len(E), 8192)])

    best, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(args.head_epochs):
        head.train()
        perm = np.random.permutation(n)
        for i in range(0, n, args.head_batch_size):
            b = np.sort(perm[i:i + args.head_batch_size])
            opt.zero_grad(set_to_none=True)
            crit(head(to_dev(Etr[b])), Ytr[b].to(dev)).backward()
            opt.step()
        v = crit(predict(Eva), Yva).item()
        hist.append(v)
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {kk: t.detach().clone() for kk, t in head.state_dict().items()}
        else:
            bad += 1
        print(f"  head epoch {ep + 1}: val_mse(std units)={v:.5f}  best={best:.5f}", flush=True)
        if bad >= args.head_patience:
            break
    head.load_state_dict(best_state)
    del Etr
    pva = predict(Eva).cpu().numpy() * ys + ym
    del Eva
    # test: shard by shard, never fully in memory
    p_parts = []
    lab = np.load(out / "labels_test.npz")
    yte, snr_te = lab["y"], lab["snr"]
    for p in sorted((out / "emb").glob("test_[0-9][0-9][0-9][0-9].npy")):
        p_parts.append(predict(np.load(p)).cpu().numpy())
    pte = np.concatenate(p_parts) * ys + ym
    assert len(pte) == len(yte)
    val_mse = float(np.mean((pva - yva) ** 2))

    def metrics(y, p):
        e = p - y
        return dict(rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))),
                    r2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)), n=int(len(y)))

    res = dict(model=args.model, input=args.input, chunk_len=args.chunk_len, keep_chunks=True,
               chunk_head=mode, chunk_proj_dim=args.chunk_proj_dim if mode == "proj" else None,
               emb_shape=[c, k, d], head_params=int(sum(t.numel() for t in head.parameters())),
               head_epochs_run=len(hist), best_epoch=int(np.argmin(hist)) + 1, n_train=int(n),
               val=metrics(yva, pva), test=metrics(yte, pte),
               mean_baseline_test_rmse=float(np.sqrt(np.mean((yte - ym) ** 2))))
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--input", required=True, choices=("256hz", "2048hz"))
    ap.add_argument("--chunk_len", type=int, default=512)
    ap.add_argument("--data_dir", default="data/LIGO/2048Hz_v2")
    ap.add_argument("--out_root", default="/fast/barmstrong/outputs/ligo_fm")
    ap.add_argument("--results_root", default="results/LIGO",
                    help="test_predictions.csv / metrics.json go to <results_root>/v2_fm_<model>_<input>[_c<chunk_len>]")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=512, help="chunks per forward pass")
    ap.add_argument("--amp_bf16", action="store_true",
                    help="bf16 autocast for the frozen backbone (weights stay fp32)")
    ap.add_argument("--shard_size", type=int, default=20000, help="samples per embedding shard")
    ap.add_argument("--max_samples", type=int, default=None, help="per split (smoke tests)")
    ap.add_argument("--time_limit_h", type=float, default=4.0,
                    help=f"stop extraction after this long and exit {REQUEUE_RC} (requeue)")
    ap.add_argument("--stage", choices=("all", "embed", "head"), default="all")
    ap.add_argument("--worker", type=int, default=0, help="this worker's index (multi-GPU split)")
    ap.add_argument("--num_workers", type=int, default=1,
                    help="split shards round-robin over this many jobs; the last to finish trains the head")
    ap.add_argument("--head_epochs", type=int, default=100)
    ap.add_argument("--head_patience", type=int, default=10)
    ap.add_argument("--head_batch_size", type=int, default=1024)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep_chunks", action="store_true",
                    help="ablation: store per-chunk embeddings (fp16) and train an order-aware head "
                         "instead of mean-pooling over chunks; outputs get a _chunks suffix")
    ap.add_argument("--chunk_head", choices=("auto", "flat", "proj"), default="auto",
                    help="flat: concat all chunk embeddings -> MLP; proj: shared Linear(d, chunk_proj_dim) "
                         "per chunk, then concat -> MLP; auto: flat if <= 4 chunks in total, else proj")
    ap.add_argument("--chunk_proj_dim", type=int, default=64)
    # ── consistency with the Kepler / TESS / Project 8 zero-shot FM runs ──
    ap.add_argument("--pool", choices=("mean", "last", "auto"), default="mean",
                    help="token pooling inside each chunk; auto = PAPER_POOL (last for Time-MoE / TimesFM)")
    ap.add_argument("--moirai_horizon", type=int, default=0,
                    help="append horizon/patch prediction-mask tokens like a MOIRAI forecast pass")
    ap.add_argument("--head", choices=("mlp", "ridge"), default="mlp")
    ap.add_argument("--ridge_alpha", type=float, default=1.0)
    ap.add_argument("--probe_fit", choices=("train", "val"), default="train",
                    help="split the ridge probe is fitted (and standardised) on; val = diagnostic fit on the test-distribution (O3a, low-SNR) validation set, results get a _valfit suffix")
    ap.add_argument("--paper_protocol", action="store_true",
                    help="= --pool auto --moirai_horizon 128 --head ridge --ridge_alpha 1 (the protocol "
                         "behind the other datasets' zero-shot FM rows)")
    args = ap.parse_args()
    if args.paper_protocol:
        args.pool, args.moirai_horizon, args.head, args.ridge_alpha = "auto", PAPER_MOIRAI_HORIZON, "ridge", 1.0
    if args.pool == "auto":
        args.pool = PAPER_POOL[args.model]
    if args.model != "moirai":
        args.moirai_horizon = 0
    if args.head == "ridge" and args.keep_chunks:
        ap.error("--head ridge is only implemented for chunk-mean-pooled embeddings (no --keep_chunks)")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # embedding dir tag; the default (mean-pooled, no horizon, 512) runs keep their original paths
    tag = f"{args.model}_{args.input}"
    if args.chunk_len != 512:
        tag += f"_c{args.chunk_len}"
    if args.pool != "mean":
        tag += f"_{args.pool}"
    if args.moirai_horizon:
        tag += f"_h{args.moirai_horizon}"
    if args.keep_chunks:
        tag += "_chunks"
    out = Path(args.out_root) / tag
    (out / "emb").mkdir(parents=True, exist_ok=True)
    cfg = "config.json" if args.stage != "head" else f"config_head_{args.head}_{args.probe_fit}.json"
    (out / cfg).write_text(json.dumps(vars(args), indent=2))
    t0 = time.time()

    if args.stage in ("all", "embed"):
        if not extract(args, out, t0):
            print(f"time limit {args.time_limit_h} h reached, exiting {REQUEUE_RC} for requeue", flush=True)
            sys.exit(REQUEUE_RC)
        print(f"embeddings complete ({(time.time() - t0) / 3600:.2f} h this slice)", flush=True)
    if args.stage in ("all", "head"):
        if args.num_workers > 1:
            if not all_shards_done(args, out):
                print("other workers' shards still missing; leaving the head to the last worker", flush=True)
                return
            if not claim_head(out, args.head):
                print("head already claimed by another worker", flush=True)
                return
        res_tag = tag + ("_probe" if args.head == "ridge" else "")
        if args.head == "ridge" and args.probe_fit == "val":
            res_tag += "_valfit"
        res_dir = Path(args.results_root) / f"v2_fm_{res_tag}"
        if args.max_samples:
            res_dir = out / f"results{'_probe' if args.head == 'ridge' else ''}"
        if args.head == "ridge":
            train_and_eval_ridge(args, out, res_dir)
        else:
            (train_and_eval_chunks if args.keep_chunks else train_and_eval)(args, out, res_dir)


if __name__ == "__main__":
    main()
