"""Check the LIGO FM --paper_protocol embedding paths against the hook-based extraction used by the
Kepler / TESS / Project 8 runs (commit 82188e1 wrappers), on real LIGO 512-sample chunks.

* MOIRAI  embed(horizon=128): hook encoder.norm during a patch-32 horizon-128 forecast, mean over
  the 16 context tokens (82188e1 MoiraiWrapper.embed).
* TimesFM embed(pool="last"): hook stacked_transformer during the public forecast API, last patch
  (82188e1 TimesFMWrapper.embed).
* Time-MoE embed(pool="last"): full model hidden_states[-1][:, -1] (82188e1 TimeMoEWrapper.embed).

Run from ~/TimeSeriesPhysics with the FM venv on a GPU node.
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path[:0] = ["benchmarks", "src"]
dev = "cuda"
B = 8
with h5py.File("data/LIGO/2048Hz_v2/val.h5", "r") as f:
    X = f["whitened_injected"][:B // 2, :, 512:512 + 512].reshape(B, 512).astype(np.float32)


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))


ok = True


def report(name, a, b, tol):
    global ok
    r = rel(a, b)
    good = r < tol
    ok &= good
    print(f"{'PASS' if good else 'FAIL'} {name}: shape {np.shape(a)} vs {np.shape(b)}, max rel diff {r:.2e} (tol {tol})",
          flush=True)


# ── MOIRAI ──
from foundation.wrappers.moirai_wrapper import MoiraiWrapper
w = MoiraiWrapper(); w.load(device=dev, model_size="base", context_len=512, horizon=128, patch_size=32)
w._module.to(dev).eval()
cap = []
hd = w._module.encoder.norm.register_forward_hook(lambda m, i, o: cap.append((o[0] if isinstance(o, tuple) else o).detach()))
try:
    w.forecast(X, horizon=128)
finally:
    hd.remove()
ref = cap[0][:, :16].mean(1).float().cpu().numpy()
new = w.embed(X, horizon=128).embeddings
old = w.embed(X).embeddings
report("moirai horizon=128 vs forecast hook", new, ref, 1e-3)
print(f"     (old no-horizon embed differs from forecast hook by {rel(old, ref):.2e})")
del w; torch.cuda.empty_cache()

# ── TimesFM ──
from foundation.wrappers.timesfm_wrapper import TimesFMWrapper
w = TimesFMWrapper(); w.load(device=dev, model_size="base", context_len=512, horizon=128)
inner = w.model._model
cap = []
hd = inner.stacked_transformer.register_forward_hook(lambda m, i, o: cap.append((o[0] if isinstance(o, tuple) else o).detach()))
try:
    w._forecast_fn([X[i] for i in range(B)])
finally:
    hd.remove()
print(f"     timesfm hook captured {len(cap)} call(s), shapes {[tuple(c.shape) for c in cap]}")
h = cap[-1][:B]
report("timesfm pool=last vs forecast hook", w.embed(X, pool="last").embeddings, h[:, -1].float().cpu().numpy(), 1e-3)
del w; torch.cuda.empty_cache()

# ── Time-MoE ──
from foundation.wrappers.timemoe_wrapper import TimeMoEWrapper
w = TimeMoEWrapper(); w.load(device=dev, model_size="base")
normed, _, _ = w._instance_normalize(X)
with torch.no_grad():
    hs = w.model(torch.from_numpy(normed).to(dev), output_hidden_states=True).hidden_states[-1]
report("timemoe pool=last vs full-model hidden_states[-1][:, -1]", w.embed(X, pool="last").embeddings,
       hs[:, -1].float().cpu().numpy(), 1e-3)

print("ALL PASS" if ok else "SOME FAILED", flush=True)
sys.exit(0 if ok else 1)
