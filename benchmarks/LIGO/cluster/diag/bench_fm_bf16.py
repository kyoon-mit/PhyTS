"""fp32 vs bf16-autocast embedding throughput + agreement on 64 LIGO test samples at 2048 Hz.
Run from ~/TimeSeriesPhysics with the FM venv python."""
import sys, time, numpy as np, h5py, torch
sys.path.insert(0, "benchmarks/LIGO"); sys.path.insert(0, "benchmarks"); sys.path.insert(0, "src")
import fm_zeroshot_ligo as m
torch.backends.cuda.matmul.allow_tf32 = True
with h5py.File("data/LIGO/2048Hz_v2/test.h5") as f:
    X = m.read_windows(f, 0, 64, m.split_shifts("test", f["whitened_injected"].shape[0]), "2048hz")
for name in sys.argv[1:]:
    w = m.load_wrapper(name, "cuda", 512)
    out = {}
    for bf in (False, True):
        m.embed_windows(w, X[:8], 512, 512, bf); torch.cuda.synchronize()
        t = time.time(); E = m.embed_windows(w, X, 512, 512, bf); torch.cuda.synchronize()
        out[bf] = E; print(f"{name} bf16={bf}: {len(X) / (time.time() - t):.1f} samples/s", flush=True)
    a, b = out[False], out[True]
    cos = np.sum(a * b, 1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    print(f"{name}: cosine(fp32,bf16) min {cos.min():.5f} mean {cos.mean():.5f}; rel L2 {np.linalg.norm(a - b) / np.linalg.norm(a):.4f}", flush=True)
    del w; torch.cuda.empty_cache()
