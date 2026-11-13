#!/bin/bash
# Smoke test for fm_zeroshot_ligo.py --paper_protocol (per-model pooling, MOIRAI horizon tokens,
# ridge alpha=1 probe): (1) embedding paths vs the Kepler/TESS/P8 hook-based extraction,
# (2) every FM x input end to end on 1024 samples/split, (3) streamed ridge vs sklearn Ridge.
set -uo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false HF_HOME=/fast/barmstrong/hf/
cd /home/barmstrong/TimeSeriesPhysics
PY=benchmarks/foundation/.venv/bin/python
OUT=/fast/barmstrong/outputs/ligo_fm_smoke_paper; rm -rf "$OUT"
SUMMARY=()
echo "================ embedding-path checks $(date)"
$PY /lustre/home/barmstrong/phyts/cluster/verify_fm_paper_protocol.py; SUMMARY+=("verify rc=$?")
for inp in 256hz 2048hz; do for m in moment chronos timesfm timemoe moirai granite_ttm; do
  echo "================ $m $inp $(date)"
  $PY benchmarks/LIGO/fm_zeroshot_ligo.py --model $m --input $inp --paper_protocol \
    --max_samples 1024 --shard_size 512 --out_root $OUT
  RC=$?; SUMMARY+=("$m $inp rc=$RC")
done; done
echo "================ ridge vs sklearn $(date)"
$PY - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
bad = 0
for d in sorted(Path(sys.argv[1]).iterdir()):
    m = d / "results_probe" / "metrics.json"
    if not m.exists():
        print("no result:", d.name); bad += 1; continue
    load = lambda s: np.concatenate([np.load(p) for p in sorted((d / "emb").glob(f"{s}_[0-9][0-9][0-9][0-9].npy"))])
    lab = lambda s: np.load(d / f"labels_{s}.npz")["y"].astype(np.float64)
    Xtr, Xte, ytr, yte = load("train").astype(np.float64), load("test").astype(np.float64), lab("train"), lab("test")
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6     # evaluators/regression.py
    ym, ys = ytr.mean(), ytr.std() + 1e-8
    p = Ridge(alpha=1.0).fit((Xtr - mu) / sd, (ytr - ym) / ys).predict((Xte - mu) / sd) * ys + ym
    r = json.loads(m.read_text())
    rmse = float(np.sqrt(np.mean((p - yte) ** 2)))
    ok = abs(rmse - r["test"]["rmse"]) < 1e-4 * max(rmse, 1e-6) + 1e-6
    bad += not ok
    print(f"{'PASS' if ok else 'FAIL'} {d.name}: pool={r['pool']} h={r['moirai_horizon']} d={r['embed_dim']} "
          f"ours rmse={r['test']['rmse']:.6f} sklearn={rmse:.6f}")
sys.exit(1 if bad else 0)
EOF
SUMMARY+=("sklearn-check rc=$?")
printf '%s\n' "${SUMMARY[@]}"
/home/barmstrong/cluster/htcondor/notify.sh "🧪 FM paper-protocol smoke: $(printf '%s; ' "${SUMMARY[@]}")"
