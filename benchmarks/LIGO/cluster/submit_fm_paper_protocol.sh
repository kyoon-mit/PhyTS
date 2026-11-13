#!/bin/bash
# LIGO v2 zero-shot FMs under --paper_protocol (same protocol as the Kepler / TESS / Project 8 FM rows):
#  * MOMENT / Chronos / TTM: pooling unchanged (mean) -> reuse the existing embeddings, ridge head only
#    (one job, --stage head for all 6 model x input pairs).
#  * Time-MoE / TimesFM (last-token pooling) and MOIRAI (horizon-128 tokens): new embeddings,
#    Time-MoE split over several H100s (round-robin shards; the last worker fits the probe).
# H100 @ 35, 4 h slices requeued via exit 85.
# Usage: submit_fm_paper_protocol.sh [-n]   (-n: dry run, print the submissions only)
DRY=0; [[ ${1:-} == -n ]] && DRY=1
H100='(TARGET.CUDADeviceName == "NVIDIA H100 80GB HBM3" || TARGET.CUDADeviceName == "NVIDIA H100")'
SUB=/home/barmstrong/TimeSeriesPhysics/cluster/standard.sub
LOGS=/fast/barmstrong/outputs
JOB=/lustre/home/barmstrong/phyts/cluster/fm_zeroshot_ligo.sh

submit() {  # name, arguments
  local name=$1; shift
  if [[ $DRY == 1 ]]; then echo "[dry] $name: $*"; return; fi
  condor_submit_bid 35 -append "arguments = $*" -append "requirements = $H100" \
    -append "MaxTime = 21600" -append "on_exit_remove = (ExitCode =!= 85)" \
    -append "output = $LOGS/ligo_fm_paper_${name}.\$(Cluster).out" \
    -append "error = $LOGS/ligo_fm_paper_${name}.\$(Cluster).err" "$SUB" 2>&1 \
    | grep -o 'cluster [0-9]*' | sed "s/^/$name -> /"
}

# ridge probe on the existing mean-pooled embeddings (no extraction needed)
submit probe_reuse /lustre/home/barmstrong/phyts/cluster/fm_paper_probe_reuse.sh

# new embeddings
for inp in 256hz 2048hz; do
  submit timesfm_${inp} $JOB timesfm $inp --paper_protocol
  submit moirai_${inp}  $JOB moirai  $inp --paper_protocol
done
submit timemoe_256hz_w0 $JOB timemoe 256hz --paper_protocol --worker 0 --num_workers 2
submit timemoe_256hz_w1 $JOB timemoe 256hz --paper_protocol --worker 1 --num_workers 2
for w in $(seq 0 11); do
  submit timemoe_2048hz_w$w $JOB timemoe 2048hz --paper_protocol --worker $w --num_workers 12
done
