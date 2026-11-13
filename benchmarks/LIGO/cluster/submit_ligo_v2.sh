#!/bin/bash
# Wait for the v2 train/val split files (extraction cluster 17824252), then submit one
# condor GPU job per config in configs/LIGO/v2 (S4D/LinOSS at 3 LRs, CNN full + size-matched).
# test.h5 is only read after training, so it may still be extracting when jobs start.
#
# GPU bids: a bid is fixed at submit time (condor_submit_bid N -> priority = N - 1000), so jobs
# start restricted to A100-80GB at bid 25. Any job idle for > 30 min (also after an eviction
# requeue) is switched to H100 at bid 35 via condor_qedit. B200 is never used.
# Runs until all submitted jobs have left the queue. Sends Telegram notifications.
NOTIFY=/home/barmstrong/cluster/htcondor/notify.sh
D=/fast/barmstrong/LIGO_2048Hz_v2
EXTRACT=17824252
A100='(TARGET.CUDADeviceName == "NVIDIA A100-SXM4-80GB")'
H100='(TARGET.CUDADeviceName == "NVIDIA H100 80GB HBM3" || TARGET.CUDADeviceName == "NVIDIA H100")'
Q=/usr/bin/condor_q_raw  # live queue; /usr/local/bin/condor_q serves a copy up to 60 s old
BID_A100=25
BID_H100=35
IDLE_SWITCH=1800  # seconds idle on A100 before moving to H100

until [[ -f $D/train.h5 && -f $D/val.h5 ]]; do
  if ! $Q $EXTRACT -af JobStatus 2>/dev/null | grep -q .; then
    sleep 60  # condor_q can briefly return nothing; re-check before giving up
    if ! $Q $EXTRACT -af JobStatus 2>/dev/null | grep -q . && ! [[ -f $D/train.h5 && -f $D/val.h5 ]]; then
      echo "extraction jobs gone but files missing"
      $NOTIFY "❌ LIGO v2: extraction jobs ${EXTRACT} gone but train/val files missing — nothing submitted"
      exit 1
    fi
  fi
  sleep 120
done
$NOTIFY "📦 LIGO v2: train.h5 and val.h5 ready, submitting 8 training jobs (A100 @ ${BID_A100})"

cd /lustre/home/barmstrong/phyts/cluster
RUNS=(s4d_lr3e-4 s4d_lr1e-3 s4d_lr3e-3 linoss_lr3e-4 linoss_lr1e-3 linoss_lr3e-3 conv1d conv1d_small)
OK=(); FAILED=(); CLUSTERS=()
for r in "${RUNS[@]}"; do
  if OUT=$(condor_submit_bid $BID_A100 \
    -append "arguments = /lustre/home/barmstrong/phyts/cluster/train_ligo_cfg.sh configs/LIGO/v2/train_ligo_${r}_gaussnll_regression.yaml" \
    -append "requirements = $A100" \
    -append "output = /fast/barmstrong/outputs/ligo_v2_$r.\$(Cluster).out" \
    -append "error = /fast/barmstrong/outputs/ligo_v2_$r.\$(Cluster).err" \
    /home/barmstrong/TimeSeriesPhysics/cluster/standard.sub 2>&1); then
    C=$(grep -o 'cluster [0-9]*' <<<"$OUT" | awk '{print $2}')
    echo "$r -> $C"; OK+=("$r=$C"); CLUSTERS+=("$C")
  else
    echo "$OUT"; FAILED+=("$r")
  fi
done
MSG="🚀 LIGO v2: submitted ${#OK[@]}/${#RUNS[@]} jobs (A100 @ ${BID_A100}; idle > $((IDLE_SWITCH / 60)) min -> H100 @ ${BID_H100})"$'\n'"$(printf '%s\n' "${OK[@]}")"
[[ ${#FAILED[@]} -gt 0 ]] && MSG+=$'\n'"❌ failed to submit: ${FAILED[*]}"
$NOTIFY "$MSG"
echo "submitted $(date)"

# Move jobs idle too long on A100 to H100 (with the H100 bid). JobStatus 1 = idle.
while [[ ${#CLUSTERS[@]} -gt 0 ]]; do
  sleep 300
  LEFT=()
  for C in "${CLUSTERS[@]}"; do
    AD=$($Q "$C" -af JobStatus EnteredCurrentStatus 2>/dev/null)
    if [[ -z $AD ]]; then
      # gone from the queue (finished or removed) -- double-check before dropping it
      sleep 5; AD=$($Q "$C" -af JobStatus EnteredCurrentStatus 2>/dev/null)
      [[ -z $AD ]] && continue
    fi
    LEFT+=("$C")
    read -r STATUS SINCE <<<"$AD"
    [[ $STATUS == 1 && $(( $(date +%s) - SINCE )) -gt $IDLE_SWITCH ]] || continue
    REQ=$($Q "$C" -af:r Requirements)
    [[ $REQ == *A100-SXM4-80GB* ]] || continue  # already moved
    NEWREQ=${REQ//$A100/$H100}
    if [[ $NEWREQ != "$REQ" ]] && condor_qedit "$C" Requirements "$NEWREQ" >/dev/null \
        && condor_qedit "$C" JobPrio $((BID_H100 - 1000)) >/dev/null; then
      NAME=$($Q "$C" -af Args | grep -o 'train_ligo_[^ ]*\.yaml' | head -1)
      echo "$(date) moved $C to H100"
      $NOTIFY "🔀 ${C} idle > $((IDLE_SWITCH / 60)) min on A100, moved to H100 @ ${BID_H100}: ${NAME}"
    else
      echo "$(date) failed to move $C (requirements: $REQ)"
      $NOTIFY "⚠️ ${C}: could not move from A100 to H100 — check condor_q -l ${C}"
    fi
  done
  CLUSTERS=("${LEFT[@]}")
done
echo "all jobs left the queue $(date)"
