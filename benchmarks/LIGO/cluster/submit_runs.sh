#!/bin/bash
# Submit LIGO v2 training runs and keep each on the cheapest GPU that is free.
# Usage: submit_runs.sh <run> [<run> ...]   e.g. submit_runs.sh linoss_ssm_lr1e-3 s4d_lr1e-3
#   <run> selects configs/LIGO/v2/train_ligo_<run>_gaussnll_regression.yaml
# Jobs start on A100-80GB at bid 25; any job idle > 30 min (also after an eviction requeue) is
# moved to H100 at bid 35 via condor_qedit. B200 is never used. Runs until all jobs have left
# the queue. Jobs run in time slices: JobTimer stops training after 4 h (at an epoch boundary),
# train_ligo_cfg.sh exits 85 and on_exit_remove requeues the job, which resumes from last.ckpt.
# MaxTime (6 h) is only a hard backstop. A job that was moved to H100 is reset to A100 @ bid 25
# while it runs, so its next slice tries the cheaper GPU first. Start it detached:  setsid nohup ./submit_runs.sh ... > log 2>&1 < /dev/null &
NOTIFY=/home/barmstrong/cluster/htcondor/notify.sh
A100='(TARGET.CUDADeviceName == "NVIDIA A100-SXM4-80GB")'
H100='(TARGET.CUDADeviceName == "NVIDIA H100 80GB HBM3" || TARGET.CUDADeviceName == "NVIDIA H100")'
Q=/usr/bin/condor_q_raw  # live queue; /usr/local/bin/condor_q serves a copy up to 60 s old
BID_A100=25
BID_H100=35
IDLE_SWITCH=1800
MAX_TIME=21600     # hard per-slice limit in s (periodic_remove in standard.sub); slices are 4 h
REQUEUE_RC=85      # exit code of train_ligo_cfg.sh when a time slice ended (not the run)
CFGDIR=/home/barmstrong/TimeSeriesPhysics/configs/LIGO/v2

[[ $# -gt 0 ]] || { echo "usage: $0 <run> [<run> ...]"; exit 1; }
for r in "$@"; do
  [[ -f $CFGDIR/train_ligo_${r}_gaussnll_regression.yaml ]] || { echo "missing config for $r"; exit 1; }
done

cd /lustre/home/barmstrong/phyts/cluster
OK=(); FAILED=(); CLUSTERS=()
for r in "$@"; do
  if OUT=$(condor_submit_bid $BID_A100 \
    -append "arguments = /lustre/home/barmstrong/phyts/cluster/train_ligo_cfg.sh configs/LIGO/v2/train_ligo_${r}_gaussnll_regression.yaml" \
    -append "requirements = $A100" \
    -append "MaxTime = $MAX_TIME" \
    -append "on_exit_remove = (ExitCode =!= $REQUEUE_RC)" \
    -append "output = /fast/barmstrong/outputs/ligo_v2_$r.\$(Cluster).out" \
    -append "error = /fast/barmstrong/outputs/ligo_v2_$r.\$(Cluster).err" \
    /home/barmstrong/TimeSeriesPhysics/cluster/standard.sub 2>&1); then
    C=$(grep -o 'cluster [0-9]*' <<<"$OUT" | awk '{print $2}')
    echo "$r -> $C"; OK+=("$r=$C"); CLUSTERS+=("$C")
  else
    echo "$OUT"; FAILED+=("$r")
  fi
done
MSG="🚀 LIGO v2: submitted ${#OK[@]}/$# jobs (A100 @ ${BID_A100}; idle > $((IDLE_SWITCH / 60)) min -> H100 @ ${BID_H100})"$'\n'"$(printf '%s\n' "${OK[@]}")"
[[ ${#FAILED[@]} -gt 0 ]] && MSG+=$'\n'"❌ failed to submit: ${FAILED[*]}"
$NOTIFY "$MSG"
echo "submitted $(date)"

while [[ ${#CLUSTERS[@]} -gt 0 ]]; do
  sleep 300
  LEFT=()
  for C in "${CLUSTERS[@]}"; do
    AD=$($Q "$C" -af JobStatus EnteredCurrentStatus 2>/dev/null)
    if [[ -z $AD ]]; then
      sleep 5; AD=$($Q "$C" -af JobStatus EnteredCurrentStatus 2>/dev/null)
      [[ -z $AD ]] && continue
    fi
    LEFT+=("$C")
    read -r STATUS SINCE <<<"$AD"
    REQ=$($Q "$C" -af:r Requirements)
    if [[ $STATUS == 2 && $REQ == *H100* ]]; then   # running on H100: next slice starts on A100 again
      NEWREQ=${REQ//$H100/$A100}
      [[ $NEWREQ != "$REQ" ]] && condor_qedit "$C" Requirements "$NEWREQ" >/dev/null \
        && condor_qedit "$C" JobPrio $((BID_A100 - 1000)) >/dev/null \
        && echo "$(date) $C running on H100, next slice back to A100"
      continue
    fi
    [[ $STATUS == 1 && $(( $(date +%s) - SINCE )) -gt $IDLE_SWITCH ]] || continue
    [[ $REQ == *A100-SXM4-80GB* ]] || continue
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
