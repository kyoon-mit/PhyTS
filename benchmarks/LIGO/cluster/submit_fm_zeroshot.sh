#!/bin/bash
# Submit LIGO v2 zero-shot FM jobs (one per model x input) and keep them on the cheapest free GPU.
# Usage: submit_fm_zeroshot.sh [-g h100|a100] [-i "256hz 2048hz"] [-m "moment chronos ..."] [-- extra fm_zeroshot_ligo.py args]
# -g h100 (default): H100 @ bid 35, no GPU switching. -g a100: same scheduling as submit_runs.sh
# (A100-80GB @ 25, idle > 30 min -> H100 @ 35). 4 h slices requeued via exit 85 (resume from the
# finished embedding shards), MaxTime 6 h backstop. Start detached:
#   setsid nohup ./submit_fm_zeroshot.sh > /fast/barmstrong/outputs/submit_fm_zeroshot.log 2>&1 < /dev/null &
NOTIFY=/home/barmstrong/cluster/htcondor/notify.sh
A100='(TARGET.CUDADeviceName == "NVIDIA A100-SXM4-80GB")'
H100='(TARGET.CUDADeviceName == "NVIDIA H100 80GB HBM3" || TARGET.CUDADeviceName == "NVIDIA H100")'
Q=/usr/bin/condor_q_raw
BID_A100=25; BID_H100=35; IDLE_SWITCH=1800; MAX_TIME=21600; REQUEUE_RC=85
MODELS="moment chronos timesfm timemoe moirai granite_ttm"
INPUTS="256hz 2048hz"
GPU=h100
while getopts "g:m:i:" o; do case $o in g) GPU=$OPTARG;; m) MODELS=$OPTARG;; i) INPUTS=$OPTARG;; *) exit 1;; esac; done
if [[ $GPU == h100 ]]; then REQ0=$H100; BID0=$BID_H100; else REQ0=$A100; BID0=$BID_A100; fi
shift $((OPTIND - 1)); [[ ${1:-} == -- ]] && shift
EXTRA="$*"

OK=(); FAILED=(); CLUSTERS=()
for inp in $INPUTS; do for m in $MODELS; do
  if OUT=$(condor_submit_bid $BID0 \
    -append "arguments = /lustre/home/barmstrong/phyts/cluster/fm_zeroshot_ligo.sh $m $inp $EXTRA" \
    -append "requirements = $REQ0" \
    -append "MaxTime = $MAX_TIME" \
    -append "on_exit_remove = (ExitCode =!= $REQUEUE_RC)" \
    -append "output = /fast/barmstrong/outputs/ligo_fm_${m}_${inp}.\$(Cluster).out" \
    -append "error = /fast/barmstrong/outputs/ligo_fm_${m}_${inp}.\$(Cluster).err" \
    /home/barmstrong/TimeSeriesPhysics/cluster/standard.sub 2>&1); then
    C=$(grep -o 'cluster [0-9]*' <<<"$OUT" | awk '{print $2}')
    echo "${m}_${inp} -> $C"; OK+=("${m}_${inp}=$C"); CLUSTERS+=("$C")
  else
    echo "$OUT"; FAILED+=("${m}_${inp}")
  fi
done; done
MSG="🚀 LIGO FM zero-shot: submitted ${#OK[@]} jobs (${GPU} @ ${BID0})"$'\n'"$(printf '%s\n' "${OK[@]}")"
[[ ${#FAILED[@]} -gt 0 ]] && MSG+=$'\n'"❌ failed to submit: ${FAILED[*]}"
$NOTIFY "$MSG"; echo "submitted $(date)"
[[ $GPU == h100 ]] && exit 0   # nothing to switch; jobs requeue themselves

while [[ ${#CLUSTERS[@]} -gt 0 ]]; do
  sleep 300
  LEFT=()
  for C in "${CLUSTERS[@]}"; do
    AD=$($Q "$C" -af JobStatus EnteredCurrentStatus 2>/dev/null)
    [[ -z $AD ]] && { sleep 5; AD=$($Q "$C" -af JobStatus EnteredCurrentStatus 2>/dev/null); [[ -z $AD ]] && continue; }
    LEFT+=("$C")
    read -r STATUS SINCE <<<"$AD"
    REQ=$($Q "$C" -af:r Requirements)
    if [[ $STATUS == 2 && $REQ == *H100* ]]; then
      NEWREQ=${REQ//$H100/$A100}
      [[ $NEWREQ != "$REQ" ]] && condor_qedit "$C" Requirements "$NEWREQ" >/dev/null \
        && condor_qedit "$C" JobPrio $((BID_A100 - 1000)) >/dev/null
      continue
    fi
    [[ $STATUS == 1 && $(( $(date +%s) - SINCE )) -gt $IDLE_SWITCH ]] || continue
    [[ $REQ == *A100-SXM4-80GB* ]] || continue
    NEWREQ=${REQ//$A100/$H100}
    if condor_qedit "$C" Requirements "$NEWREQ" >/dev/null && condor_qedit "$C" JobPrio $((BID_H100 - 1000)) >/dev/null; then
      echo "$(date) moved $C to H100"
    else
      $NOTIFY "⚠️ ${C}: could not move FM job from A100 to H100"
    fi
  done
  CLUSTERS=("${LEFT[@]}")
done
$NOTIFY "✅ LIGO FM zero-shot: all jobs left the queue"
echo "all jobs left the queue $(date)"
