#!/bin/bash
# Wait for the windowed 2048 Hz split files, then submit one condor GPU job per model.
D=/fast/barmstrong/LIGO_2048Hz
until [[ -f $D/train.h5 && -f $D/val.h5 && -f $D/test.h5 ]]; do
  if ! condor_q 17698489 -af JobStatus 2>/dev/null | grep -q .; then
    [[ -f $D/train.h5 && -f $D/val.h5 && -f $D/test.h5 ]] || { echo "extraction jobs gone but files missing"; exit 1; }
  fi
  sleep 60
done
cd /lustre/home/barmstrong/phyts/cluster
for m in conv1d s4d linoss; do
  condor_submit_bid 25 \
    -append "arguments = /lustre/home/barmstrong/phyts/cluster/train_ligo.sh $m" \
    -append "requirements = (TARGET.CUDAGlobalMemoryMb > 70000)" \
    -append "output = /fast/barmstrong/outputs/ligo2048_$m.\$(Cluster).out" \
    -append "error = /fast/barmstrong/outputs/ligo2048_$m.\$(Cluster).err" \
    /home/barmstrong/TimeSeriesPhysics/cluster/standard.sub
done
