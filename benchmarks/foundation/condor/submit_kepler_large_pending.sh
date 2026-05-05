#!/usr/bin/env bash
# Submit the 10 missing Kepler Q9v3 "large"-tier cells to Condor.
# Cells (model × regime):
#   chronos-large:     LoRA r=8, LoRA r=16, Full SFT     (3)
#   moment-large:      LoRA r=8, LoRA r=16, Full SFT     (3)
#   timemoe-large:     LoRA r=8, LoRA r=16, Full SFT     (3)
#   granite_ttm-large: Full SFT                          (1)
#
# All run with NiceUser=False, batch_size=2 (large models OOM at bs=4),
# RequestMemory bumped to 24G, and require GPUs_GlobalMemoryMb >= 16000.
# Walltime extended to 12 h.
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB="${JOB_DIR}/kepler_finetune.job"

NICE='NiceUser=False'
BS_OVERRIDE='--batch_size 2'
COMMON_TAIL='--epochs 100 --patience 20'

REQUIREMENTS='(Machine != "libra.esat.kuleuven.be") && (GPUs_Capability >= 7.0) && (GPUs_Capability <= 9.0) && (GPUs_GlobalMemoryMb >= 16000)'

submit_lora() {
  local model="$1"; local r="$2"; local alpha="$3"
  local tag="lora_${model}_large_r${r}_condor"
  local out="plots/kepler_q9v3/lora_${model}_large_r${r}_condor"
  local extra="${BS_OVERRIDE} --adapter lora --lora_r ${r} --lora_alpha ${alpha} ${COMMON_TAIL} --head_lr 1e-3 --backbone_lr 1e-4 --out_dir ${out}"
  echo "==> ${tag}"
  condor_submit \
    -a "${NICE}" \
    -a "tag=${tag}" \
    -a "model=${model}" \
    -a "model_size=large" \
    -a "extra_args=${extra}" \
    -a "RequestMemory=24G" \
    -a "RequestWalltime=43200" \
    -a "Requirements=${REQUIREMENTS}" \
    "${JOB}"
}

submit_full() {
  local model="$1"
  local tag="full_ft_${model}_large_condor"
  local out="plots/kepler_q9v3/full_ft_${model}_large_condor"
  local extra="${BS_OVERRIDE} --adapter full ${COMMON_TAIL} --head_lr 1e-3 --backbone_lr 1e-5 --out_dir ${out}"
  echo "==> ${tag}"
  condor_submit \
    -a "${NICE}" \
    -a "tag=${tag}" \
    -a "model=${model}" \
    -a "model_size=large" \
    -a "extra_args=${extra}" \
    -a "RequestMemory=24G" \
    -a "RequestWalltime=43200" \
    -a "Requirements=${REQUIREMENTS}" \
    "${JOB}"
}

# chronos / moment / timemoe: LoRA r=8, LoRA r=16, Full SFT
for m in chronos moment timemoe; do
  submit_lora "$m" 8 16
  submit_lora "$m" 16 32
  submit_full "$m"
done

# granite_ttm: only Full SFT (LoRA wrapper-skipped)
submit_full "granite_ttm"

echo
echo "Submitted. Track with: condor_q \$USER"
