#!/usr/bin/env bash
# Create wandb sweeps for TESS (cls/reg × selected architectures), capture sweep
# IDs, then submit SLURM wandb agents via run_sweep.sh for each pair.
#
# Requires: repo root .venv, wandb login, network for "wandb sweep"; sbatch for
#           agent submission (typical Engaging login node).
#
# Usage:
#   bash benchmarks/TESS/create_sweeps_and_submit.sh [options]
#
# Outputs (defaults; override TESS_POOL_ROOT to relocate):
#   SLURM logs / WANDB_DIR / checkpoints — see benchmarks/TESS/run_sweep.sh
#   Sweep-id table: $POOL/logs/sweep_batch/create_sweeps_last_ids.tsv
#   After the sweep table, stderr prints one ``run_sweep.sh`` command per sweep
#   (default cls,reg × four models ⇒ eight lines) for copy-paste re-submits.
#   ($POOL defaults to /home/allisone/orcd/pool/UROP_2025_Summer/TimeSeriesPhysics)
#
# Options:
#   --tasks TASKS     Comma-separated: cls, reg, or both (default: cls,reg)
#   --models LIST    Comma-separated subset of:
#                      linoss_damped,s4d,cnn,transformer
#                    (default: all four)
#   --project NAME   wandb project (default: $WANDB_PROJECT or TimeSeriesPhysics)
#   --n-agents N     Passed to run_sweep.sh (default: 4)
#   --downsample-long-lc   Enable long-LC cadence decimation for this batch (overrides default below)
#   --create-only    Only run "wandb sweep"; do not call run_sweep.sh
#   -h, --help       Show this message
#
# Example:
#   bash benchmarks/TESS/create_sweeps_and_submit.sh --n-agents 8

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
POOL="${TESS_POOL_ROOT:-/home/allisone/orcd/pool/UROP_2025_Summer/TimeSeriesPhysics}"
# Tab-separated sweep ids from this batch (override with TESS_SWEEP_SUMMARY)
SUMMARY_FILE="${TESS_SWEEP_SUMMARY:-$POOL/logs/sweep_batch/create_sweeps_last_ids.tsv}"

TASKS="cls,reg"
MODELS="cnn,linoss_damped,s4d,transformer"
PROJECT="${WANDB_PROJECT:-TimeSeriesPhysics}"
N_AGENTS=1
CREATE_ONLY=false

# Long light curves (len > 1500, 10 min cadence → 30 min): set true or use --downsample-long-lc.
DOWNSAMPLE_LONG_LC=false

usage() {
    sed -n '1,25p' "$0" | tail -n +2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks)       TASKS="$2"; shift 2 ;;
        --models)      MODELS="$2"; shift 2 ;;
        --project)     PROJECT="$2"; shift 2 ;;
        --n-agents)    N_AGENTS="$2"; shift 2 ;;
        --downsample-long-lc) DOWNSAMPLE_LONG_LC=true; shift ;;
        --create-only) CREATE_ONLY=true; shift ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if ! command -v wandb >/dev/null 2>&1; then
    if [[ -x "$ROOT/.venv/bin/wandb" ]]; then
        export PATH="$ROOT/.venv/bin:$PATH"
    fi
fi
command -v wandb >/dev/null 2>&1 || {
    echo "ERROR: wandb not on PATH (try: cd $ROOT && source .venv/bin/activate)" >&2
    exit 1
}

IFS=',' read -r -a TASK_ARR <<< "${TASKS// /}"
IFS=',' read -r -a MODEL_ARR <<< "${MODELS// /}"

mkdir -p "$(dirname "$SUMMARY_FILE")"
: >"$SUMMARY_FILE"

for task in "${TASK_ARR[@]}"; do
    task=$(echo "$task" | tr '[:upper:]' '[:lower:]' | xargs)
    if [[ "$task" != "cls" && "$task" != "reg" ]]; then
        echo "ERROR: invalid task '$task' (use cls or reg)" >&2
        exit 1
    fi

    for model in "${MODEL_ARR[@]}"; do
        model=$(echo "$model" | xargs)
        config="$ROOT/configs/TESS/sweep/sweep_${task}_${model}.yaml"
        if [[ ! -f "$config" ]]; then
            echo "ERROR: missing sweep config: $config" >&2
            exit 1
        fi

        echo "=== wandb sweep: $task / $model ===" >&2
        out=$(cd "$ROOT" && WANDB_DIR="${WANDB_DIR:-$POOL}" wandb sweep "$config" --project "$PROJECT" 2>&1) || {
            echo "$out" >&2
            echo "ERROR: wandb sweep failed for $config" >&2
            exit 1
        }
        printf '%s\n' "$out" >&2

        # wandb output format varies by version:
        #   "Create sweep with ID: ..."  or  "wandb: Creating sweep with ID: ..."
        #   "Sweep URL: ..."  or  "wandb: View sweep at: .../sweeps/<id>"
        #   "wandb agent <entity>/<project>/<id>"
        sweep_id=$(printf '%s\n' "$out" | awk '/[Cc]reat(ing|e) sweep with ID:/{print $NF; exit}')
        if [[ -z "${sweep_id:-}" ]]; then
            if [[ "$out" =~ https://wandb\.ai/[^/]+/[^/]+/sweeps/([^$'/? \t\r\n']+) ]]; then
                sweep_id="${BASH_REMATCH[1]}"
            elif [[ "$out" =~ wandb\ agent\ [^/]+/[^/]+/([^$' \t\r\n']+) ]]; then
                sweep_id="${BASH_REMATCH[1]}"
            fi
        fi
        if [[ -z "${sweep_id:-}" ]]; then
            echo "ERROR: could not parse sweep id from wandb output (look for '... sweep with ID:' or .../sweeps/<id>)" >&2
            printf '%s\n' "$out" >&2
            exit 1
        fi
        sweep_id=$(echo "$sweep_id" | xargs)

        echo -e "${task}\t${model}\t${sweep_id}" >>"$SUMMARY_FILE"
        echo "Recorded: $task $model → $sweep_id" >&2

        if [[ "$CREATE_ONLY" != true ]]; then
            echo "=== run_sweep.sh: model_type=$model sweep_id=$sweep_id ===" >&2
            # Pass TESS_DOWNSAMPLE_LONG_LC explicitly so agent jobs match this script (not submit-host env).
            _tess_ds=""
            if [[ "$DOWNSAMPLE_LONG_LC" == true ]]; then
                _tess_ds=1
            fi
            env WANDB_PROJECT="$PROJECT" TESS_DOWNSAMPLE_LONG_LC="$_tess_ds" bash "$SCRIPT_DIR/run_sweep.sh" \
                --model_type "$model" \
                --sweep_id "$sweep_id" \
                --n_agents "$N_AGENTS"
        fi
    done
done

echo "" >&2
echo "Sweep id table (tab-separated): $SUMMARY_FILE" >&2
column -t -s $'\t' "$SUMMARY_FILE" >&2 || cat "$SUMMARY_FILE" >&2

echo "" >&2
echo "Re-submit run_sweep.sh commands (same env as above; copy if SLURM jobs failed):" >&2
_tess_rs=""
if [[ "$DOWNSAMPLE_LONG_LC" == true ]]; then
    _tess_rs=1
fi
while IFS=$'\t' read -r _sw_task _sw_model _sw_id; do
    [[ -z "${_sw_task:-}" ]] && continue
    printf 'env WANDB_PROJECT=%q TESS_DOWNSAMPLE_LONG_LC=%q bash %q --model_type %q --sweep_id %q --n_agents %q\n' \
        "$PROJECT" "$_tess_rs" "$SCRIPT_DIR/run_sweep.sh" "$_sw_model" "$_sw_id" "$N_AGENTS" >&2
done <"$SUMMARY_FILE"
