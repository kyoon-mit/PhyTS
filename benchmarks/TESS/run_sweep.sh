#!/bin/bash
# TESS hyperparameter sweep — submit wandb agents as SLURM jobs.
#
# Workflow:
#   1. Create a sweep (run ONCE from a machine with internet + wandb login):
#        wandb sweep configs/TESS/sweep/sweep_<cls|reg>_<model_type>.yaml \
#            --project TimeSeriesPhysics
#      → prints: sweep_id (e.g. abc123def)
#
#   2. Optional: export WANDB_ENTITY=... (else default workspace from wandb.Api())
#
#   3. Submit agents (from the Engaging login node):
#        bash benchmarks/TESS/run_sweep.sh \
#            --model_type mlp \
#            --sweep_id abc123def \
#            [--n_agents 4] \
#            [--jax]          # add for linoss_imex / linoss_damped
#
# Each agent is one SLURM job that runs trials from the sweep until
# wandb stops it (sweep exhausted or time limit reached).
#
# Monitor:
#   squeue -u $USER
#   wandb sweep status <entity>/TimeSeriesPhysics/<sweep_id>

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKDIR=$(cd "$SCRIPT_DIR/../.." && pwd)
POOL="${TESS_POOL_ROOT:-data}"
DATA_DIR=$POOL/TESS
CKPT_DIR=$POOL/checkpoints/sweeps
WANDB_ROOT=$POOL
LOGS=$WORKDIR/logs/logs/sweeps
mkdir -p "$LOGS" "$CKPT_DIR" "$WANDB_ROOT"

# ── Parse arguments ───────────────────────────────────────────────────────────
MODEL_TYPE=""
SWEEP_ID=""
N_AGENTS=4
JAX_MODE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_type)  MODEL_TYPE="$2"; shift 2 ;;
        --sweep_id)    SWEEP_ID="$2";   shift 2 ;;
        --n_agents)    N_AGENTS="$2";   shift 2 ;;
        --jax)         JAX_MODE=true;   shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$MODEL_TYPE" ] || [ -z "$SWEEP_ID" ]; then
    echo "Usage: $0 --model_type <type> --sweep_id <id> [--n_agents N] [--jax]"
    echo "  model_type: mlp | s4d | cnn | cnn_attn | transformer | linoss_imex | linoss_damped"
    echo "  --jax: required for linoss_imex / linoss_damped (set automatically)"
    echo ""
    echo "Sweep config paths (pass to 'wandb sweep' once to create the sweep):"
    echo "  Classification: configs/TESS/sweep/sweep_cls_<model_type>.yaml"
    echo "  Regression:     configs/TESS/sweep/sweep_reg_<model_type>.yaml"
    exit 1
fi

# Auto-enable JAX for LinOSS models
if [[ "$MODEL_TYPE" == linoss_* ]]; then
    JAX_MODE=true
fi

# ── Venv setup ────────────────────────────────────────────────────────────────
if [ ! -d "$WORKDIR/.venv" ]; then
    echo "ERROR: $WORKDIR/.venv not found."
    echo "On the login node: cd $WORKDIR && # activate your venv: source .venv/bin/activate && uv sync [--extra jax --extra cu13]"
    exit 1
fi

LINOSS_JAX_CUDA_EXTRA="${LINOSS_JAX_CUDA_EXTRA:-cu13}"

if command -v uv >/dev/null 2>&1; then
    if [ "$JAX_MODE" = true ]; then
        if [ "$LINOSS_JAX_CUDA_EXTRA" = "none" ] || [ "$LINOSS_JAX_CUDA_EXTRA" = "cpu" ]; then
            (cd "$WORKDIR" && uv sync --extra jax) || { echo "ERROR: uv sync --extra jax failed"; exit 1; }
        else
            (cd "$WORKDIR" && uv sync --extra jax --extra "$LINOSS_JAX_CUDA_EXTRA") || {
                echo "ERROR: uv sync --extra jax --extra $LINOSS_JAX_CUDA_EXTRA failed"; exit 1;
            }
        fi
    else
        (cd "$WORKDIR" && uv sync) || { echo "ERROR: uv sync failed"; exit 1; }
    fi
fi

PYTHON_EXE="$WORKDIR/.venv/bin/python"
if [ ! -x "$PYTHON_EXE" ]; then
    echo "ERROR: $PYTHON_EXE missing. Run uv sync first."; exit 1
fi

ENTITY="${WANDB_ENTITY:-$(WANDB_SILENT=true "$PYTHON_EXE" -c 'import wandb; print(wandb.Api().viewer.entity)' 2>/dev/null)}"
[ -z "$ENTITY" ] && { echo >&2 'Set WANDB_ENTITY or run wandb login'; exit 1; }
SWEEP_PATH="$ENTITY/TimeSeriesPhysics/$SWEEP_ID"

# ── SLURM common ──────────────────────────────────────────────────────────────
SBATCH_COMMON="
  --gpus=1
  --nodes=1
  --ntasks-per-node=1
  --cpus-per-task=8
  --mem=32G
  --time=6:00:00
  --chdir=$WORKDIR
"

# Classification + regression sweeps use the same TESS root (Hub shards mirrored
# into .../.cache/TESS). Override via TESS_DATA_DIR if needed.
BASE_ENV="# activate your venv: source .venv/bin/activate &&
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 PANDAS_USE_PYARROW=1 \
    WANDB_DIR=$WANDB_ROOT TESS_DATA_DIR=$DATA_DIR TESS_CKPT_DIR=$CKPT_DIR"

# Sweep YAMLs invoke benchmarks/TESS/repo_python.sh → repo .venv directly.
# Still prepend .venv/bin to PATH for wandb and any subprocess using bare python.
VENV_BIN_PATH="export PATH=\"$WORKDIR/.venv/bin:\$PATH\""

if [ "$JAX_MODE" = true ]; then
    ACTIVATE="$BASE_ENV && $VENV_BIN_PATH && export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7"
else
    ACTIVATE="$BASE_ENV && $VENV_BIN_PATH"
fi

# Log visible GPU/driver in SLURM .out files (helps verify allocation on Engaging).
NVIDIA_SMI_PROBE="echo '=== nvidia-smi (job start) ===' && nvidia-smi && echo ''"

RUN="srun --cpu-bind=none $PYTHON_EXE"

# ── Submit N agents ───────────────────────────────────────────────────────────
echo "Submitting $N_AGENTS agents for sweep $SWEEP_PATH (model_type=$MODEL_TYPE)"

for i in $(seq 1 "$N_AGENTS"); do
    JOB=$(sbatch --parsable \
      $SBATCH_COMMON \
      --job-name="tess_sweep_${MODEL_TYPE}_${i}" \
      --output="$LOGS/tess_sweep_${MODEL_TYPE}_%j.out" \
      --error="$LOGS/tess_sweep_${MODEL_TYPE}_%j.err" \
      --wrap="$ACTIVATE && $NVIDIA_SMI_PROBE && wandb agent $SWEEP_PATH")
    echo "  Agent $i → job $JOB"
done

echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  wandb sweep https://wandb.ai/${SWEEP_PATH}"
