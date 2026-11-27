#!/usr/bin/env bash
#SBATCH --job-name=sjd_harmonize
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
umask 077

BASE=/data/salazarda
IMAGE="$BASE/containers/vllm-0.11.2.sif"
PYTHON="$BASE/sjd_container_env/bin/python"

submitted_root="${OBJ1_ROOT:-${SLURM_SUBMIT_DIR:-}}"

if [[ -z "$submitted_root" || ! -f "$submitted_root/src/semantic_harmonization/repository_root.sh" ]]; then
    echo "Submit from the obj1_sjd repository root." >&2
    exit 1
fi

source "$submitted_root/src/semantic_harmonization/repository_root.sh"

repo="$(sjd_obj1_root "$submitted_root")"
base_physical="$(cd -- "$BASE" && pwd -P)"
if [[ "$repo" == "$base_physical/"* ]]; then
    repo="$BASE/${repo#"$base_physical/"}"
fi

cd "$repo"

log_dir="$repo/work/semantic_harmonization"

mkdir -p "$log_dir"

exec >"$log_dir/job-${SLURM_JOB_ID}.log" 2>&1

module load apptainer

export APPTAINER_CACHEDIR="$BASE/cache/apptainer"
export APPTAINER_TMPDIR="$BASE/tmp/apptainer"

export APPTAINERENV_HF_HUB_OFFLINE=1
export APPTAINERENV_TRANSFORMERS_OFFLINE=1
export APPTAINERENV_HF_HUB_DISABLE_TELEMETRY=1
export APPTAINERENV_VLLM_NO_USAGE_STATS=1
export APPTAINERENV_DO_NOT_TRACK=1
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_WANDB_MODE=disabled

export APPTAINERENV_VLLM_HOST_IP=127.0.0.1
export APPTAINERENV_VLLM_ENABLE_V1_MULTIPROCESSING=0

export APPTAINERENV_XDG_CACHE_HOME="$BASE/cache"
export APPTAINERENV_FLASHINFER_WORKSPACE_BASE="$BASE/cache/flashinfer"
export APPTAINERENV_TRITON_CACHE_DIR="$BASE/cache/triton"
export APPTAINERENV_VLLM_CACHE_ROOT="$BASE/cache/vllm"
export APPTAINERENV_TMPDIR="$BASE/tmp"

export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export APPTAINERENV_OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export APPTAINERENV_MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p \
    "$BASE/cache/container-home" \
    "$BASE/cache/flashinfer" \
    "$BASE/cache/triton" \
    "$BASE/cache/vllm" \
    "$BASE/tmp"

test -f "$IMAGE"
test -x "$PYTHON"
export APPTAINERENV_SJD_CANONICAL_REPO="$repo"

echo "Starting semantic harmonization"
echo "Job ID: $SLURM_JOB_ID"
echo "Container: $IMAGE"
echo "Repository: $repo"
echo "Started: $(date)"

BIND_ARGS=(--bind "$BASE:$BASE")

if [[ "$repo" != "$BASE/"* ]]; then
    BIND_ARGS+=(--bind "$repo:$repo")
fi

exec apptainer exec \
    --nv \
    --cleanenv \
    --home "$BASE/cache/container-home:/home/salazarda" \
    "${BIND_ARGS[@]}" \
    --pwd "$repo" \
    "$IMAGE" "$PYTHON" -u -m src.semantic_harmonization.run_pipeline
