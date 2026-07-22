#!/usr/bin/env bash

set -Eeuo pipefail

# ModelArts single-node / 8-GPU CALVIN v2 launcher.
# This script intentionally uses PyTorch torchrun through start_mtp_calvin.py;
# it does not use Accelerate or DeepSpeed.

readonly CONDA_BASE="/opt/huawei/explorer-env/dataset/Common_wl/miniconda3"
readonly CONDA_ENV_NAME="mowe"
readonly MOWE_PYTHON="/opt/huawei/explorer-env/dataset/Common_wl/miniconda3/envs/mowe/bin/python"
readonly CUDA_ROOT="/opt/huawei/explorer-env/dataset/trellis_ckpt/cuda/cuda118"

readonly MOWE_ROOT="/home/ma-user/work/algorithm/chaoxintao_2/MoWE/MoWE"
readonly CALVIN_RLDS_ROOT="/home/ma-user/work/dataset/calvin"
readonly OPENVLA_ROOT="/home/ma-user/work/algorithm/chaoxintao_2/MoWE/openvla-7b"
readonly OPENVLA_REVISION="47a0ec7fc4ec123775a391911046cf33cf9ed83f"
readonly DINO_ROOT="/home/ma-user/work/algorithm/chaoxintao_2/MoWE/facebook-dinov2-small"
readonly CALVIN_STORE="/home/ma-user/work/dataset/calvin/mowe_store/calvin_abc_rlds_h16"
readonly MOWE_RUNS="/home/ma-user/work/algorithm/chaoxintao_2/MoWE/outputs"
readonly RUN_ID="${MOWE_CALVIN_RUN_ID:-calvin_abc_original_openvla_h16_v2}"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

require_dir() {
    local label="$1"
    local path="$2"
    [[ -d "$path" ]] || fail "$label directory does not exist: $path"
}

[[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]] \
    || fail "Conda initialization script does not exist under $CONDA_BASE"
[[ -x "$MOWE_PYTHON" ]] || fail "mowe Python is not executable: $MOWE_PYTHON"
require_dir "CUDA" "$CUDA_ROOT"

export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:$CONDA_BASE/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

require_dir "MoWE repository" "$MOWE_ROOT"
require_dir "CALVIN RLDS" "$CALVIN_RLDS_ROOT"
require_dir "OpenVLA snapshot" "$OPENVLA_ROOT"
require_dir "DINO snapshot" "$DINO_ROOT"
[[ -f "$MOWE_ROOT/start_mtp_calvin.py" ]] \
    || fail "CALVIN launcher is missing: $MOWE_ROOT/start_mtp_calvin.py"

shopt -s nullglob
CALVIN_SHARDS=("$CALVIN_RLDS_ROOT"/calvin_abc-train.tfrecord-?????-of-00512)
shopt -u nullglob
[[ "${#CALVIN_SHARDS[@]}" -eq 512 ]] \
    || fail "Expected 512 CALVIN ABC shards, found ${#CALVIN_SHARDS[@]} in $CALVIN_RLDS_ROOT"
unset CALVIN_SHARDS

mkdir -p "$(dirname "$CALVIN_STORE")" "$MOWE_RUNS"

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONPATH="$MOWE_ROOT:$MOWE_ROOT/external/openvla-oft${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM="false"
export TF_CPP_MIN_LOG_LEVEL="2"
export PYTHONUNBUFFERED="1"
export PYTHONHASHSEED="1701"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="1"
export WANDB_MODE="disabled"
export WANDB_DISABLED="true"

cd "$MOWE_ROOT"

echo "Checking mowe environment and the single-node 8-GPU contract..."
"$MOWE_PYTHON" - <<'PY'
import json

import prismatic
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import transformers

payload = {
    "python_environment": "mowe",
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count(),
    "bf16_supported": torch.cuda.is_bf16_supported(),
    "transformers": transformers.__version__,
    "tensorflow": tf.__version__,
    "tensorflow_datasets": tfds.__version__,
    "prismatic_import": True,
}
print(json.dumps(payload, indent=2))
assert torch.cuda.is_available(), "CUDA is not available in the mowe environment."
assert torch.cuda.device_count() == 8, torch.cuda.device_count()
assert torch.cuda.is_bf16_supported(), "The formal CALVIN contract requires BF16 support."
PY

COMMON_ARGS=(
    --repo-root "$MOWE_ROOT"
    --dataset-root "$CALVIN_RLDS_ROOT"
    --feature-store "$CALVIN_STORE"
    --openvla-checkpoint "$OPENVLA_ROOT"
    --openvla-revision "$OPENVLA_REVISION"
    --dino-checkpoint "$DINO_ROOT"
    --run-root-dir "$MOWE_RUNS"
    --run-id "$RUN_ID"
    --python "$MOWE_PYTHON"
    --world-size 8
    --cuda-devices 0,1,2,3,4,5,6,7
)

echo "Running the CALVIN v2 dry-run contract check..."
"$MOWE_PYTHON" start_mtp_calvin.py "${COMMON_ARGS[@]}" --dry-run

if [[ "${MOWE_CALVIN_DRY_RUN_ONLY:-0}" == "1" ]]; then
    echo "MOWE_CALVIN_DRY_RUN_ONLY=1; dry-run passed and formal training was not started."
    exit 0
fi

echo "Dry-run passed. Starting the resumable CALVIN v2 conversion/training chain..."
exec "$MOWE_PYTHON" start_mtp_calvin.py "${COMMON_ARGS[@]}"
