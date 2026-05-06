#!/usr/bin/env bash
set -euo pipefail

# Example: SAC-BRC critic LoRA on a DMC humanoid task.
# Usage: GPU=0 ENV_NAMES=humanoid-walk SEED=0 bash scripts/run_lora_example.sh

GPU="${GPU:-0}"
ENV_NAMES="${ENV_NAMES:-humanoid-walk}"
SEED="${SEED:-0}"
PROJECT_NAME="${PROJECT_NAME:-SAC_BRC_LoRA}"
WANDB_MODE="${WANDB_MODE:-offline}"

DEPTH_CRITIC="${DEPTH_CRITIC:-2}"
WIDTH_CRITIC="${WIDTH_CRITIC:-4096}"
LORA_RANK="${LORA_RANK:-128}"
LORA_ALPHA="${LORA_ALPHA:-${LORA_RANK}}"
LORA_A_INIT="${LORA_A_INIT:-normal}"
LORA_B_INIT="${LORA_B_INIT:-normal}"
LORA_WD="${LORA_WD:-6e-4}"

if [[ -n "${CONDA_ENV:-}" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE

cd "$(dirname "$0")/.."

python train.py \
    --seed "${SEED}" \
    --env_names "${ENV_NAMES}" \
    --method lora \
    --depth_critic "${DEPTH_CRITIC}" \
    --width_critic "${WIDTH_CRITIC}" \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_a_init "${LORA_A_INIT}" \
    --lora_b_init "${LORA_B_INIT}" \
    --lora_weight_decay "${LORA_WD}" \
    --wandb_project "${PROJECT_NAME}" \
    --log_to_wandb \
    --offline_evaluation \
    --render
