#!/usr/bin/env bash
set -euo pipefail

# Example: FastTD3 + SimbaV2 critic LoRA on MuJoCo Playground.
# Usage: GPU=0 ENV_NAME=G1JoystickFlatTerrain SEED=0 bash scripts/run_lora_example.sh

GPU="${GPU:-0}"
ENV_NAME="${ENV_NAME:-G1JoystickFlatTerrain}"
SEED="${SEED:-0}"
PROJECT_NAME="${PROJECT_NAME:-FastTD3_SimbaV2_LoRA}"
WANDB_MODE="${WANDB_MODE:-offline}"

CRITIC_BLOCKS="${CRITIC_BLOCKS:-2}"
CRITIC_DIM="${CRITIC_DIM:-512}"
ACTOR_BLOCKS="${ACTOR_BLOCKS:-1}"
ACTOR_DIM="${ACTOR_DIM:-256}"
CRITIC_LR="${CRITIC_LR:-3e-4}"
CRITIC_LR_END="${CRITIC_LR_END:-3e-5}"
ACTOR_LR="${ACTOR_LR:-3e-4}"
ACTOR_LR_END="${ACTOR_LR_END:-3e-5}"
LORA_RANK="${LORA_RANK:-96}"
LORA_ALPHA="${LORA_ALPHA:-${LORA_RANK}}"
LORA_A_INIT="${LORA_A_INIT:-normal}"
LORA_B_INIT="${LORA_B_INIT:-normal}"
LORA_LR="${LORA_LR:-${CRITIC_LR}}"
LORA_LR_END="${LORA_LR_END:-${CRITIC_LR_END}}"
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

cd "$(dirname "$0")/../fast_td3"

python train.py \
    --env_name "${ENV_NAME}" \
    --seed "${SEED}" \
    --project "${PROJECT_NAME}" \
    --agent fasttd3_simbav2 \
    --critic_num_blocks "${CRITIC_BLOCKS}" \
    --critic_hidden_dim "${CRITIC_DIM}" \
    --actor_num_blocks "${ACTOR_BLOCKS}" \
    --actor_hidden_dim "${ACTOR_DIM}" \
    --batch_size 8192 \
    --critic_learning_rate "${CRITIC_LR}" \
    --critic_learning_rate_end "${CRITIC_LR_END}" \
    --actor_learning_rate "${ACTOR_LR}" \
    --actor_learning_rate_end "${ACTOR_LR_END}" \
    --weight_decay 0.0 \
    --use_wandb \
    --compile \
    --amp \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_a_init "${LORA_A_INIT}" \
    --lora_b_init "${LORA_B_INIT}" \
    --lora_learning_rate "${LORA_LR}" \
    --lora_learning_rate_end "${LORA_LR_END}" \
    --lora_weight_decay "${LORA_WD}" \
    --critic_lora_enable \
    --no_actor_lora_enable
