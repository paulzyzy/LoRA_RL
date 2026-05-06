#!/usr/bin/env bash
set -euo pipefail

# Example: SAC + SimbaV2 critic LoRA on a DMC task.
# Usage: GPU=0 ENV_NAME=humanoid-run SEED=0 bash scripts/run_lora_example.sh

GPU="${GPU:-0}"
ENV_NAME="${ENV_NAME:-humanoid-run}"
SEED="${SEED:-0}"
PROJECT_NAME="${PROJECT_NAME:-SAC_SimbaV2_LoRA}"
WANDB_MODE="${WANDB_MODE:-offline}"

CRITIC_BLOCKS="${CRITIC_BLOCKS:-2}"
CRITIC_DIM="${CRITIC_DIM:-512}"
ACTOR_BLOCKS="${ACTOR_BLOCKS:-1}"
ACTOR_DIM="${ACTOR_DIM:-128}"
LORA_RANK="${LORA_RANK:-96}"
LORA_ALPHA="${LORA_ALPHA:-${LORA_RANK}}"
LORA_A_INIT="${LORA_A_INIT:-normal}"
LORA_B_INIT="${LORA_B_INIT:-normal}"
LORA_WD="${LORA_WD:-2e-4}"

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

python run_online.py \
    --config_path ./configs \
    --config_name online_rl \
    --overrides project_name="${PROJECT_NAME}" \
    --overrides seed="${SEED}" \
    --overrides env=dmc \
    --overrides env.env_name="${ENV_NAME}" \
    --overrides agent=simbaV2 \
    --overrides agent.actor_num_blocks="${ACTOR_BLOCKS}" \
    --overrides agent.actor_hidden_dim="${ACTOR_DIM}" \
    --overrides agent.critic_num_blocks="${CRITIC_BLOCKS}" \
    --overrides agent.critic_hidden_dim="${CRITIC_DIM}" \
    --overrides agent.critic_low_rank.enable=true \
    --overrides agent.critic_low_rank.rank="${LORA_RANK}" \
    --overrides agent.critic_low_rank.A_init="${LORA_A_INIT}" \
    --overrides agent.critic_low_rank.B_init="${LORA_B_INIT}" \
    --overrides agent.critic_weight_L2norm=true \
    --overrides agent.critic_LoRA_weight_decay="${LORA_WD}" \
    --overrides agent.lora_alpha="${LORA_ALPHA}" \
    --overrides group_name=dmc_lora \
    --overrides exp_name="${ENV_NAME}_b${CRITIC_BLOCKS}_d${CRITIC_DIM}_r${LORA_RANK}_s${SEED}"
