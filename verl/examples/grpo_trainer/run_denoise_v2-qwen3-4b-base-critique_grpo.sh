#!/usr/bin/env bash
set -euo pipefail
export WANDB_MODE=offline
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_NAME=${MODEL_NAME:-Qwen3-4B-Base}
export PROJECT_NAME=${PROJECT_NAME:-Critique-GRPO-DenoiseRL-v2-4B}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-"critique-grpo-denoise-v2-${MODEL_NAME}-bsz16-n16-critique1"}

exec "${SCRIPT_DIR}/run_denoise_v2-qwen3-base-critique_grpo.sh" "$@"
