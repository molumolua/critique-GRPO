#!/usr/bin/env bash
set -euo pipefail
export WANDB_MODE=offline
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CRITIQUE_ROOT="$(cd -- "${VERL_ROOT}/.." && pwd)"
DENOISE_ROOT=${DENOISE_ROOT:-"${CRITIQUE_ROOT}/../DenoiseRL"}

# Offline by default: every artifact is resolved from an explicit local path.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export WANDB_MODE=${WANDB_MODE:-offline}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export PYTHONPATH="${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# The model-specific wrappers set these three values.
: "${MODEL_NAME:?MODEL_NAME must be set by a 4B or 8B recipe}"
: "${PROJECT_NAME:?PROJECT_NAME must be set by a 4B or 8B recipe}"
: "${EXPERIMENT_NAME:?EXPERIMENT_NAME must be set by a 4B or 8B recipe}"

MODEL_PATH=${MODEL_PATH:-"${DENOISE_ROOT}/../Model/Qwen/${MODEL_NAME}"}
TRAIN_FILE=${TRAIN_FILE:-"${DENOISE_ROOT}/data/MATH7500.with_wrong_boxed.qwen2.5-1.5b.parquet"}

VAL_AIME25=${VAL_AIME25:-"${DENOISE_ROOT}/data/aime25_test.parquet"}
VAL_BBEH=${VAL_BBEH:-"${DENOISE_ROOT}/data/bbeh_data.parquet"}
VAL_MATH500=${VAL_MATH500:-"${DENOISE_ROOT}/data/MATH500-test.parquet"}
VAL_AMC23=${VAL_AMC23:-"${DENOISE_ROOT}/data/amc23_test.parquet"}
VAL_AIME24=${VAL_AIME24:-"${DENOISE_ROOT}/data/aime24_test.parquet"}
VAL_MMLU_PRO=${VAL_MMLU_PRO:-"${DENOISE_ROOT}/data/MMLU-Pro-Valid.parquet"}
TEST_FILES=${TEST_FILES:-"[\"${VAL_AIME25}\",\"${VAL_BBEH}\",\"${VAL_MATH500}\",\"${VAL_AMC23}\",\"${VAL_AIME24}\",\"${VAL_MMLU_PRO}\"]"}

RAY_DATA_HOME=${RAY_DATA_HOME:-"${DENOISE_ROOT}"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}"}

# Shared DenoiseRL-v2 optimization and sampling configuration.
NUM_GPUS=${NUM_GPUS:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
ROLLOUT_N=${ROLLOUT_N:-16}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-0}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-10000}
SAVE_FREQ=${SAVE_FREQ:-40}
TEST_FREQ=${TEST_FREQ:-40}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.6}
VAL_TOP_P=${VAL_TOP_P:-0.95}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.7}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
SP_SIZE=${SP_SIZE:-1}
OFFLOAD=${OFFLOAD:-True}
REF_OFFLOAD=${REF_OFFLOAD:-True}

ACTOR_MAX_TOKEN_LEN=$((2 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)))
INFER_MAX_TOKEN_LEN=$((2 * (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)))
MAX_NUM_BATCHED_TOKENS=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

# Critique-GRPO-specific settings: one of the 16 trajectories is replaced by
# a locally generated, ground-truth-guided refinement. No external critic API
# is used. The off-policy objective is retained, while token-mean aggregation
# and PPO clipping are restored to match DenoiseRL-v2.
CRITIQUE_TYPE=${CRITIQUE_TYPE:-simple_gt}
N_PREFIX=${N_PREFIX:-1}
OFF_POLICY_RESHAPE=${OFF_POLICY_RESHAPE:-p_div_p_0.1}

COMMAND=(
    python3 -m verl.critique_grpo.critique_main_ppo
    "data.train_files=${TRAIN_FILE}"
    "data.val_files=${TEST_FILES}"
    data.prompt_key=prompt
    data.shuffle=False
    data.truncation=left
    data.filter_overlong_prompts=False
    "+data.filter_usable_wrong_solutions=True"
    data.return_raw_chat=True
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "+data.max_target_length=${MAX_RESPONSE_LENGTH}"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    data.val_batch_size=512
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
    algorithm.norm_adv_by_std_in_grpo=True
    algorithm.grpo_use_std=True
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    "+actor_rollout_ref.model.override_config.attention_dropout=0.0"
    "+actor_rollout_ref.model.override_config.embd_pdrop=0.0"
    "+actor_rollout_ref.model.override_config.resid_pdrop=0.0"
    actor_rollout_ref.actor.use_dynamic_bsz=True
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_MAX_TOKEN_LEN}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.optim.lr=${LEARNING_RATE}"
    "actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS}"
    actor_rollout_ref.actor.optim.weight_decay=0
    actor_rollout_ref.actor.grad_clip=1.0
    actor_rollout_ref.actor.clip_ratio=0.2
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.2
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.entropy_coeff=0.0
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.use_off_policy_loss=True
    actor_rollout_ref.actor.off_policy_normalize=False
    "actor_rollout_ref.actor.off_policy_reshape=${OFF_POLICY_RESHAPE}"
    actor_rollout_ref.actor.off_policy_loss_impl=token
    actor_rollout_ref.actor.loss_remove_token_mean=False
    actor_rollout_ref.actor.loss_remove_clip=False
    "actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SP_SIZE}"
    "actor_rollout_ref.actor.fsdp_config.param_offload=${OFFLOAD}"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OFFLOAD}"
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${INFER_MAX_TOKEN_LEN}"
    "actor_rollout_ref.ref.ulysses_sequence_parallel_size=${SP_SIZE}"
    "actor_rollout_ref.ref.fsdp_config.param_offload=${REF_OFFLOAD}"
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${INFER_MAX_TOKEN_LEN}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    actor_rollout_ref.rollout.n_val=1
    "actor_rollout_ref.rollout.n_prefix=${N_PREFIX}"
    "actor_rollout_ref.rollout.max_prefix_len=${MAX_RESPONSE_LENGTH}"
    "actor_rollout_ref.rollout.critique_type=${CRITIQUE_TYPE}"
    "actor_rollout_ref.rollout.temperature=${TEMPERATURE}"
    "actor_rollout_ref.rollout.top_p=${TOP_P}"
    "actor_rollout_ref.rollout.top_k=${TOP_K}"
    "actor_rollout_ref.rollout.val_temperature=${VAL_TEMPERATURE}"
    "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}"
    "actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}"
    "actor_rollout_ref.rollout.val_kwargs.top_k=${TOP_K}"
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE}"
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
    reward_model.enable=False
    reward_model.reward_manager=naive
    "trainer.logger=['console','wandb']"
    "trainer.project_name=${PROJECT_NAME}"
    "trainer.experiment_name=${EXPERIMENT_NAME}"
    "trainer.n_gpus_per_node=${NUM_GPUS}"
    trainer.nnodes=1
    trainer.val_before_train=False
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "trainer.default_local_dir=${CKPTS_DIR}"
    trainer.resume_mode=auto
    trainer.max_actor_ckpt_to_keep=1
)

if [[ ${RECIPE_DRY_RUN:-0} == 1 ]]; then
    printf 'Resolved command:\n'
    printf ' %q' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
    printf 'Local model directory not found: %s\n' "${MODEL_PATH}" >&2
    printf 'Set MODEL_PATH to a local Qwen checkpoint; downloading is disabled.\n' >&2
    exit 1
fi

for data_file in \
    "${TRAIN_FILE}" \
    "${VAL_AIME25}" \
    "${VAL_BBEH}" \
    "${VAL_MATH500}" \
    "${VAL_AMC23}" \
    "${VAL_AIME24}" \
    "${VAL_MMLU_PRO}"; do
    if [[ ! -f "${data_file}" ]]; then
        printf 'Local dataset file not found: %s\n' "${data_file}" >&2
        exit 1
    fi
done

cd "${VERL_ROOT}"
exec "${COMMAND[@]}"
