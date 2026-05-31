#!/usr/bin/env bash
set -euo pipefail

# Reference full-training example.
# The current training stack requires DeepSpeed. This script keeps the default
# finetuning target and is closer to the configuration used for real runs, but
# may require substantial GPU memory and may be better suited to multi-GPU jobs.
#
# Optional environment variables:
#   TRAIN_DATA_GLOB
#   MODEL_DIR
#   OUTPUT_DIR
#   DEEPSPEED_CONFIG
#   NUM_PROCESSES
#   MODEL_DTYPE
#   MAX_LENGTH
#   NUM_EPOCHS
#   PER_DEVICE_BATCH_SIZE
#   GRADIENT_ACCUMULATION_STEPS
#   NUM_WARMUP_STEPS
#   LOGGING_STEPS
#   SAVE_STEPS
#   USE_ORACLE
#   LOG_FILE
#   RESUME_FROM_CHECKPOINT  # path to a step_<N> dir to resume from (default: none).
#                           # On resume, every arg except OUTPUT_DIR/MAX_TRAIN_STEPS
#                           # must match the original run's config.json (incl. LOG_FILE).

TRAIN_DATA_GLOB="${TRAIN_DATA_GLOB:-processed_data/spokenwoz_sample/train_text_oracle_a0b1_events-*.parquet}"
MODEL_DIR="${MODEL_DIR:-init_models/moshiko-one_streams-bfloat16}"
OUTPUT_DIR="${OUTPUT_DIR:-output/moshiko-finetuned}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-ds_configs/zero3-bfp16-warmlr-act_ckpt.json}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
MAX_LENGTH="${MAX_LENGTH:-512}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-100}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-500}"
USE_ORACLE="${USE_ORACLE:-1}"
LOG_FILE="${LOG_FILE:-logs/training_$(date +%Y%m%d_%H%M%S).log}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

MOSHI_SPEAKERS="${MOSHI_SPEAKERS:-B}"

mkdir -p "$(dirname "$LOG_FILE")"
echo "Logging to $LOG_FILE"

# Single-process runs: skip the GCP gIB net plugin entirely. The VM's gIB shim
# enforces specific NCCL_* env vars and silently aborts init when they don't
# match. Stripping /usr/local/gib/lib64 from LD_LIBRARY_PATH makes NCCL not see
# the plugin at all, so we can use the bundled NCCL with TCP sockets.
if [ "${NUM_PROCESSES}" = "1" ]; then
    export LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '/usr/local/gib' | paste -sd: -)"
    unset NCCL_NET NCCL_TUNER_CONFIG_PATH
    export NCCL_NET=Socket
fi

# System CUDA (12.9) differs from torch's build CUDA (12.1). DeepSpeed refuses
# to JIT-compile the CPU-Adam extension across that mismatch by default; the
# minor-version delta is benign, so skip the strict check.
export DS_SKIP_CUDA_CHECK=1

EXTRA_ARGS=()
if [ "${USE_ORACLE}" = "1" ]; then
    EXTRA_ARGS+=(--use_oracle)
fi
EXTRA_ARGS+=(--moshi_speakers "${MOSHI_SPEAKERS}")
EXTRA_ARGS+=(--log_file "${LOG_FILE}")
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    EXTRA_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

uv run accelerate launch \
    --num_processes "${NUM_PROCESSES}" \
    --num_machines 1 \
    --use_deepspeed \
    --deepspeed_config_file "${DEEPSPEED_CONFIG}" \
    finetune.py \
        --launcher accelerate \
        --output_dir "${OUTPUT_DIR}" \
        --train_data_files "${TRAIN_DATA_GLOB}" \
        --model_dir "${MODEL_DIR}" \
        --model_dtype "${MODEL_DTYPE}" \
        --max_length "${MAX_LENGTH}" \
        --min_length 128 \
        --num_train_epochs "${NUM_EPOCHS}" \
        --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --num_warmup_steps "${NUM_WARMUP_STEPS}" \
        --logging_steps "${LOGGING_STEPS}" \
        --save_steps "${SAVE_STEPS}" \
        "${EXTRA_ARGS[@]}"
