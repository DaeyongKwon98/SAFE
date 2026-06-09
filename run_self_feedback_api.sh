#!/usr/bin/env bash
set -euo pipefail

model_id="gpt-5.4-mini"

PYTHON_BIN="${PYTHON_BIN:-/workspace/daeyong/conda_envs/vllm/bin/python}"
export PYTHONPATH="/workspace/daeyong:${PYTHONPATH:-}"

: "${OPENAI_API_KEY:?OPENAI_API_KEY environment variable is required}"

SAMPLE_SIZE="${SAMPLE_SIZE:-200}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_STEPS="${MAX_STEPS:-10}"
MAX_RETRIES="${MAX_RETRIES:-3}"
LOG_EVERY="${LOG_EVERY:-10}"
OPENAI_CONCURRENCY="${OPENAI_CONCURRENCY:-4}"

for dataset in "hotpotqa" "musique"
do
  echo "Starting GPT self-feedback inference for model: ${model_id} / dataset: ${dataset}"

  "$PYTHON_BIN" inference_self_feedback_gpt_api.py \
    --dataset "$dataset" \
    --model_id "$model_id" \
    --sample_size "$SAMPLE_SIZE" \
    --sample_seed "$SAMPLE_SEED" \
    --batch_size "$BATCH_SIZE" \
    --max_steps "$MAX_STEPS" \
    --max_retries "$MAX_RETRIES" \
    --log_every "$LOG_EVERY" \
    --openai_concurrency "$OPENAI_CONCURRENCY"
done
