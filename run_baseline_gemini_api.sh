#!/usr/bin/env bash
set -euo pipefail

model_id="gemini-3.1-flash-lite-preview"

PYTHON_BIN="${PYTHON_BIN:-/workspace/daeyong/conda_envs/vllm/bin/python}"
export PYTHONPATH="/workspace/daeyong:${PYTHONPATH:-}"

if [[ -z "${GEMINI_API_KEY:-}" && -n "${GOOGLE_API_KEY:-}" ]]; then
  export GEMINI_API_KEY="${GOOGLE_API_KEY}"
fi
: "${GEMINI_API_KEY:?GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required}"

SAMPLE_SIZE="${SAMPLE_SIZE:-200}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_STEPS="${MAX_STEPS:-10}"
GEMINI_CONCURRENCY="${GEMINI_CONCURRENCY:-8}"

for dataset in "2wiki" "hotpotqa" "musique"
do
  echo "Starting Gemini baseline inference for model: ${model_id} / dataset: ${dataset}"

  "$PYTHON_BIN" inference_baseline_gemini_api.py \
    --dataset "$dataset" \
    --model_id "$model_id" \
    --sample_size "$SAMPLE_SIZE" \
    --sample_seed "$SAMPLE_SEED" \
    --batch_size "$BATCH_SIZE" \
    --max_steps "$MAX_STEPS" \
    --gemini_concurrency "$GEMINI_CONCURRENCY"
done
