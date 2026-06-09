#!/usr/bin/env bash
set -euo pipefail

feedback_model="qwen3-8b-no_premature_conclusion"
generator_model="gemini-3.1-flash-lite-preview"

PYTHON_BIN="${PYTHON_BIN:-/workspace/daeyong/conda_envs/vllm/bin/python}"
export PYTHONPATH="/workspace/daeyong:${PYTHONPATH:-}"

if [[ -z "${GEMINI_API_KEY:-}" && -n "${GOOGLE_API_KEY:-}" ]]; then
  export GEMINI_API_KEY="${GOOGLE_API_KEY}"
fi
: "${GEMINI_API_KEY:?GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required}"

for dataset in "2wiki" "hotpotqa" "musique"
do
  echo "Starting Gemini API inference for model: ${generator_model} / dataset: ${dataset}"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}" "$PYTHON_BIN" inference_gemini_api.py \
    --dataset "$dataset" \
    --generator_model "$generator_model" \
    --feedback_model "$feedback_model" \
    --max_steps 10 \
    --max_retries 3 \
    --batch_size 32 \
    --gemini_concurrency 8 \
    --track_cache_stats \
    --cache_stats_mode exact_or_fallback
done
