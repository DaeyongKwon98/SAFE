#!/usr/bin/env bash
set -euo pipefail

feedback_model="qwen3-8b-no_premature_conclusion"
generator_model="gpt-5.4-mini"
evaluator_quantization="${EVALUATOR_QUANTIZATION:-none}"
evaluator_max_tokens="${EVALUATOR_MAX_TOKENS:-256}"

PYTHON_BIN="${PYTHON_BIN:-/workspace/daeyong/conda_envs/vllm/bin/python}"
export PYTHONPATH="/workspace/daeyong:${PYTHONPATH:-}"

: "${OPENAI_API_KEY:?OPENAI_API_KEY environment variable is required}"

debug_print_args=()
if [[ "${DEBUG_PRINT_IO:-1}" == "1" ]]; then
  debug_print_args+=(
    --debug_print_io
    --debug_print_limit "${DEBUG_PRINT_LIMIT:-3}"
    --debug_print_chars "${DEBUG_PRINT_CHARS:-4000}"
  )
fi

for dataset in "2wiki" "hotpotqa" "musique"
do
  echo "Starting GPT API inference for model: ${generator_model} / dataset: ${dataset}"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}" "$PYTHON_BIN" inference_gpt_api.py \
    --dataset "$dataset" \
    --generator_model "$generator_model" \
    --feedback_model "$feedback_model" \
    --max_steps 10 \
    --max_retries 3 \
    --evaluator_quantization "$evaluator_quantization" \
    --evaluator_max_tokens "$evaluator_max_tokens" \
    --batch_size 32 \
    --openai_concurrency 8 \
    --track_cache_stats \
    --cache_stats_mode exact_or_fallback \
    "${debug_print_args[@]}"
done
