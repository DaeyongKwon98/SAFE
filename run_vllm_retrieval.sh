#!/usr/bin/env bash
set -euo pipefail

feedback_model="${FEEDBACK_MODEL:-qwen3-8b-no_premature_conclusion}"
generator_model_name="${GENERATOR_MODEL_NAME:-gemma12b}"
generator_model_path="${GENERATOR_MODEL_PATH:-/workspace/hf_transformers/gemma-3-12b-it}"
generator_quantization="${GENERATOR_QUANTIZATION:-none}"
generator_tensor_parallel_size="${GENERATOR_TENSOR_PARALLEL_SIZE:-4}"
generator_max_model_len="${GENERATOR_MAX_MODEL_LEN:-10000}"
evaluator_quantization="${EVALUATOR_QUANTIZATION:-none}"
evaluator_max_tokens="${EVALUATOR_MAX_TOKENS:-256}"
evaluator_max_model_len="${EVALUATOR_MAX_MODEL_LEN:-8000}"

PYTHON_BIN="${PYTHON_BIN:-/workspace/daeyong/conda_envs/vllm/bin/python}"
PYTHON_ENV_DIR="$(dirname "$(dirname "$PYTHON_BIN")")"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

MAX_STEPS="${MAX_STEPS:-10}"
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-3}"
RETRIEVAL_MAX_PER_SAMPLE="${RETRIEVAL_MAX_PER_SAMPLE:-1}"
RETRIEVAL_CORPUS_ROOT="${RETRIEVAL_CORPUS_ROOT:-/workspace/daeyong/benchmarks}"
TRACK_CACHE_STATS="${TRACK_CACHE_STATS:-1}"
CACHE_STATS_MODE="${CACHE_STATS_MODE:-exact_or_fallback}"
DRY_RUN="${DRY_RUN:-0}"

export DAEYONG_VLLM_TORCH_PRELOAD="${DAEYONG_VLLM_TORCH_PRELOAD:-1}"
export PYTHONPATH="/workspace/daeyong:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$PYTHON_ENV_DIR/lib:${LD_LIBRARY_PATH:-}"

read -r -a DATASET_LIST <<< "${DATASETS:-2wiki hotpotqa musique}"

cache_args=()
if [[ "$TRACK_CACHE_STATS" == "1" ]]; then
  cache_args+=(--track_cache_stats --cache_stats_mode "$CACHE_STATS_MODE")
fi

echo "Starting vLLM inference with Missing Evidence BM25 retrieval"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"
echo "GENERATOR_MODEL=${generator_model_path}"
echo "FEEDBACK_MODEL=${feedback_model}"
echo "EVALUATOR_QUANTIZATION=${evaluator_quantization}"
echo "EVALUATOR_MAX_TOKENS=${evaluator_max_tokens}"
echo "GENERATOR_MAX_MODEL_LEN=${generator_max_model_len}"
echo "EVALUATOR_MAX_MODEL_LEN=${evaluator_max_model_len}"
echo "RETRIEVAL_TOP_K=${RETRIEVAL_TOP_K}"
echo "RETRIEVAL_MAX_PER_SAMPLE=${RETRIEVAL_MAX_PER_SAMPLE}"
echo "DATASETS=${DATASET_LIST[*]}"
echo "DRY_RUN=${DRY_RUN}"

for dataset in "${DATASET_LIST[@]}"; do
  corpus_path="${RETRIEVAL_CORPUS_ROOT}/${dataset}_corpus.json"
  if [[ ! -f "$corpus_path" ]]; then
    echo "Missing corpus file: ${corpus_path}" >&2
    exit 1
  fi

  echo
  echo "Starting retrieval inference: model=${generator_model_path} dataset=${dataset}"

  cmd=(
    "$PYTHON_BIN" /workspace/daeyong/inference_vllm_with_retrieval.py
    --dataset "$dataset" \
    --generator_model "$generator_model_path" \
    --generator_quantization "$generator_quantization" \
    --generator_tensor_parallel_size "$generator_tensor_parallel_size" \
    --generator_max_model_len "$generator_max_model_len" \
    --feedback_model "$feedback_model" \
    --max_steps "$MAX_STEPS" \
    --max_retries "$MAX_RETRIES" \
    --evaluator_quantization "$evaluator_quantization" \
    --evaluator_max_tokens "$evaluator_max_tokens" \
    --evaluator_max_model_len "$evaluator_max_model_len" \
    --retrieval_top_k "$RETRIEVAL_TOP_K" \
    --retrieval_max_per_sample "$RETRIEVAL_MAX_PER_SAMPLE" \
    --retrieval_corpus_path "$corpus_path" \
    "${cache_args[@]}"
  )

  echo "CMD: CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} ${cmd[*]}"
  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "${cmd[@]}"
done

feedback_model_clean="$(basename "$feedback_model")"
feedback_model_clean="${feedback_model_clean//-/_}"
output_folder="/workspace/daeyong/inference_results/dev_kg_correct_1ksample_no_premature_conclusion_${MAX_STEPS}_${MAX_RETRIES}_${feedback_model_clean}_bm25_retrieval"

echo
echo "Inference complete."
echo "Output folder: ${output_folder}"
echo "Generator short name: ${generator_model_name}"
