#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/daeyong"
INFER_SCRIPT="${ROOT}/inference_vllm_ablation.py"

MAX_STEPS="${MAX_STEPS:-10}"
MAX_RETRIES="${MAX_RETRIES:-3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/inference_results/dev_kg_correct_1ksample_10_3_diagnosis_guidance_ablation}"

FEEDBACK_MODEL="${FEEDBACK_MODEL:-${ROOT}/trained_models/qwen3-8b-no_premature_conclusion}"

DATASETS=(2wiki hotpotqa musique)
GENERATOR_MODELS=(
  "/workspace/hf_transformers/Meta-Llama-3.1-8B-Instruct"
  "/workspace/hf_transformers/gemma-3-12b-it"
  "/workspace/hf_transformers/Qwen3-8B"
)
GENERATION_FEEDBACK_MODES=(diagnosis_only guidance_only)

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "${ROOT}/conda_envs/vllm/bin/python" ]]; then
  PYTHON_BIN="${ROOT}/conda_envs/vllm/bin/python"
else
  PYTHON_BIN="python3"
fi

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

require_path() {
  local target="$1"
  local label="$2"
  if [[ ! -e "${target}" ]]; then
    echo "[ERROR] Missing ${label}: ${target}" >&2
    exit 1
  fi
}

resolve_feedback_model_path() {
  local model_arg="$1"
  if [[ "${model_arg}" == /* ]]; then
    printf '%s\n' "${model_arg}"
  else
    printf '%s\n' "${ROOT}/trained_models/${model_arg}"
  fi
}

FEEDBACK_MODEL_PATH="$(resolve_feedback_model_path "${FEEDBACK_MODEL}")"
EXTRA_ARGS=("$@")

require_path "${INFER_SCRIPT}" "inference script"
require_path "${FEEDBACK_MODEL_PATH}" "feedback model (adapter path)"

for model in "${GENERATOR_MODELS[@]}"; do
  require_path "${model}" "generator model"
done

for d in "${DATASETS[@]}"; do
  require_path "${ROOT}/benchmarks/${d}_dev_kg_correct.csv" "benchmark CSV (${d})"
done

mkdir -p "${OUTPUT_ROOT}"

log "Python: ${PYTHON_BIN}"
log "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"
log "Inference script: ${INFER_SCRIPT}"
log "Feedback model arg: ${FEEDBACK_MODEL}"
log "Feedback model path: ${FEEDBACK_MODEL_PATH}"
log "Output root: ${OUTPUT_ROOT}"
log "Fixed params: max_steps=${MAX_STEPS}, max_retries=${MAX_RETRIES}"
log "Generator models: ${GENERATOR_MODELS[*]}"
log "Modes: ${GENERATION_FEEDBACK_MODES[*]}"
log "Datasets: ${DATASETS[*]}"

for model in "${GENERATOR_MODELS[@]}"; do
  for mode in "${GENERATION_FEEDBACK_MODES[@]}"; do
    log "===== model=${model} | mode=${mode} ====="

    for dataset in "${DATASETS[@]}"; do
      log "Running dataset=${dataset} model=${model} mode=${mode}"
      cmd=(
        env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
        "${PYTHON_BIN}"
        "${INFER_SCRIPT}"
        --dataset "${dataset}"
        --generator_model "${model}"
        --feedback_model "${FEEDBACK_MODEL}"
        --generation_feedback_mode "${mode}"
        --max_steps "${MAX_STEPS}"
        --max_retries "${MAX_RETRIES}"
        --ablation_output_root "${OUTPUT_ROOT}"
      )

      if (( ${#EXTRA_ARGS[@]} > 0 )); then
        cmd+=("${EXTRA_ARGS[@]}")
      fi

      "${cmd[@]}"
    done
  done
done

log "Done. Check results under: ${OUTPUT_ROOT}"
