#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/daeyong"
INFER_SCRIPT="${ROOT}/inference_vllm.py"

MAX_STEPS=10
MAX_RETRIES=3

DATASETS=(2wiki hotpotqa musique)
DROP_PROFILE="${DROP_PROFILE:-leave_one_out}"

if [[ "${DROP_PROFILE}" == "leave_one_out" ]]; then
  ABLATION_MODEL_ROOT="${ROOT}/trained_models/qwen3-8b-2wiki_added_ver3_errortype_ablation"
  OUTPUT_ROOT="${ROOT}/inference_results/dev_kg_correct_1ksample_with_noises_10_3_errortype_ablation"
  DROP_MODELS=(
    drop_contradictory
    drop_inefficiency
    drop_information_miss
    drop_logical_fallacy
    drop_off_topic
    drop_overthinking
    drop_premature_attribution
    drop_redundancy
    drop_unsupported
    drop_wrong_conclusion
  )
elif [[ "${DROP_PROFILE}" == "fixed_group4" ]]; then
  ABLATION_MODEL_ROOT="${ROOT}/trained_models/qwen3-8b-2wiki_added_ver3_errortype_group4_ablation"
  OUTPUT_ROOT="${ROOT}/inference_results/dev_kg_correct_1ksample_with_noises_10_3_errortype_group4_ablation"
  DROP_MODELS=(
    drop_contradictory_information_miss_unsupported_premature_attribution
    drop_off_topic_inefficiency_redundancy_overthinking
  )
else
  echo "[ERROR] Unsupported DROP_PROFILE: ${DROP_PROFILE} (expected: leave_one_out | fixed_group4)" >&2
  exit 1
fi

GENERATOR_MODEL="${GENERATOR_MODEL:-/workspace/hf_transformers/Qwen3-8B}"
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

require_path "${INFER_SCRIPT}" "inference script"
require_path "${GENERATOR_MODEL}" "generator model"

for d in "${DATASETS[@]}"; do
  require_path "${ROOT}/benchmarks/${d}_dev_kg_correct.csv" "benchmark CSV (${d})"
done

for drop_key in "${DROP_MODELS[@]}"; do
  adapter_path="${ABLATION_MODEL_ROOT}/${drop_key}"
  require_path "${adapter_path}" "adapter directory (${drop_key})"
  require_path "${adapter_path}/adapter_config.json" "adapter config (${drop_key})"
done

mkdir -p "${OUTPUT_ROOT}"

log "Python: ${PYTHON_BIN}"
log "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"
log "Drop profile: ${DROP_PROFILE}"
log "Generator model: ${GENERATOR_MODEL}"
log "Adapter root: ${ABLATION_MODEL_ROOT}"
log "Output root: ${OUTPUT_ROOT}"
log "Fixed params: max_steps=${MAX_STEPS}, max_retries=${MAX_RETRIES}"

for drop_key in "${DROP_MODELS[@]}"; do
  adapter_path="${ABLATION_MODEL_ROOT}/${drop_key}"
  log "===== Adapter: ${drop_key} ====="

  for dataset in "${DATASETS[@]}"; do
    log "Running dataset=${dataset} adapter=${drop_key}"
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON_BIN}" "${INFER_SCRIPT}" \
      --dataset "${dataset}" \
      --generator_model "${GENERATOR_MODEL}" \
      --feedback_model "${adapter_path}" \
      --max_steps "${MAX_STEPS}" \
      --max_retries "${MAX_RETRIES}" \
      --ablation_output_root "${OUTPUT_ROOT}"
  done

done

log "Done. Check results under: ${OUTPUT_ROOT}"
