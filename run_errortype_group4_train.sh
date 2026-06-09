#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/daeyong"
TRAIN_SCRIPT="${ROOT}/finetune_validationset_fast_ablation.py"

MODEL_ID="${MODEL_ID:-/workspace/hf_transformers/Qwen3-8B}"
DATASET_PATH="${DATASET_PATH:-${ROOT}/fourth_finetuning_data/2wiki_added_ver3.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/trained_models/qwen3-8b-2wiki_added_ver3_errortype_group4_ablation}"

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

if [[ -n "${TORCHRUN_BIN:-}" ]]; then
  :
elif [[ -x "${ROOT}/conda_envs/vllm/bin/torchrun" ]]; then
  TORCHRUN_BIN="${ROOT}/conda_envs/vllm/bin/torchrun"
else
  TORCHRUN_BIN="torchrun"
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

require_path "${TRAIN_SCRIPT}" "training script"
require_path "${DATASET_PATH}" "dataset"
if [[ "${MODEL_ID}" == /* ]]; then
  require_path "${MODEL_ID}" "model path"
fi

mkdir -p "${OUTPUT_ROOT}"

EXTRA_ARGS=("$@")

log "Python/Torch launcher: ${TORCHRUN_BIN}"
log "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"
log "NPROC_PER_NODE=${NPROC_PER_NODE}"
log "MODEL_ID=${MODEL_ID}"
log "DATASET_PATH=${DATASET_PATH}"
log "OUTPUT_ROOT=${OUTPUT_ROOT}"
log "Target A (drop 4): Overthinking, Inefficiency, Off-topic, Redundancy"
log "  -> drop_key=drop_off_topic_inefficiency_redundancy_overthinking"
log "  -> prompt=evaluate_system_prompt_drop_off_topic_inefficiency_redundancy_overthinking"
log "Target B (drop 4): Information Miss, Premature Attribution, Contradictory, Unsupported"
log "  -> drop_key=drop_contradictory_information_miss_unsupported_premature_attribution"
log "  -> prompt=evaluate_system_prompt_drop_contradictory_information_miss_unsupported_premature_attribution"

CMD=(
  env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
  "${TORCHRUN_BIN}"
  --nproc_per_node "${NPROC_PER_NODE}"
  "${TRAIN_SCRIPT}"
  --model_id "${MODEL_ID}"
  --dataset_path "${DATASET_PATH}"
  --output_root "${OUTPUT_ROOT}"
  --ablation_target_mode fixed_group4
  --evaluator_prompt_mode ablation
  --missing_ablation_prompt_policy error
)

if (( ${#EXTRA_ARGS[@]} > 0 )); then
  CMD+=("${EXTRA_ARGS[@]}")
fi

log "Run command: ${CMD[*]}"
"${CMD[@]}"

log "Done. Trained adapters should be under: ${OUTPUT_ROOT}"
