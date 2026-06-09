#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/daeyong"
LOG_DIR="${ROOT}/run_logs"
RUN_ID="${RUN_ID:-missing_evidence_retrieval_$(date '+%Y%m%d_%H%M%S')}"
MASTER_LOG="${LOG_DIR}/${RUN_ID}.log"

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
TRAIN_PATTERN="${TRAIN_PATTERN:-finetune_validationset_fast_missing_evidence.py}"
GPU_MAX_USED_MIB="${GPU_MAX_USED_MIB:-4096}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-180}"

FEEDBACK_MODEL="${FEEDBACK_MODEL:-${ROOT}/trained_models/qwen3-8b-missing_evidence_training_data}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/conda_envs/vllm/bin/python}"

DATASETS="${DATASETS:-2wiki hotpotqa musique}"
MAX_STEPS="${MAX_STEPS:-10}"
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-3}"
RETRIEVAL_MAX_PER_SAMPLE="${RETRIEVAL_MAX_PER_SAMPLE:-1}"
TRACK_CACHE_STATS="${TRACK_CACHE_STATS:-1}"

mkdir -p "${LOG_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${MASTER_LOG}"
}

require_path() {
  local target="$1"
  local label="$2"
  if [[ ! -e "${target}" ]]; then
    log "ERROR: missing ${label}: ${target}"
    exit 1
  fi
}

training_pids() {
  pgrep -f "${TRAIN_PATTERN}" || true
}

gpu_snapshot() {
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits |
    awk -F',' -v gpus="${CUDA_DEVICES}" '
      BEGIN {
        split(gpus, gpu_list, ",")
        for (i in gpu_list) wanted[gpu_list[i] + 0] = 1
      }
      {
        idx = $1 + 0
        if (idx in wanted) {
          gsub(/^ +| +$/, "", $2)
          gsub(/^ +| +$/, "", $3)
          gsub(/^ +| +$/, "", $4)
          printf "gpu=%s mem=%s/%sMiB util=%s%%\n", idx, $2, $3, $4
        }
      }'
}

gpu_busy_count() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    awk -F',' -v gpus="${CUDA_DEVICES}" -v max_mem="${GPU_MAX_USED_MIB}" '
      BEGIN {
        split(gpus, gpu_list, ",")
        for (i in gpu_list) wanted[gpu_list[i] + 0] = 1
      }
      {
        idx = $1 + 0
        mem = $2 + 0
        if ((idx in wanted) && mem > max_mem) busy += 1
      }
      END { print busy + 0 }'
}

wait_for_training() {
  while true; do
    mapfile -t pids < <(training_pids)
    if (( ${#pids[@]} == 0 )); then
      log "No matching training process remains for pattern: ${TRAIN_PATTERN}"
      return
    fi

    log "Waiting for training PIDs to finish: ${pids[*]}"
    gpu_snapshot | tee -a "${MASTER_LOG}"
    sleep "${WAIT_INTERVAL_SEC}"
  done
}

wait_for_gpus() {
  while true; do
    busy="$(gpu_busy_count)"
    if [[ "${busy}" == "0" ]]; then
      log "GPUs ${CUDA_DEVICES} are below ${GPU_MAX_USED_MIB}MiB used."
      gpu_snapshot | tee -a "${MASTER_LOG}"
      return
    fi

    log "Waiting for GPUs ${CUDA_DEVICES} to free; ${busy} GPU(s) still above ${GPU_MAX_USED_MIB}MiB."
    gpu_snapshot | tee -a "${MASTER_LOG}"
    sleep "${WAIT_INTERVAL_SEC}"
  done
}

run_generator() {
  local model_name="$1"
  local model_path="$2"
  local model_log="${LOG_DIR}/${RUN_ID}_${model_name}.log"

  require_path "${model_path}" "generator model ${model_name}"
  log "Starting generator=${model_name} model_path=${model_path}; detailed log=${model_log}"

  env \
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
    FEEDBACK_MODEL="${FEEDBACK_MODEL}" \
    GENERATOR_MODEL_NAME="${model_name}" \
    GENERATOR_MODEL_PATH="${model_path}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    DATASETS="${DATASETS}" \
    MAX_STEPS="${MAX_STEPS}" \
    MAX_RETRIES="${MAX_RETRIES}" \
    RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K}" \
    RETRIEVAL_MAX_PER_SAMPLE="${RETRIEVAL_MAX_PER_SAMPLE}" \
    TRACK_CACHE_STATS="${TRACK_CACHE_STATS}" \
    bash "${ROOT}/run_vllm_retrieval.sh" >> "${model_log}" 2>&1

  log "Finished generator=${model_name}"
}

log "Queued Missing Evidence BM25 retrieval inference"
log "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"
log "Feedback adapter=${FEEDBACK_MODEL}"
log "Datasets=${DATASETS}"
log "Script keeps inference_vllm_with_retrieval.py sample(n=200)[:100]."

require_path "${ROOT}/inference_vllm_with_retrieval.py" "inference script"
require_path "${ROOT}/run_vllm_retrieval.sh" "retrieval launcher"
require_path "${FEEDBACK_MODEL}" "feedback adapter"
require_path "${PYTHON_BIN}" "python binary"
for dataset in ${DATASETS}; do
  require_path "${ROOT}/benchmarks/${dataset}_dev_kg_correct.csv" "benchmark CSV ${dataset}"
  require_path "${ROOT}/benchmarks/${dataset}_corpus.json" "retrieval corpus ${dataset}"
done

wait_for_training
wait_for_gpus

run_generator "gemma12b" "/workspace/hf_transformers/gemma-3-12b-it"
run_generator "qwen8b" "/workspace/hf_transformers/Qwen3-8B"
run_generator "llama8b" "/workspace/hf_transformers/Meta-Llama-3.1-8B-Instruct"

log "All queued retrieval inference runs completed."
