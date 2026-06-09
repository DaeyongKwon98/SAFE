#!/usr/bin/env bash
set -u -o pipefail

PYTHON_BIN="${PYTHON_BIN:-/workspace/daeyong/conda_envs/vllm_new/bin/python}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
MODEL_ID="/workspace/hf_transformers/Qwen3.6-27B"
MODEL_SHORT="qwen36_27b"
MAX_STEPS="${MAX_STEPS:-10}"
MAX_RETRIES="${MAX_RETRIES:-3}"
SAMPLE_SIZE="${SAMPLE_SIZE:-200}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-100}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
RUN_POSTPROCESS="${RUN_POSTPROCESS:-0}"
OUTPUT_DIR="/workspace/daeyong/inference_results/self_feedback_kg_correct_1k_sample_${MAX_STEPS}_${MAX_RETRIES}"
PYTHON_ENV_DIR="$(dirname "$(dirname "$PYTHON_BIN")")"
export DAEYONG_VLLM_TORCH_PRELOAD="${DAEYONG_VLLM_TORCH_PRELOAD:-1}"
export PYTHONPATH="/workspace/daeyong:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$PYTHON_ENV_DIR/lib:${LD_LIBRARY_PATH:-}"

MODELS=(
  "${MODEL_ID}"
)

DATASETS=("2wiki" "hotpotqa" "musique")

FAIL_COUNT=0
FAILED_STEPS=()

run_step() {
  local step_name="$1"
  shift

  echo
  echo "========== ${step_name} =========="
  echo "CMD: $*"

  if "$@"; then
    echo "[OK] ${step_name}"
    return 0
  fi

  local exit_code=$?
  echo "[FAIL] ${step_name} (exit=${exit_code})"
  FAIL_COUNT=$((FAIL_COUNT + 1))
  FAILED_STEPS+=("${step_name}::exit=${exit_code}")
  return 0
}

echo "Starting pipeline: Self-Feedback inference"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"
echo "MODEL_ID=${MODEL_ID}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "SAMPLE_LIMIT=${SAMPLE_LIMIT}"
echo "SAMPLE_SEED=${SAMPLE_SEED}"
echo "RUN_POSTPROCESS=${RUN_POSTPROCESS}"

for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    run_step "Stage1 self_feedback model=$(basename "${model}") dataset=${dataset}" \
      env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON_BIN}" /workspace/daeyong/inference_self_feedback_vllm.py \
      --dataset "${dataset}" \
      --model_id "${model}" \
      --sample_size "${SAMPLE_SIZE}" \
      --sample_limit "${SAMPLE_LIMIT}" \
      --sample_seed "${SAMPLE_SEED}" \
      --max_steps "${MAX_STEPS}" \
      --max_retries "${MAX_RETRIES}"
  done
done

if [[ "${RUN_POSTPROCESS}" == "1" ]]; then
  run_step "Stage2 final_answer folder=${OUTPUT_DIR}" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON_BIN}" /workspace/daeyong/final_answer_self_feedback.py \
    --folder_path "${OUTPUT_DIR}" \
    --models "${MODEL_SHORT}"

  run_step "Stage3 oss_binary folder=${OUTPUT_DIR}" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON_BIN}" /workspace/daeyong/oss_answer_binary_self_feedback.py \
    --folder_path "${OUTPUT_DIR}" \
    --models "${MODEL_SHORT}"
fi

echo
echo "========== Pipeline Summary =========="
if [ "${FAIL_COUNT}" -eq 0 ]; then
  echo "All stages completed successfully."
  exit 0
fi

echo "Failures: ${FAIL_COUNT}"
for item in "${FAILED_STEPS[@]}"; do
  echo "- ${item}"
done
exit 1
