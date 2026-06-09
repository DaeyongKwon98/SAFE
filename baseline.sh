#!/usr/bin/env bash
set -u -o pipefail

PYTHON_BIN="${PYTHON_BIN:-/workspace/daeyong/conda_envs/vllm_new/bin/python}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
MODEL_ID="/workspace/hf_transformers/Qwen3.6-27B"
MODEL_NAME="Qwen3.6-27B"
SAMPLE_SIZE="${SAMPLE_SIZE:-200}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-100}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
RUN_POSTPROCESS="${RUN_POSTPROCESS:-0}"
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

echo "Starting baseline pipeline: baseline_reasoning inference"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"
echo "MODEL_ID=${MODEL_ID}"
echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
echo "SAMPLE_LIMIT=${SAMPLE_LIMIT}"
echo "SAMPLE_SEED=${SAMPLE_SEED}"
echo "RUN_POSTPROCESS=${RUN_POSTPROCESS}"

for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    run_step "Stage1 baseline_reasoning model=$(basename "${model}") dataset=${dataset}" \
      env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON_BIN}" /workspace/daeyong/inference_baseline_vllm.py \
      --dataset "${dataset}" \
      --model_id "${model}" \
      --sample_size "${SAMPLE_SIZE}" \
      --sample_limit "${SAMPLE_LIMIT}" \
      --sample_seed "${SAMPLE_SEED}"
  done
done

if [[ "${RUN_POSTPROCESS}" == "1" ]]; then
  run_step "Stage2 baseline_final_answer" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON_BIN}" /workspace/daeyong/inference_baseline_final_answer.py \
    --models "${MODEL_NAME}"

  run_step "Stage3 baseline_oss_binary" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON_BIN}" /workspace/daeyong/oss_answer_binary_baseline.py \
    --models "${MODEL_NAME}"
fi

echo
echo "========== Baseline Pipeline Summary =========="
if [ "${FAIL_COUNT}" -eq 0 ]; then
  echo "All stages completed successfully."
  exit 0
fi

echo "Failures: ${FAIL_COUNT}"
for item in "${FAILED_STEPS[@]}"; do
  echo "- ${item}"
done
exit 1
