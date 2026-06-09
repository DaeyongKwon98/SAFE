feedback_model="qwen3-8b-no_premature_conclusion"
# Qwen3.6 27B generator runs in bf16 by default; override with GENERATOR_QUANTIZATION=bnb4 if needed.
generator_model_name="qwen36_27b"
generator_model_path="/workspace/hf_transformers/Qwen3.6-27B"
generator_quantization="${GENERATOR_QUANTIZATION:-none}"
generator_tensor_parallel_size="${GENERATOR_TENSOR_PARALLEL_SIZE:-4}"
evaluator_quantization="${EVALUATOR_QUANTIZATION:-none}"
evaluator_max_tokens="${EVALUATOR_MAX_TOKENS:-256}"
feedback_model_clean="$(basename "$feedback_model")"
feedback_model_clean="${feedback_model_clean//-/_}"
output_folder="/workspace/daeyong/inference_results/dev_kg_correct_1ksample_no_premature_conclusion_10_3_${feedback_model_clean}"
PYTHON_BIN="${PYTHON_BIN:-/workspace/daeyong/conda_envs/vllm_new/bin/python}"
PYTHON_ENV_DIR="$(dirname "$(dirname "$PYTHON_BIN")")"
export DAEYONG_VLLM_TORCH_PRELOAD="${DAEYONG_VLLM_TORCH_PRELOAD:-1}"
export PYTHONPATH="/workspace/daeyong:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$PYTHON_ENV_DIR/lib:${LD_LIBRARY_PATH:-}"

debug_print_args=()
if [[ "${DEBUG_PRINT_IO:-1}" == "1" ]]; then
  debug_print_args+=(
    --debug_print_io
    --debug_print_limit "${DEBUG_PRINT_LIMIT:-3}"
    --debug_print_chars "${DEBUG_PRINT_CHARS:-4000}"
  )
fi

for model in \
"$generator_model_path"
# "/workspace/hf_transformers/Meta-Llama-3.1-8B-Instruct" \
# "/workspace/hf_transformers/gemma-3-12b-it" \
# "/workspace/hf_transformers/Qwen3-4B-Instruct-2507" \
# "/workspace/hf_transformers/Qwen3-8B" \
# "/workspace/hf_transformers/gemma-4-31B-it" \
# "/workspace/hf_transformers/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
do
  for dataset in "2wiki" "hotpotqa" "musique"
  do
    echo "Starting inference for model: $model / dataset: $dataset"

    CUDA_VISIBLE_DEVICES="4,5,6,7" "$PYTHON_BIN" inference_vllm.py --dataset "$dataset" --generator_model "$model" --generator_quantization "$generator_quantization" --generator_tensor_parallel_size "$generator_tensor_parallel_size" --feedback_model "$feedback_model" --track_cache_stats --cache_stats_mode exact_or_fallback --max_steps 10 --max_retries 3 --evaluator_quantization "$evaluator_quantization" --evaluator_max_tokens "$evaluator_max_tokens" "${debug_print_args[@]}"

  done
done

CUDA_VISIBLE_DEVICES="4,5,6,7" "$PYTHON_BIN" final_answer.py --folder_path "$output_folder" --models "$generator_model_name"
CUDA_VISIBLE_DEVICES="4,5,6,7" "$PYTHON_BIN" oss_answer_binary.py --folder_path "$output_folder" --models "$generator_model_name"
