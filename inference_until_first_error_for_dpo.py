import pandas as pd
import numpy as np
import argparse
import os
import json
import ast
import re
from tqdm import tqdm
from copy import deepcopy
from collections import deque

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

from prompts import evaluate_system_prompt_premature_attribution, generate_single_step_system_prompt

def load_generator_model(model_id: str, gpu_memory_utilization: float = 0.45):
    print(f"Generator 모델 로딩 중 (vLLM)... Model: '{model_id}'")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    llm = LLM(
        model=model_id,
        tensor_parallel_size=2,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=8192,
        enforce_eager=True,
        enable_prefix_caching=True
    )
    return llm, tokenizer

def load_finetuned_evaluator(base_model_id: str, adapter_path: str, gpu_memory_utilization: float = 0.45):
    print(f"평가자(Evaluator) 모델 로딩 중 (vLLM 4bit)... Base: '{base_model_id}'")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=base_model_id,
        enable_lora=True,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        tensor_parallel_size=2,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=8192,
        max_lora_rank=64,
        enforce_eager=True,
        enable_prefix_caching=True
    )
    return llm, tokenizer

def parse_eval_response(response_text: str) -> dict:
    """Evaluator 응답 파싱 (기존 로직 유지)"""
    fallback_result = {
        "error_type": "Parsing Error",
        "diagnosis": "No JSON object found.",
        "guidance": "Check output."
    }
    if not response_text: return fallback_result
    text = response_text.strip()
    
    markdown_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if markdown_match: text = markdown_match.group(1)

    start_idx = text.find('{')
    if start_idx == -1: return fallback_result
    json_str = text[start_idx:].strip()

    parsed = None
    # 1. Try standard load
    try:
        parsed = json.loads(json_str)
    except:
        # 2. Try truncating trailing characters
        end_idx = json_str.rfind('}')
        if end_idx != -1:
            try: parsed = json.loads(json_str[:end_idx+1])
            except: pass
    
    if parsed is None:
        # 3. Try literal eval
        try: parsed = ast.literal_eval(json_str)
        except: pass

    if parsed and isinstance(parsed, dict):
        if "error_type" not in parsed: parsed["error_type"] = "Unknown"
        return parsed
    else:
        return fallback_result

def run_inference_until_first_error(
    df: pd.DataFrame,
    gen_llm: LLM,
    gen_tokenizer,
    eval_llm: LLM,
    eval_tokenizer,
    adapter_path: str,
    result_file_path: str,
    max_steps: int = 10,
    batch_size: int = 32
):
    # 1. 작업 큐 초기화
    pending_queue = deque()
    for _, row in df.iterrows():
        pending_queue.append(row)

    active_states = []
    
    def append_to_json_file(file_path, new_data):
        if not new_data: return
        data = []
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except: pass
        data.extend(new_data)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def create_new_state(row):
        context_source = row['retrieved_passages']
        if isinstance(context_source, str):
            try: passages = ast.literal_eval(context_source)
            except: passages = [context_source]
        elif isinstance(context_source, list): passages = context_source
        else: passages = []
        
        return {
            "id": row['id'],
            "question": row['question'],
            "passages": passages,
            "ground_truth": row.get('answer', 'N/A'),
            "trace": [],  # [{step: str, feedback: dict, is_error: bool}, ...]
            "finished": False,
            "stop_reason": None, # 'Error', 'Answer', 'Max Steps'
            "temp_gen_text": None
        }

    pbar = tqdm(total=len(df), desc="Inference (Stop at First Error)")

    while pending_queue or active_states:
        # [Step 1] Refill Active Batch
        while len(active_states) < batch_size and pending_queue:
            new_row = pending_queue.popleft()
            active_states.append(create_new_state(new_row))
        
        if not active_states: break

        # [Step 2] Generator Phase
        gen_prompts = []
        
        for state in active_states:
            # 현재까지의 성공한 스텝들만 context로 제공
            # trace에 있는 모든 스텝은 이전 턴에서 'Correct' 판정을 받은 것들임 (마지막 에러 제외)
            previous_steps = [t['step'] for t in state['trace']]
            
            passages_str = '\n'.join([f"Passage {i+1}: {p}" for i, p in enumerate(state["passages"])])
            previous_steps_str = '\n'.join(previous_steps) if previous_steps else "(No previous steps.)"
            
            # 피드백은 제공하지 않음 (이전 스텝이 맞았다고 가정하고 다음 스텝 생성)
            # 혹은 "Correct" 피드백을 넣어줄 수도 있지만, 여기서는 단순화하여 생략하거나
            # "Previous step was correct" 정도만 명시 가능. 
            # 사용자 요청: "feedback model이 첫번째 에러를 탐지할 때 까지만" -> 즉, 수정 기회는 없음.
            
            prompt_user = f"""Question:
{state['question']}

Retrieved Passages:
{passages_str}

Previous Reasoning Steps:
{previous_steps_str}

Generate next step (start with `Step {len(previous_steps) + 1}:`)"""
            
            messages = [{"role": "system", "content": generate_single_step_system_prompt}, {"role": "user", "content": prompt_user}]
            gen_prompts.append(gen_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))

        # Batch Generation
        stop_candidates = [gen_tokenizer.eos_token_id, gen_tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        actual_stop_tokens = [t for t in stop_candidates if t is not None]

        gen_outputs = gen_llm.generate(
            gen_prompts, 
            SamplingParams(temperature=0.0, max_tokens=256, stop_token_ids=actual_stop_tokens), 
            use_tqdm=False
        )

        states_for_eval = []
        eval_prompts = []

        # [Step 3] Evaluator Phase Preparation
        for i, output in enumerate(gen_outputs):
            state = active_states[i]
            generated_text = output.outputs[0].text.strip()

            # Cleaning
            for marker in ["<start_of_turn>", "User:", "## Question"]:
                if marker in generated_text: generated_text = generated_text.split(marker)[0].strip()
            
            expected_start = f"Step {len(state['trace']) + 1}:"
            if not generated_text.startswith("Step"):
                generated_text = f"{expected_start} {generated_text}"
            
            state["temp_gen_text"] = generated_text
            states_for_eval.append(state)

            # Build Eval Prompt
            passages = state['passages']
            previous_steps = [t['step'] for t in state['trace']]
            
            context_str = '\n'.join([f"Passage {i+1}: {p}" for i, p in enumerate(passages)]).strip()
            previous_steps_str = '\n'.join(previous_steps).strip()
            
            user_content = f"""### Task: Evaluate the Correctness of the Reasoning Step

Question:
{state['question']}

Retrieved Passages:
{context_str}

Previous Steps:
{previous_steps_str}

Step to evaluate:
{state['temp_gen_text']}
""".strip()
            messages = [{"role": "system", "content": evaluate_system_prompt_premature_attribution}, {"role": "user", "content": user_content}]
            eval_prompts.append(eval_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))

        # [Step 4] Evaluator Execution
        eval_outputs = eval_llm.generate(
            eval_prompts, 
            SamplingParams(temperature=0.0, max_tokens=256, stop_token_ids=[eval_tokenizer.eos_token_id]), 
            lora_request=LoRARequest("evaluator_adapter", 1, adapter_path), 
            use_tqdm=False
        )

        # [Step 5] Process Results & Check Stop Conditions
        next_active_states = []
        finished_results = []

        for i, output in enumerate(eval_outputs):
            state = states_for_eval[i]
            raw_eval = output.outputs[0].text.strip()
            parsed_eval = parse_eval_response(raw_eval)
            
            error_type = parsed_eval.get("error_type", "Unknown")
            is_correct = "correct" in error_type.lower()
            has_answer_tag = "####ANSWER" in state["temp_gen_text"]

            # Record this turn
            turn_record = {
                "step": state["temp_gen_text"],
                "feedback": parsed_eval,
                "is_error": not is_correct
            }
            state["trace"].append(turn_record)

            # --- Stop Logic ---
            if not is_correct:
                # 1. Error Detected -> STOP
                state["finished"] = True
                state["stop_reason"] = "First Error Detected"
            elif has_answer_tag:
                # 2. Final Answer Reached (and it's correct so far) -> STOP
                state["finished"] = True
                state["stop_reason"] = "Answer Generated"
            elif len(state["trace"]) >= max_steps:
                # 3. Max Steps Reached -> STOP
                state["finished"] = True
                state["stop_reason"] = "Max Steps Reached"
            else:
                # 4. Correct Step, Continue
                state["finished"] = False

            if state["finished"]:
                # Save simplified result
                res_obj = {
                    "id": state["id"],
                    "question": state["question"],
                    "passages": state["passages"],
                    "trace": state["trace"], # [step1, feedback1, step2_error, feedback2_error]
                    "stop_reason": state["stop_reason"],
                    "ground_truth": state["ground_truth"]
                }
                finished_results.append(res_obj)
                pbar.update(1)
            else:
                next_active_states.append(state)

        # Save & Advance
        if finished_results:
            append_to_json_file(result_file_path, finished_results)
        
        active_states = next_active_states

    pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--generator_model", type=str, required=True)
    parser.add_argument("--feedback_model", type=str, required=True)
    args = parser.parse_args()

    # Model Name Parsing
    if "llama" in args.generator_model.lower(): model_name = "llama8b"
    elif "qwen" in args.generator_model.lower(): model_name = "qwen7b"
    elif "gemma" in args.generator_model.lower(): model_name = "gemma12b"
    else: model_name = "unknown"

    base_model_id = "/workspace/hf_transformers/Qwen2.5-7B-Instruct"
    adapter_path = f"/workspace/daeyong/trained_models/{args.feedback_model}"

    # Load Models
    gen_llm, gen_tokenizer = load_generator_model(args.generator_model, gpu_memory_utilization=0.45)
    eval_llm, eval_tokenizer = load_finetuned_evaluator(base_model_id, adapter_path, gpu_memory_utilization=0.45)

    # Load Dataset
    if args.dataset == '2wiki':
        df = pd.read_csv("/workspace/daeyong/benchmarks/2wiki_for_first_dpo.csv").sample(n=300, random_state=42)
    elif args.dataset == 'hotpotqa':
        df = pd.read_csv("/workspace/daeyong/benchmarks/hotpotqa_for_first_dpo.csv").sample(n=300, random_state=42)
    elif args.dataset == 'musique':
        df = pd.read_csv("/workspace/daeyong/benchmarks/musique_for_first_dpo.csv").sample(n=300, random_state=42)
    else:
        print(f"Dataset {args.dataset} not found")
        exit()

    feedback_model_clean = args.feedback_model.strip("/").replace("/", "_").replace("-", "_")
    output_dir = f"/workspace/daeyong/inference_results/{feedback_model_clean}_until_first_error_dpo"
    os.makedirs(output_dir, exist_ok=True)

    result_file_path = os.path.join(output_dir, f"{model_name}_{args.dataset}_trace.json")
    
    # Resume Logic
    processed_ids = set()
    if os.path.exists(result_file_path):
        try:
            with open(result_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                processed_ids = {item['id'] for item in data}
            print(f"🔄 Resuming... Found {len(processed_ids)} processed examples.")
        except: pass
    
    df = df[~df['id'].isin(processed_ids)]

    if len(df) > 0:
        run_inference_until_first_error(
            df, gen_llm, gen_tokenizer, eval_llm, eval_tokenizer,
            adapter_path=adapter_path,
            result_file_path=result_file_path
        )
        print("✅ Done.")
    else:
        print("✅ Nothing to process.")