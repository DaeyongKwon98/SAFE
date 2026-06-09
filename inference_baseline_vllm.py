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
from transformers import AutoTokenizer

from prompts import generate_single_step_system_prompt_nofeedback

def load_vllm_model(model_id: str, gpu_memory_utilization: float = 0.90):
    print(f"Loading Single Model (vLLM)... Model: '{model_id}'")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    llm = LLM(
        model=model_id,
        tensor_parallel_size=4,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=8192,
        enforce_eager=False, 
        enable_prefix_caching=True,
        seed=42
    )
    
    print("✅ Model loaded successfully.")
    return llm, tokenizer

def run_iterative_reasoning(
    df: pd.DataFrame,
    llm: LLM,
    tokenizer,
    result_file_path: str,
    stats_file_path: str,
    max_steps: int = 10,
    batch_size: int = 32,
    disable_thinking: bool = False,
):
    # 1. 초기화
    pending_queue = deque()
    for _, row in df.iterrows():
        pending_queue.append(row)

    active_states = [] 
    total_stats = {"generator_calls": 0, "total_tokens": 0, "completed_count": 0}
    
    def append_to_json_file(file_path, new_data):
        if not new_data: return
        data = []
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except: pass
        if isinstance(new_data, list): data.extend(new_data)
        else: data.append(new_data)
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
            "step_texts": [],       # 생성된 Step들을 저장하는 리스트
            "finished": False,
            "ground_truth": row.get('answer', 'N/A')
        }

    pbar = tqdm(total=len(df), desc="Iterative Reasoning Processing")
    
    stop_tokens = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    stop_tokens = [t for t in stop_tokens if t is not None]
    
    # Generator용 Sampling Params
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
        stop_token_ids=stop_tokens
    )

    # -------------------------------------------------------------------------
    # Main Loop
    # -------------------------------------------------------------------------
    while pending_queue or active_states:
        
        # [Refill] 배치 크기만큼 채우기
        while len(active_states) < batch_size and pending_queue:
            active_states.append(create_new_state(pending_queue.popleft()))
        if not active_states: break

        # =========================================================================
        # SINGLE PHASE: GENERATION (Next Step)
        # =========================================================================
        gen_prompts = []
        gen_indices = []
        
        for idx, state in enumerate(active_states):          
            passages_str = '\n'.join([f"Passage {i+1}: {p}" for i, p in enumerate(state["passages"])])
            previous_steps_str = '\n'.join(state["step_texts"]) if state["step_texts"] else "(No previous steps. Start with Step 1.)"
            
            # 다음 스텝 번호 계산
            next_step_num = len(state["step_texts"]) + 1
            
            # 프롬프트 구성
            user_content = f"""Question: {state['question']}

Retrieved Passages:
{passages_str}

Previous Reasoning Steps:
{previous_steps_str}

Generate next step (start with `Step {next_step_num}:`)."""

            messages = [
                {"role": "system", "content": generate_single_step_system_prompt_nofeedback}, 
                {"role": "user", "content": user_content}
            ]

            template_kwargs = {"add_generation_prompt": True, "tokenize": False}
            if disable_thinking:
                template_kwargs["enable_thinking"] = False
            gen_prompts.append(tokenizer.apply_chat_template(messages, **template_kwargs))
            gen_indices.append(idx)

        # vLLM Batch Inference
        if gen_prompts:
            gen_outputs = llm.generate(gen_prompts, sampling_params, use_tqdm=False)
            total_stats["generator_calls"] += len(gen_outputs)

            for i, output in enumerate(gen_outputs):
                state_idx = gen_indices[i]
                state = active_states[state_idx]
                
                gen_text = output.outputs[0].text.strip()
                total_stats["total_tokens"] += len(output.prompt_token_ids) + len(output.outputs[0].token_ids)

                # 메타데이터 토큰 제거 (Cleaning)
                for marker in ["<start_of_turn>", "User:", "## Question"]:
                    if marker in gen_text: gen_text = gen_text.split(marker)[0].strip()
                
                # 결과 저장
                state["step_texts"].append(gen_text)
                
                # 종료 태그 발견
                if "END_OF_REASONING" in gen_text:
                    state["finished"] = True

        # =========================================================================
        # CLEANUP & SAVE
        # =========================================================================
        next_active_states = []
        finished_results = []

        for state in active_states:
            # 종료 조건 체크 2: 최대 스텝 도달
            if not state["finished"] and len(state["step_texts"]) >= max_steps:
                state["finished"] = True

            if state["finished"]:
                res_obj = {
                    "id": state["id"],
                    "question": state["question"],
                    "passages": state['passages'],
                    "generated_steps": state["step_texts"],
                    "ground_truth": state["ground_truth"]
                }
                finished_results.append(res_obj)
                pbar.update(1)
            else:
                next_active_states.append(state)

        if finished_results:
            append_to_json_file(result_file_path, finished_results)
            total_stats["completed_count"] += len(finished_results)
            append_to_json_file(stats_file_path, [deepcopy(total_stats)])

        active_states = next_active_states

    pbar.close()
    return total_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="/workspace/hf_transformers/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--sample_limit", type=int, default=0)
    parser.add_argument("--sample_seed", type=int, default=42)
    args = parser.parse_args()

    base_path = "/workspace/daeyong"
    
    try:
        # 데이터셋 로드
        df = pd.read_csv(f"{base_path}/benchmarks/{args.dataset}_dev_kg_correct.csv")
        if args.sample_size > 0:
            sample_n = min(args.sample_size, len(df))
            df = df.sample(n=sample_n, random_state=args.sample_seed)
        if args.sample_limit > 0:
            df = df.head(min(args.sample_limit, len(df)))
        print(f"Loaded {len(df)} samples from {args.dataset}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()

    model_name_clean = args.model_id.split("/")[-1]
    output_dir = f"{base_path}/inference_results/no_feedback_{model_name_clean}"
    os.makedirs(output_dir, exist_ok=True)
    
    result_file_path = os.path.join(output_dir, f"{args.dataset}_results.json")
    stats_file_path = os.path.join(output_dir, f"{args.dataset}_stats.json")

    # Resume Logic
    processed_ids = set()
    if os.path.exists(result_file_path):
        try:
            with open(result_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                processed_ids = {item['id'] for item in data}
            print(f"Resuming... {len(processed_ids)} already processed.")
        except: pass
    
    df = df[~df['id'].isin(processed_ids)]

    if len(df) > 0:
        llm, tokenizer = load_vllm_model(args.model_id)
        model_id_lower = args.model_id.lower()
        is_qwen3_model = "qwen3" in model_id_lower

        run_iterative_reasoning(
            df=df,
            llm=llm,
            tokenizer=tokenizer,
            result_file_path=result_file_path,
            stats_file_path=stats_file_path,
            max_steps=10,    
            batch_size=256,
            disable_thinking=is_qwen3_model,
        )
    else:
        print("All data processed.")
