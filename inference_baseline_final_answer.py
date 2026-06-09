import pandas as pd
import argparse
import os
import json
import ast
import gc
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# --- System Prompt Definition ---
ANSWER_GENERATION_SYSTEM_PROMPT = """You are an expert answering agent.
The reasoning process is complete. Your task is to formulate the FINAL ANSWER based on the provided history.

INSTRUCTIONS:
1. Do not generate any new reasoning steps.
2. Directly output the final answer.
3. YOU MUST USE THE FOLLOWING FORMAT:
####ANSWER: your_final_answer_here (Final Answer)"""

def parse_cli_tokens(values):
    tokens = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if token:
                tokens.append(token)
    return list(dict.fromkeys(tokens))

def load_vllm_model(model_path: str, gpu_memory_utilization: float = 0.90):
    print(f"\n🔵 Loading Model: '{model_path}'")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    llm = LLM(
        model=model_path,
        tensor_parallel_size=4,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=3000, # oss는 3000, 나머지는 2000
        enforce_eager=False, 
        enable_prefix_caching=True
    )
    
    print("✅ Model loaded successfully.")
    return llm, tokenizer

def run_answer_generation(
    df: pd.DataFrame,
    llm: LLM,
    tokenizer,
    result_file_path: str,
    disable_thinking: bool = False,
):
    prompts = []
    metadata = [] 
    
    print(f"🔄 Preparing prompts for {len(df)} samples...")
    
    for _, row in df.iterrows():
        reasoning_input = row.get('generated_steps', '')
        if isinstance(reasoning_input, str):
            if reasoning_input.strip().startswith('[') and reasoning_input.strip().endswith(']'):
                try: 
                    steps = ast.literal_eval(reasoning_input)
                    reasoning_str = '\n'.join([f"Step {i+1}: {step}" for i, step in enumerate(steps)])
                except: 
                    reasoning_str = reasoning_input 
            else:
                reasoning_str = reasoning_input
        elif isinstance(reasoning_input, list):
            reasoning_str = '\n'.join([f"Step {i+1}: {step}" for i, step in enumerate(reasoning_input)])
        else:
            reasoning_str = "No reasoning provided."
        
        user_content = f"""Question: {row['question']}

Provided Reasoning Process:
{reasoning_str}

Based on the reasoning process above, what is the final answer?
Remember to start your response with ####ANSWER:"""

        messages = [
            {"role": "system", "content": ANSWER_GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        template_kwargs = {"add_generation_prompt": True, "tokenize": False}
        if disable_thinking:
            template_kwargs["enable_thinking"] = False
        full_prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
        prompts.append(full_prompt)
        
        metadata.append({
            "id": row['id'],
            "question": row['question'],
            "reasoning_steps": reasoning_str,
            "ground_truth": row.get('ground_truth', 'N/A')
        })

    stop_tokens = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    stop_tokens = [t for t in stop_tokens if t is not None]
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=128, # oss는 1024, 나머지는 128
        stop_token_ids=stop_tokens
    )

    print(f"🚀 Starting Answer Generation for {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    total_tokens = 0
    
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text.strip()
        total_tokens += len(output.outputs[0].token_ids)
        
        meta = metadata[i]
        
        final_ans = None
        if "####ANSWER:" in generated_text:
            try:
                final_ans = generated_text.split("####ANSWER:")[-1].strip().split("\n")[0]
            except: 
                final_ans = "Parsing Error"
        else:
            final_ans = generated_text 
        
        res_obj = {
            "id": meta["id"],
            "question": meta["question"],
            "input_reasoning": meta["reasoning_steps"],
            "full_response": generated_text,
            "final_answer_extracted": final_ans,
            "ground_truth": meta["ground_truth"]
        }
        results.append(res_obj)

    print(f"💾 Saving results to {result_file_path}...")
    with open(result_file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

# --- Main Logic ---
if __name__ == "__main__":
    # 모델 이름과 실제 경로 매핑 (사용 환경에 맞게 수정 필요)
    MODEL_MAPPING = {
        "Qwen3-4B-Instruct-2507": "/workspace/hf_transformers/Qwen3-4B-Instruct-2507",
        "Qwen3-8B": "/workspace/hf_transformers/Qwen3-8B",
        "Qwen3.6-27B": "/workspace/hf_transformers/Qwen3.6-27B",
        "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8": "/workspace/hf_transformers/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
    }
    # MODEL_MAPPING = {
    #     "gpt-oss-120b": os.path.join("/workspace/hf_transformers", "gpt-oss-120b")
    # }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Qwen3-4B-Instruct-2507", "Qwen3-8B", "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"],
    )
    parser.add_argument("--datasets", nargs="+", default=["2wiki", "hotpotqa", "musique"])
    args = parser.parse_args()

    target_models = parse_cli_tokens(args.models)
    # target_models = ["gpt-oss-120b"]
    datasets = parse_cli_tokens(args.datasets)

    print(f"🚀 Start Sequential Processing: {len(target_models)} Models x {len(datasets)} Datasets")

    for model_name in target_models:
        model_path = MODEL_MAPPING.get(model_name)
        if not model_path or not os.path.exists(model_path):
            print(f"❌ Model path not found for {model_name}: {model_path}")
            continue

        # 1. 모델 로드 (Loop 내부로 이동)
        try:
            llm, tokenizer = load_vllm_model(model_path)
        except Exception as e:
            print(f"❌ Failed to load model {model_name}: {e}")
            continue
        is_qwen3_model = "qwen3" in model_name.lower()
        
        current_input_dir = os.path.join("/workspace/daeyong/inference_results", f"no_feedback_{model_name}")

        # 2. 해당 모델에 대한 3개 데이터셋 처리
        for dataset in datasets:
            print(f"\n-----------------------------------------------------------")
            print(f"📂 Processing: [{model_name}] -> [{dataset}]")
            print(f"-----------------------------------------------------------")

            input_file = os.path.join(current_input_dir, f"{dataset}_results.json")
            result_file_path = os.path.join(current_input_dir, f"{dataset}_final_answer.json")

            if not os.path.exists(input_file):
                print(f"⚠️ Input file missing, skipping: {input_file}")
                continue

            # Resume Logic
            processed_ids = set()
            if os.path.exists(result_file_path):
                try:
                    with open(result_file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        processed_ids = {item['id'] for item in data}
                    print(f"🔄 Resuming... {len(processed_ids)} already processed.")
                except: pass

            try:
                df = pd.read_json(input_file)
                df_filtered = df[~df['id'].isin(processed_ids)]

                if len(df_filtered) > 0:
                    run_answer_generation(
                        df=df_filtered,
                        llm=llm,
                        tokenizer=tokenizer,
                        result_file_path=result_file_path,
                        disable_thinking=is_qwen3_model,
                    )
                    print(f"✅ Completed: {dataset}")
                else:
                    print(f"⏭️  All data already processed.")
            except Exception as e:
                print(f"❌ Error processing {dataset}: {e}")

        # 3. 모델 메모리 해제 (다음 모델 로드를 위해 필수)
        print(f"🧹 Unloading model {model_name} to free GPU memory...")
        del llm
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print("✅ Memory cleared.\n")

    print("\n🎉 All tasks finished successfully.")
