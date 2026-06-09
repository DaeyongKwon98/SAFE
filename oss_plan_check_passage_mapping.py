import pandas as pd
from tqdm import tqdm
import json
import os
import re
import ast
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from prompts import passage_mapping_2wiki, passage_mapping_hotpotqa, passage_mapping_musique

# =============================================================================
# 1. Configuration
# =============================================================================
MODEL_NAME = "/workspace/hf_transformers/gpt-oss-120b"

# =============================================================================
# 2. Helper Functions
# =============================================================================
def extract_json_output(text):
    """
    모델 출력에서 JSON 또는 파이썬 리스트 형태의 데이터를 안전하게 추출합니다.
    """
    if not isinstance(text, str):
        return text

    # 1. 텍스트 정규화: 불필요한 제어 문자 제거 및 이스케이프된 줄바꿈 처리
    text = text.strip()
    
    # 2. 대괄호 추출 로직 (Greedy 매칭으로 변경하여 가장 바깥쪽 [ ] 를 잡음)
    # [.*]는 처음 '['부터 마지막 ']'까지 통째로 잡습니다.
    match = re.search(r"(\[.*\])", text, re.DOTALL)
    
    if not match:
        # 마크다운 코드 블록 내부에 있을 경우를 위해 한 번 더 시도
        match = re.search(r"```(?:json)?\s*(\[.*\])\s*```", text, re.DOTALL)
        
    if match:
        content = match.group(1).strip()
    else:
        return text

    # 3. 파싱 시도 (JSON -> AST)
    result = None
    try:
        # 표준 JSON 파싱
        result = json.loads(content)
    except json.JSONDecodeError:
        try:
            # JSON 실패 시 파이썬 리터럴(Single Quote 등) 파싱
            result = ast.literal_eval(content)
        except (ValueError, SyntaxError):
            # 파싱에 완전히 실패한 경우 정제된 문자열이라도 반환
            return content

    # 4. Double-encoded 처리 (결과가 여전히 문자열인 경우 한 번 더 파싱)
    if isinstance(result, str):
        return extract_json_output(result)
    
    return result


# =============================================================================
# 3. Main Execution Logic
# =============================================================================

def main(args):
    # --- Path Setup ---
    base_dir = "/workspace/daeyong"
    
    if args.dataset == "2wiki":
        plan_path = f"{base_dir}/reasoning_plans/2wiki_dev_plan.json"
        data_path = f"{base_dir}/benchmarks/2wiki_dev.csv"
        output_path = f"{base_dir}/reasoning_plans/2wiki_dev_plan_passage_mapped.json"
        system_prompt = passage_mapping_2wiki
    elif args.dataset == "hotpotqa":
        plan_path = f"{base_dir}/reasoning_plans/hotpotqa_dev_plan.json"
        data_path = f"{base_dir}/benchmarks/hotpotqa_dev.csv"
        output_path = f"{base_dir}/reasoning_plans/hotpotqa_dev_plan_passage_mapped.json"
        system_prompt = passage_mapping_hotpotqa
    elif args.dataset == "musique":
        plan_path = f"{base_dir}/reasoning_plans/musique_dev_plan.json"
        data_path = f"{base_dir}/benchmarks/musique_dev.csv"
        output_path = f"{base_dir}/reasoning_plans/musique_dev_plan_passage_mapped.json"
        system_prompt = passage_mapping_musique

    print(f"📂 Plan Path: {plan_path}")
    print(f"📂 Data Path: {data_path}")

    # --- Load Data ---
    if not os.path.exists(plan_path):
        print("❌ Plan file not found!")
        return

    df_plans = pd.read_json(plan_path)
    df_data = pd.read_csv(data_path)

    # --- Merge Data ---
    # Question 기준으로 Inner Join (Plan이 생성된 데이터만 처리)
    df_plans['question'] = df_plans['question'].str.strip()
    df_data['question'] = df_data['question'].str.strip()
    
    merged_df = pd.merge(df_plans, df_data, on='question', how='inner')
    print(f"✅ Merged Data Count: {len(merged_df)} items")

    # --- Resume Logic ---
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            existing_results = json.load(f)
        processed_questions = {item['question'] for item in existing_results}
        print(f"🔄 Resuming... Found {len(processed_questions)} processed items.")
        merged_df = merged_df[~merged_df['question'].isin(processed_questions)]
    else:
        existing_results = []

    if merged_df.empty:
        print("✅ No new questions to process.")
        return

    # --- Initialize vLLM ---
    print(f"🚀 Loading vLLM Model: {MODEL_NAME}")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=4,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        max_model_len=8000,
        enable_prefix_caching=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=4000,
    )

    # --- Batch Processing ---
    BATCH_SIZE = 100
    records = merged_df.to_dict('records')
    final_results = existing_results

    print(f"🚀 Starting execution in batches of {BATCH_SIZE}...")

    for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Processing Batches"):
        batch_records = records[i : i + BATCH_SIZE]
        batch_prompts = []
        
        # Prepare Prompts
        for row in batch_records:
            question = row['question']
            if isinstance(row['plan'], str):
                plan_list = ast.literal_eval(row['plan'])
            else:
                plan_list = row['plan']
                
            if isinstance(row["gt_passages"], str):
                gt_passages = ast.literal_eval(row["gt_passages"])
            else:
                gt_passages = row["gt_passages"]
                
            formatted_passages = "\n".join([f"Passage {idx+1}: {p}" for idx, p in enumerate(gt_passages)])
            formatted_plan = "\n".join(plan_list)
            
            user_content = f"""Question: {question}

Ground Truth Context:
{formatted_passages}

Reasoning Plan:
{formatted_plan}"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            
            full_prompt = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=False
            )
            batch_prompts.append(full_prompt)

        # Generate
        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)

        # Process Results
        new_results = []
        for row, output in zip(batch_records, outputs):
            generated_text = output.outputs[0].text.split("assistantfinal")[-1].strip()
            
            parsed_execution = []
            
            try:
                parsed_execution = extract_json_output(generated_text)
                # if isinstance(parsed_execution, str):
                #     parsed_execution = json.loads(parsed_execution)
            except Exception:
                # 파싱 실패 시 Raw Output 저장 (디버깅용)
                parsed_execution = {"error": "parsing_failed", "raw_output": generated_text}

            result_entry = {
                "question": row['question'],
                "plan": row['plan'],
                "passage_mapping": parsed_execution,
            }
            new_results.append(result_entry)

        # Save Incrementally
        final_results.extend(new_results)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"🎉 All Completed. Total items saved: {len(final_results)}")
    print(f"📂 Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["2wiki", "hotpotqa", "musique"], required=True)
    args = parser.parse_args()
    main(args)