import pandas as pd
from tqdm import tqdm
import json
import os
import re
import ast
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# =============================================================================
# 1. Configuration & System Prompt
# =============================================================================
MODEL_NAME = "/workspace/hf_transformers/gpt-oss-120b"

# 시스템 프롬프트
FINAL_ANSWER_SYSTEM_PROMPT = """You are an expert AI assistant specializing in multi-hop question answering.
Your task is to synthesize the **Final Answer** for a given question, strictly following the provided **Ideal Reasoning Steps**.

**Instructions:**
1. You will be provided with a `Question`, `Ground Truth Context`, and `Ideal Reasoning Steps`.
2. The `Ideal Reasoning Steps` contain the correct logical path and evidence to solve the question.
3. You must derive the final answer **solely** based on the conclusion reached in the `Ideal Reasoning Steps`.
4. Do not generate new reasoning or external knowledge. Trust the steps provided.
5. Prioritize Conciseness: Provide the answer in the concise form possible (e.g., entity, date, short phrase). Avoid full sentences or conversational fillers unless the question explicitly requires a descriptive answer.
6. **Format:** Output ONLY a valid JSON object with a single key "final_answer".
   - Example: {"final_answer": "Steve Jobs"}
   - Example: {"final_answer": "1994"}
   - Example: {"final_answer": "yes"}

**Input Format:**
Question: ...
Ground Truth Context: ...
Ideal Reasoning Steps: ...

**Output:**
{"final_answer": "Your final answer here"}
""".strip()

# =============================================================================
# 2. Helper Functions
# =============================================================================
def extract_json_output(text):
    """
    모델 출력에서 JSON 데이터를 안전하게 추출합니다.
    """
    if not isinstance(text, str):
        return text

    text = text.strip()
    
    # Markdown Code Block 제거
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        content = match.group(1).strip()
    else:
        # 가장 바깥쪽 중괄호 {} 찾기
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            content = match.group(1).strip()
        else:
            return text

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(content)
        except (ValueError, SyntaxError):
            return content

def format_steps_list(steps_data):
    """
    Ideal Steps(List of Dicts)에서 'ideal_step' 텍스트만 추출하여 포맷팅합니다.
    """
    try:
        # 문자열인 경우 파싱
        if isinstance(steps_data, str):
            steps = ast.literal_eval(steps_data)
        else:
            steps = steps_data
        
        formatted = []
        for step in steps:
            # step은 {"ideal_step": "Step 1...", "supporting_index": ...} 형태
            if isinstance(step, dict) and "ideal_step" in step:
                text = step["ideal_step"]
            else:
                # 딕셔너리가 아닌 경우(단순 문자열 등) fallback
                text = str(step)
            formatted.append(text)
            
        return "\n".join(formatted)
    except Exception:
        return str(steps_data)

# =============================================================================
# 3. Main Logic
# =============================================================================
def main(args):
    # --- Path Definition ---
    base_dir = "/workspace/daeyong"
    input_path = f"{base_dir}/ideal_steps/{args.dataset}_dev_filtered_fixed.json"
    output_path = f"{base_dir}/ideal_steps/{args.dataset}_dev_final_answer_fixed.json"
    benchmark_path = f"{base_dir}/benchmarks/{args.dataset}_dev.csv"
    # benchmark_path = f"{base_dir}/benchmarks/{args.dataset}_dev_answerable.json"

    print(f"📂 Input Path: {input_path}")
    print(f"📂 Benchmark Path: {benchmark_path}")
    print(f"📂 Output Path: {output_path}")

    # --- Load Data ---
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return

    # 1. Load Filtered Reasoning Steps (JSON)
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
        df_input = pd.DataFrame(data)
    except ValueError:
        print("❌ JSON decoding failed. Trying pandas read_json...")
        df_input = pd.read_json(input_path)

    # 2. Load Benchmark Passages (CSV)
    if not os.path.exists(benchmark_path):
        print(f"❌ Benchmark file not found: {benchmark_path}")
        return
    passage_df = pd.read_csv(benchmark_path)
    # passage_df = pd.read_json(benchmark_path)

    # Preprocessing for Merge
    df_input['question'] = df_input['question'].str.strip()
    passage_df['question'] = passage_df['question'].str.strip()

    # 3. Merge Dataframes (question 기준)
    # df_input에는 ideal_steps가 있고, passage_df에는 원본 gt_passages가 있음
    # 컬럼 충돌 방지를 위해 passage_df에서는 필요한 컬럼만 선택
    passage_subset = passage_df[['question', 'gt_passages']]
    # passage_subset = passage_df[['question', 'retrieved_passages']]
    
    # Inner Join: 필터링된 질문들에 대해서만 Passage 정보를 붙임
    merged_df = pd.merge(df_input, passage_subset, on='question', how='inner')
    
    print(f"✅ Loaded Input: {len(df_input)} records.")
    print(f"✅ Merged Data: {len(merged_df)} records (Passages attached).")

    # --- Resume Logic ---
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            try:
                existing_results = json.load(f)
            except json.JSONDecodeError:
                existing_results = []
        
        processed_questions = {item['question'] for item in existing_results if 'question' in item}
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
        max_model_len=6000,
        enable_prefix_caching=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2048, 
    )

    # --- Batch Processing ---
    BATCH_SIZE = 100
    records = merged_df.to_dict('records')
    final_results = existing_results

    print(f"🚀 Starting execution in batches of {BATCH_SIZE}...")

    for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Processing Batches"):
        batch_records = records[i : i + BATCH_SIZE]
        batch_prompts = []
        
        # Prompt Construction
        for row in batch_records:
            question = row.get('question', '')
            
            # Merge된 데이터프레임에서 gt_passages 가져오기
            gt_passages = row.get('gt_passages', [])
            
            # ideal_steps 리스트에서 텍스트만 추출하여 포맷팅
            ideal_steps_raw = row.get('ideal_steps', [])
            formatted_steps = format_steps_list(ideal_steps_raw)
            
            user_content = f"""Question: {question}

Ground Truth Context:
{gt_passages}

Ideal Reasoning Steps:
{formatted_steps}"""

            messages = [
                {"role": "system", "content": FINAL_ANSWER_SYSTEM_PROMPT},
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

        # Process Outputs
        new_results = []
        for row, output in zip(batch_records, outputs):
            generated_text = output.outputs[0].text.strip().split("assistantfinal")[-1].strip()
            
            # JSON Parsing
            parsed_result = extract_json_output(generated_text)
            
            # 결과 저장 (기존 데이터 유지 + final_answer 추가)
            result_entry = row.copy()
            
            if isinstance(parsed_result, dict) and "final_answer" in parsed_result:
                result_entry["generated_final_answer"] = parsed_result["final_answer"]
            else:
                # 파싱 실패 시 원본 텍스트 저장 (Fallback)
                result_entry["generated_final_answer"] = generated_text
            
            new_results.append(result_entry)

        # Incremental Save
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