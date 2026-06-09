system_prompt = """You are a data generation assistant designed to create specific "Reasoning Error" examples for training feedback models.
Your task is to generate a chain of reasoning steps (Chain of Thought) that demonstrates a **"Correct Reasoning but Wrong Conclusion"** error.

**Input:**
- A "comparison" question about movie directors (e.g., who is younger, who died earlier).
- A list of retrieved passages containing the necessary information.

**Output:**
- A python list of strings, where each string is a reasoning step.

**Strict Generation Rules:**
1.  **Attribution Steps (Start):** Correctly identify the directors of both films and their birth/death years from the passages. Use the format: `Step N: According to Passage X, [Fact]. (Attribution)`
2.  **Logical Step (Middle):** Perform a **mathematically and logically correct** comparison. Explicitly state which director fits the condition (e.g., "Since 1990 is later than 1980, Director A is younger."). Use the format: `Step N: Since [Fact A] and [Fact B], [Correct Conclusion]. (Logical)`
3.  **Final Answer Step (End - The Error):** This is the most important step. **Deliberately choose the WRONG film.** If the logical step says "Director A is younger", the final answer MUST be "Film B". Do NOT provide the correct answer. Use the format: `Step N: ####ANSWER: [Wrong Film Name] (Final Answer)`

**Template Example:**
User Question: Which film whose director is younger, Film A or Film B?
Passage 1: Director A (born 1990) directed Film A.
Passage 2: Director B (born 1950) directed Film B.

**Your Output:**
[
"Step 1: According to Passage 1, the director of Film A is Director A. (Attribution)",
"Step 2: According to Passage 1, Director A was born in 1990. (Attribution)",
"Step 3: According to Passage 2, the director of Film B is Director B. (Attribution)",
"Step 4: According to Passage 2, Director B was born in 1950. (Attribution)",
"Step 5: Since Director A was born in 1990 and Director B was born in 1950, Director A is younger. (Logical)",
"Step 6: ####ANSWER: Film B (Final Answer)"
]

**Note:**
- Ensure passage numbers match the provided context.
- Ensure the extraction and comparison logic are perfect.
- **ONLY** the Final Answer step should be incorrect (the opposite mapping).
- Output **only** the list of strings.
""".strip()

import json
import os
import re
import ast
import argparse
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# =============================================================================
# 1. Configuration & System Prompt
# =============================================================================
MODEL_NAME = "/workspace/hf_transformers/gpt-oss-120b"
INPUT_FILE = "/workspace/daeyong/fourth_finetuning_data/final_swap_2.json"
OUTPUT_FILE = "/workspace/daeyong/fourth_finetuning_data/final_swap_generated_2.json"

# =============================================================================
# 2. Helper Functions
# =============================================================================
def extract_python_list(text):
    """
    모델 출력에서 Python List 형태의 문자열을 안전하게 추출 및 파싱합니다.
    """
    if not isinstance(text, str):
        return []

    text = text.strip().split("assistantfinal")[-1].strip()
    
    # Markdown Code Block 제거 (```python ... ``` 또는 ``` ... ```)
    match = re.search(r"```(?:python)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if match:
        content = match.group(1).strip()
    else:
        # 가장 바깥쪽 대괄호 [] 찾기
        match = re.search(r"(\[.*\])", text, re.DOTALL)
        if match:
            content = match.group(1).strip()
        else:
            return text  # 파싱 실패 시 원본 텍스트 반환

    try:
        # ast.literal_eval로 안전하게 파싱
        return ast.literal_eval(content)
    except (ValueError, SyntaxError):
        # 파싱 실패 시 텍스트 반환 (추후 후처리 필요)
        return content

# =============================================================================
# 3. Main Logic
# =============================================================================
def main():
    print(f"📂 Input Path: {INPUT_FILE}")
    print(f"📂 Output Path: {OUTPUT_FILE}")

    # --- Load Data ---
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        return

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df_input = pd.DataFrame(data)
    except ValueError:
        print("⚠️ JSON load failed. Trying pandas read_json...")
        df_input = pd.read_json(INPUT_FILE)
    
    print(f"✅ Loaded Data: {len(df_input)} records.")

    # 필수 컬럼 확인
    if 'question' not in df_input.columns or 'retrieved_passages' not in df_input.columns:
        print("❌ Error: Input file must contain 'question' and 'retrieved_passages' columns.")
        return

    # --- Resume Logic ---
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding='utf-8') as f:
            try:
                existing_results = json.load(f)
            except json.JSONDecodeError:
                existing_results = []
        
        # 이미 처리된 질문 필터링
        processed_questions = {item['question'] for item in existing_results if 'question' in item}
        print(f"🔄 Resuming... Found {len(processed_questions)} processed items.")
        
        # 중복 방지를 위해 필터링
        df_input = df_input[~df_input['question'].isin(processed_questions)]
    else:
        existing_results = []

    if df_input.empty:
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
        max_model_len=10000,
        enable_prefix_caching=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=6000, 
    )

    # --- Batch Processing ---
    BATCH_SIZE = 100
    records = df_input.to_dict('records')
    final_results = existing_results

    print(f"🚀 Starting execution in batches of {BATCH_SIZE}...")

    for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Processing Batches"):
        batch_records = records[i : i + BATCH_SIZE]
        batch_prompts = []
        
        # Prompt Construction
        for row in batch_records:
            question = row.get('question', '')
            passages = row.get('retrieved_passages', [])
            
            # Passages 포맷팅 (리스트인 경우 줄바꿈으로 연결)
            if isinstance(passages, list):
                passages_text = "\n".join(passages)
            else:
                passages_text = str(passages)

            user_content = f"Question: {question}\nPassages:\n{passages_text}"

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

        # Process Outputs
        new_results = []
        for row, output in zip(batch_records, outputs):
            generated_text = output.outputs[0].text.strip()
            
            # 텍스트 파싱 (리스트 추출)
            reasoning_steps = extract_python_list(generated_text)
            
            result_entry = row.copy()
            # 생성된 Wrong Conclusion Reasoning Steps 저장
            result_entry["generated_wrong_reasoning"] = reasoning_steps
            # 원본 생성 텍스트도 백업용으로 저장 (선택 사항)
            result_entry["raw_generation"] = generated_text
            
            new_results.append(result_entry)

        # Incremental Save
        final_results.extend(new_results)
        
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"🎉 All Completed. Total items saved: {len(final_results)}")
    print(f"📂 Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()