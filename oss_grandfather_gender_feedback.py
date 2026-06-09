import json
import os
import re
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# =============================================================================
# 1. Configuration & Prompts
# =============================================================================
MODEL_NAME = "/workspace/hf_transformers/gpt-oss-120b"
INPUT_FILE = "/workspace/daeyong/fourth_finetuning_data/grandfather_gender_generated.json"
OUTPUT_FILE = "/workspace/daeyong/fourth_finetuning_data/grandfather_gender_feedback.json"

KINSHIP_SIDE_FEEDBACK_SYSTEM_PROMPT = """You are an expert AI evaluator providing feedback on reasoning chains involving family lineage and side-specific relationships (Paternal vs. Maternal).
Your task is to evaluate a specific **Logical Step** where the user commits a **"Logical Fallacy (Side Confusion)"**.

**Task:**
Identify the error where the reasoning correctly identifies the lineage (e.g., Mother's father) but incorrectly labels the side (e.g., calling the mother's father a "Paternal Grandfather").

**Requirements:**
1.  **error_type**: ALWAYS return "Logical Fallacy".
2.  **diagnosis**: Explain the specific mismatch between the lineage and the side label.
    - If the link is through the **Mother**, it must be **Maternal**.
    - If the link is through the **Father**, it must be **Paternal**.
    - State: "The reasoning identifies [Grandparent] as the [Relation] of the [Mother/Father]. This makes them the [Correct Side] [Grandparent], but the step incorrectly labels them as [Wrong Side] [Grandparent]."
3.  **guidance**: Provide the corrective rule. E.g., "Correct the side label. Since the relationship is traced through the [Mother/Father], the correct term is [Maternal/Paternal] [Grandparent]."

**Output Format:**
Provide ONLY a valid JSON object:
{
  "error_type": "Logical Fallacy",
  "diagnosis": "...",
  "guidance": "..."
}

**Few-Shot Examples:**

**Example 1 (Mother's father -> Paternal Error):**
Input:
Question: Who is the paternal grandfather of Prince William?
Passage 1: Prince William's mother was Princess Diana. Diana's father was Earl Spencer.
Reasoning Chain:
Step 1: According to Passage 1, the mother of Prince William is Princess Diana. (Attribution)
Step 2: According to Passage 1, the father of Princess Diana is Earl Spencer. (Attribution)
Step 3: Since Earl Spencer is the father of Prince William's mother, Earl Spencer is the paternal grandfather of Prince William. (Logical)

Output:
{
  "error_type": "Logical Fallacy",
  "diagnosis": "The reasoning correctly identifies Earl Spencer as the father of Prince William's mother (Princess Diana). However, a mother's father is a 'Maternal' grandfather. The step incorrectly labels him as a 'Paternal' grandfather.",
  "guidance": "Relationships through the mother are 'Maternal'. Earl Spencer is the maternal grandfather; the paternal grandfather would be the father of Prince William's father (King Charles III)."
}

**Example 2 (Father's mother -> Maternal Error):**
Input:
Question: Who is the maternal grandmother of King Felipe VI?
Passage 1: Felipe VI is the son of Juan Carlos I. Juan Carlos's mother was María de las Mercedes.
Reasoning Chain:
Step 1: According to Passage 1, the father of King Felipe VI is Juan Carlos I. (Attribution)
Step 2: According to Passage 1, the mother of Juan Carlos I is María de las Mercedes. (Attribution)
Step 3: Since María de las Mercedes is the mother of King Felipe VI's father, she is the maternal grandmother of King Felipe VI. (Logical)

Output:
{
  "error_type": "Logical Fallacy",
  "diagnosis": "The step identifies María de las Mercedes as the mother of the subject's father. Logically, a father's mother is a 'Paternal' grandmother, but the reasoning incorrectly calls her the 'Maternal' grandmother.",
  "guidance": "Since María de las Mercedes is linked via the father (Juan Carlos I), she is the paternal grandmother. To find the maternal grandmother, you must look for the mother of Felipe VI's mother (Queen Sofía)."
}
""".strip()

# =============================================================================
# 2. Helper Functions
# =============================================================================
def extract_json(text):
    text = text.strip()
    if "assistantfinal" in text:
        text = text.split("assistantfinal")[-1].strip()
        
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        content = match.group(1).strip()
    else:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            content = match.group(1).strip()
        else:
            return text

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content

# =============================================================================
# 3. Main Logic (수정 버전)
# =============================================================================
def main():
    print(f"📂 Input Path: {INPUT_FILE}")
    print(f"📂 Output Path: {OUTPUT_FILE}")

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        return

    # 1. 원본 데이터 로드
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    except Exception as e:
        print(f"⚠️ JSON load failed: {e}")
        return
    
    # 2. 기존 결과 로드 (Resume)
    existing_results = []
    processed_questions = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding='utf-8') as f:
                existing_results = json.load(f)
                # 질문(question)의 앞뒤 공백을 제거하여 비교의 정확도를 높임
                processed_questions = {str(item.get('question', '')).strip() for item in existing_results}
                print(f"🔄 Resuming... Found {len(processed_questions)} processed items.")
        except Exception as e:
            print(f"⚠️ Failed to load existing output (Starting fresh): {e}")

    # 3. 미처리 데이터만 추출
    to_process = [item for item in full_data if str(item.get('question', '')).strip() not in processed_questions]
    
    if not to_process:
        print("✅ All items have already been processed.")
        return

    print(f"🚀 To be processed: {len(to_process)} records.")

    # vLLM 초기화
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

    # Batch Processing
    BATCH_SIZE = 100

    final_results = existing_results

    

    print(f"🚀 Starting execution in batches of {BATCH_SIZE}...")

    for i in tqdm(range(0, len(to_process), BATCH_SIZE), desc="Processing Batches"):
        batch_records = to_process[i : i + BATCH_SIZE]
        batch_prompts = []
        
        # 실제 모델에 들어갈 데이터 인덱스 관리
        current_batch_indices = []

        for idx, row in enumerate(batch_records):
            question = row.get('question', '')
            passages = row.get('retrieved_passages', [])
            reasoning_steps = row.get('generated_kinship_error_reasoning', [])

            if not isinstance(reasoning_steps, list) or not reasoning_steps:
                continue
            
            # Passage 포맷팅
            if isinstance(passages, list):
                passages_text = "\n".join([f"Passage {j+1}: {p}" for j, p in enumerate(passages)])
            elif isinstance(passages, str):
                try:
                    passages_list = eval(passages)
                    passages_text = "\n".join([f"Passage {j+1}: {p}" for j, p in enumerate(passages_list)])
                except:
                    passages_text = passages
            else:
                passages_text = str(passages)
            
            reasoning_text = "\n".join(reasoning_steps)

            user_content = f"""Question: {question}
            
Retrieved Passages:
{passages_text}

Reasoning Chain:
{reasoning_text}

**Task:** Locate the logical step that assigns the final grandparent term and evaluate whether the side (Paternal/Maternal) correctly matches the gender of the intermediate parent identified in the attribution steps.
"""

            messages = [
                {"role": "system", "content": KINSHIP_SIDE_FEEDBACK_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
            
            full_prompt = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=False
            )
            batch_prompts.append(full_prompt)
            current_batch_indices.append(idx)

        if not batch_prompts:
            continue

        # vLLM 생성
        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)

        # 결과 처리
        for local_idx, output in enumerate(outputs):
            # batch_records에서 해당 데이터 가져오기
            original_row = batch_records[current_batch_indices[local_idx]]
            generated_text = output.outputs[0].text.strip()
            
            feedback_json = extract_json(generated_text)
            
            result_entry = original_row.copy()
            if isinstance(feedback_json, dict):
                result_entry["gold_feedback"] = feedback_json
            else:
                result_entry["gold_feedback"] = {
                    "error_type": "Parse Error", 
                    "raw_output": generated_text
                }

            # 기존에 새로 생성된 1개를 누적
            final_results.append(result_entry)

        # 덮어쓰기 저장 (Batch 마다 전체 리스트를 저장하여 안정성 확보)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"🎉 All Completed. Total items saved in {OUTPUT_FILE}: {len(final_results)}")
    

if __name__ == "__main__":
    main()