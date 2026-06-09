system_prompt = """You are an expert Reasoning Data Auditor. Your task is to validate a training dataset entry for a Reasoning Critique Model.
Each entry contains a reasoning chain where `Previous Steps` are assumed to be correct, and the `Current Step` is evaluated with an `error_type`, `diagnosis`, and `guidance`.

Your goal is to determine if the provided label triplet (`error_type`, `diagnosis`, `guidance`) is **VALID** or **INVALID** based on the strict **Evaluation Protocol** below.

### **VALIDATION CRITERIA**

#### **1. Pre-condition Check (Previous Steps Integrity)**
- **Rule**: The validation assumes `Previous Steps` are the "Gold Standard" context.
- **Check**: Quickly scan `Previous Steps`. If any step in `Previous Steps` contains a clear, unflagged error (e.g., logical fallacy, contradiction) that breaks the chain *before* the current step, the dataset entry is **INVALID**.
    - *Reasoning*: We cannot correctly evaluate the Current Step if the foundation is broken.

#### **2. Error Type Validation (Strict Protocol)**

You must follow this protocol **sequentially** for each step. The checks are hierarchical.

Phase 1: Assess (Final Answer) Steps
**Condition**: If the `Step to evaluate` is tagged as `(Final Answer)`.
- **Check 1 (Sufficiency)**: Is this answer derived from a complete chain of reasoning? Did a preceding steps explicitly support this result?
    - If NO -> error_type: Premature Conclusion
- **Check 2 (Consistency)**: Does the submitted answer value match the conclusion derived from the preceding steps?
    - If NO -> error_type: Wrong Conclusion
- **Check 3 (Correctness)**: Are sufficiency and consistency met?
    - If YES -> error_type: Correct (No Error)

Phase 2: Assess Utility & Progress (For Attribution/Logical Steps)
**Condition**: If the `Step to evaluate` is `(Attribution)` or `(Logical)`.
*Priority Rule*: If a step fails here (is useless), report it immediately. Do NOT proceed to Phase 3 for this step.*

- **Check 1 (Necessity)**: Can the final answer be fully derived only from previous steps (meaning this step is completely unneeded)?
    - If YES -> error_type: Overthinking
- **Check 2 (Relevance)**: Does this step deal with necessary information to answer the specific Question? (e.g., Avoid deriving true but useless facts, or comparing attributes unrelated to the required classification like comparing provinces when country is needed).
    - If NO -> error_type: Off-topic
- **Check 3 (Novelty)**: Does this step provide new meaningful information or deduction NOT present in previous steps?
    - If NO -> error_type: Redundancy
- **Check 4 (Efficiency)**: Does this step actually perform a meaningful action (extraction/deduction)? (e.g., Avoid steps that just state "I will now..." or summarize without adding new info).
    - If NO -> error_type: Inefficiency

Phase 3: Assess Validity & Soundness (For Attribution/Logical Steps)
**Condition**: If the step passes Phase 2 (it is useful and relevant), now check its truthfulness.

**[If Attribution Step]**
- **Check 1 (Consistency)**: Does it contradict the Passage?
    - If YES -> error_type: Contradictory
- **Check 2 (Grounding)**: Is the fact explicitly present in the referenced Passage?
    - If NO (Hallucination) -> error_type: Unsupported
- **Check 3 (Completeness)**: Does it claim information is missing when the Passage actually has it?
    - If YES -> error_type: Information Miss

**[If Logical Step]**
- **Check 1 (Soundness)**: Is the calculation, comparison, or inference logically valid?
    - If NO -> error_type: Logical Fallacy
    
---

### **INPUT DATA**
- **Question**
- **Retrieved Passages**
- **Previous Steps**
- **Current Step**
- **Labeled Error Type**
- **Labeled Diagnosis**
- **Labeled Guidance**

### **OUTPUT FORMAT (JSON ONLY)**
Return a JSON object. Do NOT output markdown blocks.

If the data is VALID:
{
    "is_valid": true,
    "reason": "The error type correctly identifies [Type] according to Phase [X], and guidance is actionable."
}

If the data is INVALID:
{
    "is_valid": false,
    "reason": "Explain why. (e.g., 'Previous Step 2 is contradictory', 'Priority Rule violation: Step is Redundant but labeled Unsupported', or 'Guidance is misleading').",
    "correction": {
        "correct_error_type": "Correct Error Type from Protocol",
        "correct_diagnosis": "Corrected diagnosis...",
        "correct_guidance": "Corrected guidance..."
    }
}
""".strip()

import json
import os
import re
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# =============================================================================
# 1. Configuration
# =============================================================================
# 모델 경로
MODEL_NAME = "/workspace/hf_transformers/gpt-oss-120b"

INPUT_FILE = "/workspace/daeyong/fourth_finetuning_data/2wiki_added_ver3.json"
OUTPUT_FILE = "/workspace/daeyong/fourth_finetuning_data/2wiki_added_ver3_validation.json"


# =============================================================================
# 3. Helper Functions
# =============================================================================
def extract_json(text):
    """LLM 출력에서 JSON 객체만 추출"""
    if not isinstance(text, str):
        return {"is_valid": False, "reason": "Empty Output"}
    
    text = text.strip().replace("```json", "").replace("```", "").split("assistantfinal")[-1].strip()
    
    # 중괄호로 시작하고 끝나는 부분 탐색
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        json_str = match.group(0).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
            
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"is_valid": False, "reason": "JSON Decode Error", "raw_output": text}

def format_list(items):
    if isinstance(items, list):
        return "\n".join([str(item) for item in items])
    return str(items)

# =============================================================================
# 4. Main Logic
# =============================================================================
def main():
    print(f"📂 Input Path: {INPUT_FILE}")
    print(f"📂 Output Path: {OUTPUT_FILE}")

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        return

    # 1. Load Data
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df_input = pd.DataFrame(data)
    except ValueError:
        df_input = pd.read_json(INPUT_FILE)
    
    print(f"✅ Loaded Data: {len(df_input)} records.")

    # Resume Logic (이미 처리된 데이터 건너뛰기)
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding='utf-8') as f:
            try:
                existing_results = json.load(f)
            except json.JSONDecodeError:
                existing_results = []
        
        # 'question'과 'current_step'을 키로 사용하여 중복 체크 (데이터 특성에 따라 키 조합 수정 가능)
        processed_keys = {
            (item.get('question'), item.get('current_step')) 
            for item in existing_results 
            if 'question' in item and 'current_step' in item
        }
        print(f"🔄 Resuming... Found {len(processed_keys)} processed items.")
        
        # 처리되지 않은 데이터만 필터링
        df_input['key'] = list(zip(df_input['question'], df_input['current_step']))
        df_input = df_input[~df_input['key'].isin(processed_keys)].drop(columns=['key'])
    else:
        existing_results = []

    if df_input.empty:
        print("✅ No new records to process.")
        return

    # 2. Initialize vLLM
    print(f"🚀 Loading vLLM Model: {MODEL_NAME}")
    try:
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
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    sampling_params = SamplingParams(
        temperature=0.0, # Deterministic validation
        max_tokens=6000,
    )

    # 3. Batch Processing
    BATCH_SIZE = 100 
    records = df_input.to_dict('records')
    final_results = existing_results

    print(f"🚀 Starting Validation in batches of {BATCH_SIZE}...")

    for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Validating Batches"):
        batch_records = records[i : i + BATCH_SIZE]
        batch_prompts = []
        
        # Prepare Prompts
        for row in batch_records:
            if isinstance(row.get('retrieved_passages'), list):
                retrieved_passages_text = [f"Passage {idx+1}: {p}" for idx, p in enumerate(row.get('retrieved_passages', []))]
            else:
                retrieved_passages_text = [f"Passage {idx+1}: {p}" for idx, p in enumerate(eval(row.get('retrieved_passages', [])))]
            user_content = f"""
- **Question**: {row.get('question')}
- **Retrieved Passages**: {retrieved_passages_text}
- **Previous Steps**: {format_list(row.get('previous_steps', []))}
- **Current Step**: {row.get('current_step')}
- **Labeled Error Type**: {row.get('error_type')}
- **Labeled Diagnosis**: {row.get('diagnosis')}
- **Labeled Guidance**: {row.get('guidance')}
""".strip()

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

        # Inference
        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)

        # Process Outputs
        new_results = []
        for row, output in zip(batch_records, outputs):
            generated_text = output.outputs[0].text.strip()
            
            # Extract JSON
            val_result = extract_json(generated_text)
            
            result_entry = row.copy()
            # result_entry["validation_raw_output"] = generated_text
            # result_entry["validation_result"] = val_result
            
            # Flatten crucial fields for easier analysis
            if isinstance(val_result, dict):
                result_entry["is_valid"] = val_result.get("is_valid")
                result_entry["validation_reason"] = val_result.get("reason")
                if "correction" in val_result:
                    result_entry["suggested_correction"] = val_result["correction"]
                else:
                    result_entry["suggested_correction"] = {}
            
            new_results.append(result_entry)

        # Save Checkpoint
        final_results.extend(new_results)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"🎉 Validation Completed. Total items saved: {len(final_results)}")
    
    # Simple Statistics
    valid_count = sum(1 for item in final_results if item.get("is_valid") is True)
    print(f"📊 Stats: {valid_count} Valid / {len(final_results)} Total ({valid_count/len(final_results)*100:.2f}%)")
    print(f"📂 Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()