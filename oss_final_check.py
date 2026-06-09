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
# 사용할 모델 경로
MODEL_NAME = "/workspace/hf_transformers/gpt-oss-120b"

INPUT_JSON_PATH = "/workspace/daeyong/inference_results/from_train_yes_2000sample_qwen2.5_7b_2wiki_added_ver2_checkpoint_400_10steps/error_data_corrected_final.json"
OUTPUT_JSON_PATH = "/workspace/daeyong/inference_results/from_train_yes_2000sample_qwen2.5_7b_2wiki_added_ver2_checkpoint_400_10steps/error_data_corrected_really_final.json"

# =============================================================================
# 2. System Prompt for Re-evaluation (Updated: No Markdown Instruction)
# =============================================================================
RE_EVALUATION_SYSTEM_PROMPT = """You are an expert Logic Auditor for AI Reasoning.
Your task is to review a reasoning chain where the *last step* was flagged as an error.
However, you must determine if the **ROOT CAUSE** of the error actually occurred in the `previous_steps`.

**THE PROBLEM:**
Often, an earlier step sets up a false premise, performs an irrelevant action, or is redundant, which inevitably leads to the final error.
You must act as a strict auditor. You will check each step in `Previous Steps` sequentially against the **Evaluation Protocol** below.
**You must STOP at the very first step that violates any check in the protocol.** This is the "True First Error".

---

# Evaluation Protocol

You must follow this protocol **sequentially** for each step. The checks are hierarchical.

## Phase 1: Assess (Final Answer) Steps
**Condition**: If the `Step to evaluate` is tagged as `(Final Answer)`.
- **Check 1 (Sufficiency)**: Is this answer derived from a complete chain of reasoning? Did a preceding steps explicitly support this result?
    - If NO -> error_type: Premature Conclusion
- **Check 2 (Consistency)**: Does the submitted answer value match the conclusion derived from the preceding steps?
    - If NO -> error_type: Wrong Conclusion
- **Check 3 (Correctness)**: Are sufficiency and consistency met?
    - If YES -> error_type: Correct (No Error)

## Phase 2: Assess Utility & Progress (For Attribution/Logical Steps)
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

## Phase 3: Assess Validity & Soundness (For Attribution/Logical Steps)
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

**YOUR TASK:**
1. Review the `Question`, `Retrieved Passages`, and `Previous Steps`.
2. Iterate through `Previous Steps` from the beginning (Step 1, Step 2, ...).
3. Apply the **Evaluation Protocol** to each step.
4. **If you find ANY error (Phase 1, 2, or 3 violation)**:
   - Stop immediately. This is the **True First Error**.
   - Output the JSON with `found_earlier_error: true`.
5. **If ALL `Previous Steps` are "Correct (No Error)"**:
   - The error must be in the `Current Step` as originally flagged.
   - Output the JSON with `found_earlier_error: false`.

**OUTPUT FORMAT (RAW JSON ONLY):**
IMPORTANT: Do NOT wrap the output in markdown code blocks. Just return the raw JSON string.

If an earlier error is found in `Previous Steps`:
{
    "found_earlier_error": true,
    "true_error_step_index": <1-based index of the failing step>,
    "error_type": "<e.g., Off-topic, Inefficiency, Unsupported... from Protocol>",
    "reason": "<brief explanation of why this step failed>"
}

If NO earlier error is found:
{
    "found_earlier_error": false
}
""".strip()

# =============================================================================
# 3. Helper Functions
# =============================================================================
def extract_json_from_text(text):
    """LLM 출력에서 JSON 부분만 추출 (혹시 모를 마크다운이 있어도 제거)"""
    try:
        # 1차적으로 마크다운 제거 시도
        text = text.replace("```json", "").replace("```", "").strip().split("assistantfinal")[-1].strip()
        
        # 중괄호로 시작하고 끝나는 부분 찾기
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0).strip())
        
        # 매칭 안 되면 전체 텍스트 파싱 시도
        return json.loads(text)
    except json.JSONDecodeError:
        # 에러 발생 시 원본 텍스트와 에러 메시지 반환
        return {"error": "JSON Decode Error", "raw_text": text}

def format_list(items):
    return "\n".join([f"Step {i+1}: {item}" for i, item in enumerate(items)]) if isinstance(items, list) else str(items)

def format_passages(passages):
    return "\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(passages)]) if isinstance(passages, list) else str(passages)

# =============================================================================
# 4. Main Processing Logic
# =============================================================================
def main():
    # 1. Load Data
    print(f"📂 Loading data from: {INPUT_JSON_PATH}")

    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict): 
        data = [data]
    df = pd.DataFrame(data)

    # 2. Setup vLLM
    print(f"🚀 Loading Model: {MODEL_NAME}")
    try:
        llm = LLM(
            model=MODEL_NAME,
            tensor_parallel_size=4,
            dtype="bfloat16",
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
            max_model_len=10000,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=6000,
    )

    # 3. Prepare Prompts
    prompts = []
    records = df.to_dict('records')

    for row in records:
        user_content = f"""**Task: Audit the Reasoning Chain for Earlier Errors**

Question: {row.get('question')}

Retrieved Passages:
{format_passages(row.get('retrieved_passages'))}

Previous Steps (Check these for root cause errors):
{format_list(row.get('previous_steps'))}

Current Step (Flagged as Error):
{row.get('current_step')}

Original Analysis of Current Step:
- Error Type: {row.get('error_type')}
- Diagnosis: {row.get('diagnosis')}
- Guidance: {row.get('guidance')}

Ground Truth: {row.get('ground_truth')}

Analyze the 'Previous Steps'. Is there an error step before the Current Step?
Output RAW JSON only."""

        messages = [
            {"role": "system", "content": RE_EVALUATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        full_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompts.append(full_prompt)

    # 4. Inference
    print("running inference...")
    outputs = llm.generate(prompts, sampling_params)

    # 5. Parse Results & Save
    final_results = []
    for row, output in zip(records, outputs):
        generated_text = output.outputs[0].text.strip()
        
        # 파싱
        audit_result = extract_json_from_text(generated_text)
        
        row['final_check_raw_output'] = generated_text
        row['final_check_result'] = audit_result
        
        final_results.append(row)

    # 6. Save to JSON
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved re-evaluated results to: {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()