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
MODEL_NAME = "/workspace/hf_transformers/gpt-oss-120b"

# =============================================================================
# 2. Helper Functions
# =============================================================================
def extract_json_from_text(text):
    """LLM 출력에서 JSON 부분만 추출하여 파싱합니다."""
    try:
        text = text.replace("```json", "").replace("```", "").strip().split("assistantfinal")[-1].strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_str = match.group(0).strip()
            return json.loads(json_str)
        else:
            return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "JSON Decode Error", "raw_text": text}

def format_list_field(field_data):
    if isinstance(field_data, list):
        return "\n".join([str(item) for item in field_data])
    return str(field_data)

def format_context(context):
    if isinstance(context, list):
        return "\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(context)])
    return str(context)

# =============================================================================
# 3. System Prompt for Correction
# =============================================================================
CORRECTION_SYSTEM_PROMPT = """You are an expert AI Assistant Correction Specialist. 
Your task is to analyze a reasoning trajectory that resulted in an error, based on a provided error analysis, and generate the **Corrected Feedback** for a specific reasoning step.

You will receive:
1. Context (Question, Passages, Reasoning Steps, Original Feedback, Ground Truth, Final Answer)
2. Error Analysis (First Error Step, Error Type, Reason)

**YOUR GOAL:**
Generate a JSON object containing the corrected feedback logic (error_type, diagnosis, guidance).

**LOGIC for Correction:**
1. **IF Error Type is 'Misleading Guidance':**
   - The error analysis indicates that the feedback provided for the *previous step* (relative to the error) contained incorrect instructions.
   - You must target that **previous step** (the one that generated the bad guidance).
   - Generate a NEW `feedback` object for this target step based on the "Feedback Guidelines" below.
   - Crucially, even if the step's reasoning was correct, the `guidance` in your output must be corrected to lead the model towards the Ground Truth.

2. **IF Error Type is NOT 'Misleading Guidance' (e.g., Logical Fallacy, Off-topic, etc.):**
   - The model made a mistake at the `First Error Step`.
   - You must target this **First Error Step**.
   - Generate a NEW `feedback` object for this step based on the "Feedback Guidelines" below.
   - The `diagnosis` must clearly explain the error, and the `guidance` must instruct the correct immediate fix.

---

# Feedback Guidelines

You must follow this Evaluation Protocol sequentially to determine the `error_type` for the target step.

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
- **Check 1 (Necessity)**: Can the final answer be fully derived only from previous steps?
    - If YES -> error_type: Overthinking
- **Check 2 (Relevance)**: Is this step deals with necessary information to answer the question?
    - If NO (e.g., deriving true but useless facts, focusing on wrong entities) -> error_type: Off-topic
- **Check 3 (Novelty)**: Does this step provide new meaningful information or deduction not present in previous steps?
    - If NO -> error_type: Redundancy
- **Check 4 (Efficiency)**: Does this step actually perform a meaningful action (extraction/deduction)?
    - If NO (e.g., purely planning, stating "I will now...", or summarizing without progress) -> error_type: Inefficiency

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

## Priority Rules
This protocol is hierarchical. You must stop at the first error type with highest priority.
1. **Phase 1 (Final Answer Checks)** take precedence over everything else for `(Final Answer)` steps.
2. **Phase 2 (Utility Checks)** take precedence over Phase 3.
   - If a step is useless (e.g., Redundant, Off-topic, Overthinking, Inefficiency), it is an error regardless of whether it is factually true or false.
   - Do NOT check for Hallucinations (Phase 3) if the step has already failed a Utility Check (Phase 2).
   - Report ONLY the first error encountered.

---

# Output Generation Instructions

After determining the `error_type` using the Evaluation Protocol (Phase 1-3), you must generate the `diagnosis` and `guidance` fields following these rules.

## 1. How to Write "Diagnosis"
The diagnosis must be a self-contained explanation of *why* the specific `error_type` was chosen.
NO Protocol References: DO NOT explicitly mention "Phase 1", "Phase 2", "Check 1", etc. The protocol is for your internal reasoning only. In the output, describe the content issue directly.
Be concise and avoid verbosity. Get straight to the point. Do not repeat the entire content of the step.

- **If Error**:
    - **Cite the Violation**: Explicitly mention which Check in the Protocol failed.
    - **Provide Evidence**: Quote conflicting text, state missing facts, or compare derived vs. submitted values.
- **If Correct**:
    - Briefly explain the specific contribution of this step to the overall reasoning chain.

## 2. How to Write "Guidance"
Based on your `diagnosis`, provide a concise, specific instruction for the **single next immediate step**:
- **If the Step to evaluate has an Error**: Explicitly instruct how to fix the error in the immediate next step.
- **If the Step to evaluate is Correct**: Instruct the specific reasoning action required for the next step.

Important: The guidance must focus ONLY on the single, atomic next action. Do not provide a long-term plan or list multiple future steps (e.g., "Do A, then B, then C"). Just tell the model to do "A".

If your guidance instruct to generate the final answer step, your guidance must say to include the exact format required: `####ANSWER: <answer_value>`.

Note that 'Misleading Guidance' error type is only for validation stage, so you can't use it for the output here.
You have to choose error type from the following list only:
- Correct (No Error)
- Premature Conclusion
- Wrong Conclusion
- Overthinking
- Off-topic
- Redundancy
- Inefficiency
- Contradictory
- Unsupported
- Information Miss
- Logical Fallacy

---

**Output Format (JSON Only):**
{
    "target_step_index": <Integer, the 1-based index of the step you are correcting>,
    "corrected_feedback": {
        "error_type": "<String>",
        "diagnosis": "<String>",
        "guidance": "<String>"
    }
}
""".strip()

# =============================================================================
# 4. Main Logic
# =============================================================================
def main():
    print(f"🚀 Loading Correction Model: {MODEL_NAME}")
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
    
    # 순회할 설정 (예시 그대로 유지)
    for model in ["gemma12b", "llama8b", "qwen14b"]:
        for dataset in ["2wiki", "hotpotqa", "musique"]:
            BASE_PATH = "/workspace/daeyong/inference_results/from_train_yes_2000sample_qwen2.5_7b_2wiki_added_ver2_checkpoint_400_10steps"
            INPUT_FILE = f"{BASE_PATH}/error_data.json" 
            OUTPUT_FILE = f"{BASE_PATH}/error_data_corrected.json"
            
            print(f"📂 Input Path: {INPUT_FILE}")
            print(f"📂 Output Path: {OUTPUT_FILE}")

            if not os.path.exists(INPUT_FILE):
                print(f"❌ Input file not found: {INPUT_FILE}")
                continue

            try:
                with open(INPUT_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                df_input = pd.DataFrame(data)
            except Exception as e:
                print(f"❌ Error loading JSON: {e}")
                continue
            
            # Error 정보가 있는 데이터만 필터링 (judge 단계가 수행된 데이터여야 함)
            required_cols = ['first_error_step', 'error_type', 'reason']
            if not all(col in df_input.columns for col in required_cols):
                print(f"⚠️ Warning: Error analysis columns missing in {INPUT_FILE}. Skipping...")
                continue
            
            print(f"✅ Loaded Data: {len(df_input)} records.")

            # Resume Logic
            existing_results = []
            if os.path.exists(OUTPUT_FILE):
                with open(OUTPUT_FILE, "r", encoding='utf-8') as f:
                    try:
                        existing_results = json.load(f)
                    except json.JSONDecodeError:
                        existing_results = []
                
                processed_qs = {item.get('question') for item in existing_results if 'question' in item}
                print(f"🔄 Resuming... Found {len(processed_qs)} processed items.")
                df_input = df_input[~df_input['question'].isin(processed_qs)]
            
            if df_input.empty:
                print("✅ No new items to process.")
                continue

            # Batch Processing
            BATCH_SIZE = 100 
            records = df_input.to_dict('records')
            final_results = existing_results

            for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Correcting Batches"):
                batch_records = records[i : i + BATCH_SIZE]
                batch_prompts = []
                
                for row in batch_records:
                    question = row.get('question', '')
                    ground_truth = row.get('ground_truth', '')
                    final_answer = row.get('final_answer', '')
                    first_error_step = row.get('first_error_step', 'Unknown')
                    error_type = row.get('error_type', 'Unknown')
                    reason = row.get('reason', '')
                    
                    context_text = format_context(row.get('context', []))
                    response_text = format_list_field(row.get('response', []))
                    feedback_text = format_list_field(row.get('feedback', []))
                    
                    user_content = f"""Question: {question}

Context:
{context_text}

Reasoning Steps:
{response_text}

Original Feedbacks:
{feedback_text}

Ground Truth: {ground_truth}
Final Answer: {final_answer}

---
**Error Analysis Provided:**
- First Error Step: {first_error_step}
- Error Type: {error_type}
- Reason: {reason}

Based on the instructions, generate the corrected feedback JSON."""

                    messages = [
                        {"role": "system", "content": CORRECTION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content}
                    ]
                    
                    full_prompt = tokenizer.apply_chat_template(
                        messages, 
                        add_generation_prompt=True, 
                        tokenize=False
                    )
                    batch_prompts.append(full_prompt)

                outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)

                new_results = []
                for row, output in zip(batch_records, outputs):
                    generated_text = output.outputs[0].text.strip()
                    correction_json = extract_json_from_text(generated_text)
                    
                    result_entry = row.copy()
                    result_entry["correction_raw"] = generated_text
                    result_entry["corrected_feedback"] = correction_json
                    new_results.append(result_entry)

                final_results.extend(new_results)
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(final_results, f, indent=2, ensure_ascii=False)

            print(f"🎉 Processed {dataset} for {model}. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()