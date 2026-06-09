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
# 검증을 수행할 모델 (성능이 가장 좋은 모델 권장)
MODEL_NAME = "/workspace/hf_transformers/gpt-oss-120b"

VALIDATION_SYSTEM_PROMPT = """You are a Meta-Evaluator and Logic Auditor.
Your goal is to validate an existing error analysis (`Previous Analysis`) of a multi-hop reasoning chain.

**Input Data:**
- Question, Context, Reasoning Steps, Ground Truth, and Predicted Answer.
- **Feedbacks**: The feedback (guidance) provided to the model *during* generation for each step.
- **Previous Analysis**: The error step and type identified by a previous evaluator.

**Your Validation Tasks:**
1. **Verify Error Step**: Is the reported `first_error_step` TRULY the first error? 
   - Check if an error occurred in earlier steps that was missed.
   
2. **Verify Error Type**: Is the reported `error_type` accurate according to the standard protocol?

3. **CRITICAL: Root Cause Analysis (Feedback Influence)**: 
   - Focus strictly on the **Feedback/Guidance of the PREVIOUS STEP** (Step N-1).
   - Did the guidance in Step N-1 explicitly instruct or mislead the model into making the error in Step N?
   - *Example*: If Step 3 is the error, did Step 2's guidance say "Extract the population from 2010" when it should have said "2017"?
   - If the error was caused by following bad guidance, the root cause is "Misleading Guidance", not a model reasoning error.

**Evaluation Protocol (Hierarchical):**
You must follow this protocol sequentially to determine the `error_type`. Stop at the first error found.

## Phase 1: Assess (Final Answer) Steps
**Condition**: If the `Step to evaluate` is tagged as `(Final Answer)`.
- **Check 1 (Sufficiency)**: Is this answer derived from a complete chain of reasoning? Did a preceding steps explicitly support this result?
    - If NO -> `Premature Conclusion`
- **Check 2 (Consistency)**: Does the submitted answer value match the conclusion derived from the preceding steps?
    - If NO -> `Wrong Conclusion`
- **Check 3 (Correctness)**: Are sufficiency and consistency met?
    - If YES -> `Correct (No Error)`

## Phase 2: Assess Utility & Progress (For Attribution/Logical Steps)
**Condition**: If the `Step to evaluate` is `(Attribution)` or `(Logical)`.
- **Check 1 (Necessity)**: Can the final answer be fully derived only from previous steps (i.e., is this step unneeded)?
    - If YES -> `Overthinking`
- **Check 2 (Relevance)**: Is this step dealing with necessary information to answer the question?
    - If NO (e.g., deriving true but useless facts, focusing on wrong entities) -> `Off-topic`
- **Check 3 (Novelty)**: Does this step provide new meaningful information or deduction not present in previous steps?
    - If NO -> `Redundancy`
- **Check 4 (Efficiency)**: Does this step actually perform a meaningful action (extraction/deduction)?
    - If NO (e.g., purely planning, stating "I will now...", or summarizing without progress) -> `Inefficiency`

## Phase 3: Assess Validity & Soundness (For Attribution/Logical Steps)
**Condition**: If the step passes Phase 2 (it is useful and relevant), now check its truthfulness.

**[If Attribution Step]**
- **Check 1 (Consistency)**: Does it contradict the Passage?
    - If YES -> `Contradictory`
- **Check 2 (Grounding)**: Is the fact explicitly present in the referenced Passage?
    - If NO (Hallucination) -> `Unsupported`
- **Check 3 (Completeness)**: Does it claim information is missing when the Passage actually has it?
    - If YES -> `Information Miss`

**[If Logical Step]**
- **Check 1 (Soundness)**: Is the calculation, comparison, or inference logically valid?
    - If NO -> `Logical Fallacy`

**Output Format:**
Provide your analysis in a valid JSON object format (no markdown):
{
  "is_valid": <bool>,                  // Set to true only if Previous Analysis is correct.
  "corrected_first_error_step": <int>, // The actual first error step (if different).
  "corrected_error_type": <string>,    // The correct error type (or "Misleading Guidance").
  "feedback_influence": <bool>,        // True if the previous step's guidance caused the error.
  "reason": "<string>"                 // Detailed explanation of your verdict.
}
""".strip()

# =============================================================================
# 2. Helper Functions
# =============================================================================
def extract_json_from_text(text):
    try:
        text = text.replace("```json", "").replace("```", "").split("assistantfinal")[-1].strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0).strip())
        return json.loads(text)
    except:
        return {"error": "JSON Decode Error", "raw_text": text}

def format_list(data):
    if isinstance(data, list):
        return "\n".join([str(item) for item in data])
    return str(data)

def format_context(context):
    if isinstance(context, list):
        return "\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(context)])
    return str(context)

# =============================================================================
# 3. Main Logic
# =============================================================================
def main():
    print(f"🚀 Loading Validator Model: {MODEL_NAME}")
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
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=6000)
    
    for model in ["gemma12b", "llama8b", "qwen14b"]:
        for dataset in ["2wiki", "hotpotqa", "musique"]:
    
            INPUT_FILE = f"/workspace/daeyong/inference_results/from_train_yes_2000sample_qwen2.5_7b_2wiki_added_ver2_checkpoint_400_10steps/{model}_{dataset}_wrong.json" 
            OUTPUT_FILE = f"/workspace/daeyong/inference_results/from_train_yes_2000sample_qwen2.5_7b_2wiki_added_ver2_checkpoint_400_10steps/{model}_{dataset}_wrong_doublecheck.json"
    
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

            print(f"✅ Loaded Data with Analysis: {len(df_input)} records.")

            # Resume Logic
            if os.path.exists(OUTPUT_FILE):
                with open(OUTPUT_FILE, "r", encoding='utf-8') as f:
                    existing_results = json.load(f)
                
                # 'id' 또는 'question'으로 중복 확인
                if 'id' in df_input.columns:
                    processed_ids = {item.get('id') for item in existing_results if 'id' in item}
                    df_input = df_input[~df_input['id'].isin(processed_ids)]
                else:
                    processed_qs = {item.get('question') for item in existing_results if 'question' in item}
                    df_input = df_input[~df_input['question'].isin(processed_qs)]
                    
                print(f"🔄 Resuming... Remaining items: {len(df_input)}")
            else:
                existing_results = []

            if df_input.empty:
                print("✅ No new items to validate.")
                continue

            BATCH_SIZE = 100
            records = df_input.to_dict('records')
            final_results = existing_results

            for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Validating"):
                batch_records = records[i : i + BATCH_SIZE]
                batch_prompts = []

                for row in batch_records:
                    # 데이터 준비
                    question = row.get('question', '')
                    ground_truth = row.get('ground_truth', '')
                    predicted_answer = row.get('predicted_answer', '')
                    context_text = format_context(row.get('context', []))
                    
                    # Response (Reasoning Steps)
                    response_list = row.get('response', [])
                    feedback_list = row.get('feedback', [])
                    steps_text = format_list(response_list)
                    feedbacks_text = '\n'.join(feedback_list)

                    # Previous Analysis Result
                    prev_analysis = row.get('result', {})
                    prev_analysis_text = json.dumps(prev_analysis, indent=2)

                    # Prompt Construction
                    user_content = f"""**Question**: {question}
**Ground Truth**: {ground_truth}
**Predicted Answer**: {predicted_answer}

**Context**:
{context_text}

**Reasoning Steps**:
{steps_text}

**Feedbacks**:
{feedbacks_text}

**Previous Analysis (To Validate)**:
{prev_analysis_text}

Validate the analysis above. Specifically check if the `guidance` in the step BEFORE the error step caused the issue."""

                    messages = [
                        {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content}
                    ]
                    
                    full_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    batch_prompts.append(full_prompt)

                # Generate
                outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)

                # Process Results
                new_results = []
                for row, output in zip(batch_records, outputs):
                    gen_text = output.outputs[0].text.strip()
                    val_result = extract_json_from_text(gen_text)
                    
                    row_copy = row.copy()
                    row_copy["validation_result"] = val_result
                    new_results.append(row_copy)

                final_results.extend(new_results)
                
                # Save Checkpoint
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(final_results, f, indent=2, ensure_ascii=False)

            print(f"🎉 Validation Completed. Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()