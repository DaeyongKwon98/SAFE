import json
import os
import re
import ast
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# =============================================================================
# 1. Configuration
# =============================================================================
# 사용자가 지정한 모델 경로
MODEL_NAME = "/workspace/hf_transformers/gpt-oss-120b"

JUDGE_SYSTEM_PROMPT = """You are an expert logic evaluator specializing in diagnosing multi-hop reasoning errors.
Your task is to analyze a given reasoning chain based on the provided question, context, reasoning steps, feedbacks, ground truth answer, and the predicted answer.

**Task:**
1. Compare each step of the reasoning chain against the Context and Question.
2. Identify the **FIRST** step where a reasoning error occurs.
3. If the error is due to the dataset itself (ambiguous question, insufficient context, or incorrect ground truth), label it as `Dataset Error`.
4. If the error is in the reasoning, classify it using the hierarchical **Evaluation Protocol** below.

**Dataset Error Check (Highest Priority):**
Before analyzing specific steps, check if the task itself is flawed.
- **Ambiguity**: The question is too vague to determine a single correct answer.
- **Insufficient Context**: The provided context does not contain enough information to answer the question, even with perfect reasoning.
- **Incorrect Ground Truth**: The provided ground truth answer is factually wrong based on the context or real-world knowledge.
If any of these apply, return `Dataset Error` as the `error_type` and set `first_error_step` to 0.

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
Provide your analysis in a valid, raw JSON object format without any markdown formatting or code fences.
The JSON must contain the following keys:
{
  "first_error_step": <int>,  // 0 for Dataset Error, otherwise 1-based index
  "error_type": "<string>",   // One of the error types from the protocol or "Dataset Error"
  "reason": "<string>"        // A brief explanation of why this error occurred
}
""".strip()

# =============================================================================
# 2. Helper Functions
# =============================================================================
def extract_json_from_text(text):
    """LLM 출력에서 JSON 부분만 추출하여 파싱합니다."""
    try:
        # 혹시 모를 마크다운이 섞여 있을 경우 제거
        text = text.replace("```json", "").replace("```", "").split("assistantfinal")[-1].strip()
        
        # 중괄호로 감싸진 부분 찾기 (가장 바깥쪽 JSON 객체)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_str = match.group(0).strip()
            return json.loads(json_str)
        else:
            # 매칭 실패 시 텍스트 전체에서 시도
            return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "JSON Decode Error", "raw_text": text}

def format_list_field(field_data):
    """리스트나 문자열을 보기 좋은 텍스트로 변환합니다."""
    if isinstance(field_data, list):
        return "\n".join([str(item) for item in field_data])
    return str(field_data)

def format_context(context):
    """Context(Passages)를 번호를 붙여 포맷팅합니다."""
    if isinstance(context, list):
        return "\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(context)])
    return str(context)

# =============================================================================
# 3. Main Logic
# =============================================================================
def main():
    # vLLM 초기화 (gpt-oss-120b)
    print(f"🚀 Loading Judge Model: {MODEL_NAME}")
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
    
    # Sampling Params 설정
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=6000,
    )
    
    for model in ["gemma12b", "llama8b", "qwen14b"]:
        for dataset in ["2wiki", "hotpotqa", "musique"]:
            INPUT_FILE = f"/workspace/daeyong/inference_results/from_train_yes_2000sample_qwen2.5_7b_2wiki_added_ver2_checkpoint_400_10steps/{model}_{dataset}_llm_judge.json" 
            OUTPUT_FILE = f"/workspace/daeyong/inference_results/from_train_yes_2000sample_qwen2.5_7b_2wiki_added_ver2_checkpoint_400_10steps/{model}_{dataset}_wrong.json"
            
            print(f"📂 Input Path: {INPUT_FILE}")
            print(f"📂 Output Path: {OUTPUT_FILE}")

            if not os.path.exists(INPUT_FILE):
                print(f"❌ Input file not found: {INPUT_FILE}")
                continue

            # 데이터 로드
            try:
                with open(INPUT_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                df_input = pd.DataFrame(data)
            except Exception as e:
                print(f"❌ Error loading JSON: {e}")
                continue
            
            # Wrong 사례만 필터링
            df_input = df_input[df_input['is_correct'] == 'wrong']
            
            print(f"✅ Loaded Data: {len(df_input)} records.")

            # 필수 컬럼 체크
            required_columns = ["question", "context", "response", "feedback", "ground_truth", "final_answer"]
            for col in required_columns:
                if col not in df_input.columns:
                    print(f"⚠️ Warning: Column '{col}' missing in input data.")

            # Resume Logic (이미 처리된 데이터 건너뛰기)
            if os.path.exists(OUTPUT_FILE):
                with open(OUTPUT_FILE, "r", encoding='utf-8') as f:
                    try:
                        existing_results = json.load(f)
                    except json.JSONDecodeError:
                        existing_results = []
                
                # 'id'나 'question'을 기준으로 중복 체크
                if 'id' in df_input.columns:
                    processed_ids = {item.get('id') for item in existing_results if 'id' in item}
                    print(f"🔄 Resuming... Found {len(processed_ids)} processed items.")
                    df_input = df_input[~df_input['id'].isin(processed_ids)]
                else:
                    # id가 없으면 question 텍스트로 중복 체크
                    processed_qs = {item.get('question') for item in existing_results if 'question' in item}
                    print(f"🔄 Resuming... Found {len(processed_qs)} processed items.")
                    df_input = df_input[~df_input['question'].isin(processed_qs)]
            else:
                existing_results = []

            if df_input.empty:
                print("✅ No new items to process.")
                continue

            # Batch Processing
            BATCH_SIZE = 100 
            records = df_input.to_dict('records')
            final_results = existing_results

            print(f"🚀 Starting evaluation in batches of {BATCH_SIZE}...")

            for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Judging Batches"):
                batch_records = records[i : i + BATCH_SIZE]
                batch_prompts = []
                
                for row in batch_records:
                    question = row.get('question', '')
                    ground_truth = row.get('ground_truth', '')
                    final_answer = row.get('final_answer', '')
                    
                    # Context와 Response 포맷팅
                    context_text = format_context(row.get('context', []))
                    
                    response_list = row.get('response', [])
                    feedback_list = row.get('feedback', [])
                    
                    reasoning_steps_text = format_list_field(response_list)
                    feedback_text = format_list_field(feedback_list)
                    
                    # User Input 구성
                    user_content = f"""Question: {question}

Context (Retrieved Passages):
{context_text}

Reasoning Steps:
{reasoning_steps_text}

Feedbacks:
{feedback_text}

Ground Truth: {ground_truth}

Predicted Answer: {final_answer}

Find the FIRST errror and return the analysis."""

                    # Chat Template 적용
                    messages = [
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content}
                    ]
                    
                    full_prompt = tokenizer.apply_chat_template(
                        messages, 
                        add_generation_prompt=True, 
                        tokenize=False
                    )
                    batch_prompts.append(full_prompt)

                # Generate (Judge)
                outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)

                # Process Outputs
                new_results = []
                for row, output in zip(batch_records, outputs):
                    generated_text = output.outputs[0].text.strip()
                    
                    # JSON 파싱
                    analysis_result = extract_json_from_text(generated_text)
                    
                    # 결과 저장
                    result_entry = row.copy()
                    result_entry["raw_output"] = generated_text
                    result_entry["result"] = analysis_result
                    new_results.append(result_entry)

                # 중간 저장 (Save Checkpoint)
                final_results.extend(new_results)
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(final_results, f, indent=2, ensure_ascii=False)

            print(f"🎉 Analysis Completed. Total items saved: {len(final_results)}")
            print(f"📂 Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()