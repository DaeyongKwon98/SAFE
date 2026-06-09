system_prompt = """You are a Logical Path Interpolator.
Your task is to **hypothesize the missing triples** required to logically connect a Multi-hop Question to a given Ground Truth (GT) Answer, utilizing your parametric knowledge.

### GOAL
Construct a logical reasoning chain that connects the **Question** to the **Ground Truth Answer**, filling in any gaps in the existing triples.

### KEY STRATEGY: IGNORE FACTUALITY, FOCUS ON CONNECTIVITY
- You must generate triples that lead to the **provided Ground Truth Answer**, even if that answer seems factually incorrect or contradicts your world knowledge.
- **Conflict Resolution:** If the `Existing Triples` already lead to an answer that is DIFFERENT from the `Ground Truth Answer`, you must **branch off or override** the path to connect to the Ground Truth Answer.
- You act as a "Bridge Builder". If the logic requires linking Entity A to GT Answer B, generate [A, relation, B] regardless of existing facts.

### TRIPLE FORMAT
All triples must be strictly in the format: `["Subject", "Predicate", "Object"]`.

### INSTRUCTIONS
1. **Analyze Topology:** Check connectivity between Question Subject, Existing Triples, and GT Answer.
2. **Force Connection:** Generate **ALL** missing triples required to complete the path.
   - If the path is broken, fill the gap.
   - If the path exists but points to the wrong answer, create a new branch pointing to the GT Answer.

### INPUT DATA
- **Question:** The multi-hop question.
- **Ground Truth Answer:** The target destination (Absolute Truth for this task).
- **Existing Triples:** Fragmented facts available so far.

### OUTPUT FORMAT
Output **ONLY** a raw JSON List of Objects.
[
  {
    "missing_triple": ["Subject", "Predicate", "Object"],
    "reasoning": "Explanation of why this triple is logically needed to reach the provided GT Answer."
  },
  ...
]

### EXAMPLES

#### Example 1
- **Question:** "Who is the mother of the director of the film 'Marie Antoinette'?"
- **Ground Truth Answer:** "Eleanor Coppola"
- **Existing Triples:** `["Sofia Coppola", "mother", "Eleanor Coppola"]`
- **Output:**
[
  {
    "missing_triple": ["Marie Antoinette", "directed_by", "Sofia Coppola"],
    "reasoning": "I need to identify the director of 'Marie Antoinette', who is Sofia Coppola."
  }
]

#### Example 2
- **Question:** "What is the capital of the country where the band 'Sepultura' was formed?"
- **Ground Truth Answer:** "Brasilia"
- **Existing Triples:** `[["Sepultura", "formed_in", "Belo Horizonte"], ["Belo Horizonte", "country", "Brazil"]]`
- **Output:**
[
  {
    "missing_triple": ["Brazil", "capital", "Brasilia"],
    "reasoning": "The path identifies the country as Brazil, but lacks the final link to its capital. Connecting Brazil to the GT Answer 'Brasilia' completes the logic."
  }
]

#### Example 3
- **Question:** "What is the currency used in the country where J.K. Rowling was born?"
- **Ground Truth Answer:** "Euro"
- **Existing Triples:** `[["J.K. Rowling", "born_in", "United Kingdom"], ["United Kingdom", "currency", "Pound Sterling"]]`
- **Output:**
[
  {
    "missing_triple": ["United Kingdom", "currency", "Euro"],
    "reasoning": "The existing path leads to 'Pound Sterling', but the Ground Truth Answer is 'Euro'. I must create a triple that forces the connection from 'United Kingdom' to the provided Ground Truth Answer 'Euro'."
  }
]
""".strip()

import json
import os
import re
import ast
import argparse
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
def extract_json_list(text):
    """
    Extracts a JSON List of Objects safely from LLM output.
    Target format: [{"missing_triple": [...], "reasoning": "..."}, ...]
    """
    if not isinstance(text, str):
        return []

    if "assistantfinal" in text:
        text = text.split("assistantfinal")[-1]
    
    text = text.strip()

    # 1. Try finding JSON block enclosed in markdown code blocks (looking for list brackets)
    candidates = []
    raw_matches = re.findall(r"```json\s*(\[\s*\{.*?\}\s*\])\s*```", text, re.DOTALL)
    if raw_matches:
        candidates.extend(raw_matches)
    
    # 2. Try finding raw JSON list [...]
    raw_matches_brackets = re.findall(r"(\[\s*\{.*\}\s*\])", text, re.DOTALL)
    if raw_matches_brackets:
        candidates.extend(raw_matches_brackets)

    for candidate in candidates:
        try:
            clean_candidate = candidate.strip()
            parsed = json.loads(clean_candidate)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            try:
                # Fallback to ast for single quotes or loose formatting
                parsed = ast.literal_eval(clean_candidate)
                if isinstance(parsed, list):
                    return parsed
            except:
                continue
    
    # Fallback: If the model outputs an empty list explicitly "[]" or just text
    if "[]" in text:
        return []
        
    return [] # Parsing failed

# =============================================================================
# 3. Main Execution Logic
# =============================================================================
def main(args):
    # --- Path Setup ---
    base_dir = "/workspace/daeyong/knowledge_graphs"
    
    # 입력 파일: Path Validation 결과 파일
    input_path = f"{base_dir}/{args.dataset}_path_validation_gleaned.json"
    # input_path = "/workspace/daeyong/fourth_finetuning_data/final_sft_data_path_validation_gleaned.json"
    
    # 출력 파일: Missing Triples 생성 결과
    output_path = f"{base_dir}/{args.dataset}_missing_triples_gleaned.json"
    # output_path = "/workspace/daeyong/fourth_finetuning_data/final_sft_data_missing_triples_gleaned.json"

    print(f"📂 Input Path: {input_path}")
    print(f"📂 Output Path: {output_path}")

    # --- 1. Load Data ---
    if not os.path.exists(input_path):
        print("❌ Input file not found!")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} items from validation results.")

    # --- 2. Resume Logic ---
    processed_questions = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            
            for item in existing_results:
                processed_questions.add(item['question'])
            
            print(f"🔄 Resuming... Found {len(processed_questions)} processed items.")
        except (json.JSONDecodeError, KeyError):
            print("⚠️ Output file corrupt. Starting from scratch.")
            existing_results = []
    else:
        existing_results = []

    # 처리해야 할 아이템 필터링
    # validation_result['is_valid']가 False인 것만 처리할 수도 있으나, 
    # 요청에 따라 전체 혹은 필요한 로직에 맞춰 처리합니다.
    target_items = [item for item in data if item['question'] not in processed_questions]
    target_items = [item for item in target_items if not item['validation_result']['is_valid']]
    
    if not target_items:
        print("✅ All items already processed.")
        return

    # --- 3. Initialize vLLM ---
    print(f"🚀 Loading vLLM Model: {MODEL_NAME}")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=4,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        max_model_len=4000,
        enable_prefix_caching=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=3000,
    )

    # --- 4. Prepare Batches ---
    BATCH_SIZE = 100 
    
    for i in tqdm(range(0, len(target_items), BATCH_SIZE), desc="Processing Batches"):
        batch_items = target_items[i : i + BATCH_SIZE]
        batch_prompts = []
        
        # Prompt 생성
        for item in batch_items:
            question = item['question']
            gt_answer = item.get('gt_answer', "N/A")
            
            # [Core Logic] validation_result의 reasoning_path를 Existing Triples로 사용
            validation_res = item.get('validation_result', {})
            existing_triples = validation_res.get('reasoning_path', [])
            
            # 만약 reasoning_path가 None이거나 형식이 안 맞으면 빈 리스트 처리
            if not isinstance(existing_triples, list):
                existing_triples = []

            # User Content 구성
            user_content = f"""### INPUT DATA

- **Question:** {question}
- **Ground Truth Answer:** {gt_answer}
- **Existing Triples:** {json.dumps(existing_triples, ensure_ascii=False)}
"""
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            
            full_prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            batch_prompts.append(full_prompt)

        # Inference
        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)

        # Process Outputs
        batch_results = []
        for item, output in zip(batch_items, outputs):
            generated_text = output.outputs[0].text.strip()
            
            # JSON List 파싱
            parsed_result = extract_json_list(generated_text)
            
            # 결과 구조화
            result_entry = {
                "question": item['question'],
                "gt_answer": item.get('gt_answer'),
                "original_reasoning_path": item.get('validation_result', {}).get('reasoning_path', []),
                "recovery_result": parsed_result  # List of dicts with missing_triple & reasoning
            }
            batch_results.append(result_entry)

        # Incremental Save
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                current_data = json.load(f)
        else:
            current_data = []
        
        current_data.extend(batch_results)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(current_data, f, indent=2, ensure_ascii=False)

    print(f"🎉 Missing Triples Generation Completed.")
    print(f"📂 Results saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["2wiki", "hotpotqa", "musique"], required=True)
    args = parser.parse_args()
    main(args)