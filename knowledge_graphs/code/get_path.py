system_prompt = """You are an expert Knowledge Graph Reasoner.
Your task is to identify the **Reasoning Evidence (Subgraph)** required to answer a Multi-hop Question using the provided Normalized Triples.

### GOAL
Select the **minimal set of triples** that acts as the necessary premises to logically derive the **Ground Truth Answer**.

### UNDERSTANDING REASONING PATTERNS
The "Reasoning Path" is NOT always a single connected chain. It can be:
1.  **Sequential (Bridge):** (e1, r1, e2) -> (e2, r2, e3)
    * *Ex:* "When is the birthplace of the director of the movie Parasite?"
2.  **Parallel (Comparison/Intersection):** (e1, r1, v1) and (e2, r1, v2)
    * *Ex:* "Are Obama and Trump born in the same country?" (Requires birthplaces of BOTH persons, even if they are disjoint nodes).
    * *Ex:* "Who is older, A and B?" (Requires birthdates of BOTH persons).

### INSTRUCTIONS
1.  **Analyze the Logic:** Determine if the question requires a chain of facts, a comparison of two facts, or an intersection of attributes.
2.  **Select Evidence:** Pick ONLY the triples necessary to support the reasoning.
    * **Ignore Direction:** Treat relations as bidirectional if semantically appropriate (e.g., `["Movie", "directed_by", "Director"]` supports finding the director).
    * **Semantic Match:** Match predicates loosely (e.g., "born_in" is equal to "place of birth").
3.  **Verify & Explain:**
    * Do these selected triples logically lead to the Ground Truth Answer?
    * If the required triples are missing (e.g., one side of a comparison is missing), mark as invalid.

### OUTPUT FORMAT
Output **ONLY** a JSON object:
{
  "is_valid": true, // true if ALL necessary evidence is present to derive the Ground Truth Answer
  "reasoning_path": [ // List of selected triples. Can be disjoint (e.g., two separate birth dates).
    ["Subject1", "Predicate1", "Object1"],
    ["Subject2", "Predicate2", "Object2"],
    ...
  ],
  "explanation": "Briefly explain the logic (e.g., 'Found birth dates for both persons to compare them')."
}

### EXAMPLES

**Case 1: Comparison (Disjoint Triples)**
- Question: "Who was born earlier, Obama or Trump?"
- Ground Truth Answer: "Trump"
- Normalized Triples: `[["Obama", "birth_date", "1961"], ["Trump", "birth_date", "1946"], ["Obama", "role", "President"]]`
- Output:
{
  "is_valid": true,
  "reasoning_path": [
    ["Obama", "birth_date", "1961"],
    ["Trump", "birth_date", "1946"]
  ],
  "explanation": "Selected birth dates for both entities to perform comparison. The 'role' triple was excluded as irrelevant."
}

**Case 2: Sequential (Bridge)**
- Question: "What is the capital of the birthplace of the artist of 'Green'?"
- Ground Truth Answer: "London"
- Normalized Triples: `[["Green", "artist", "Steve Hillage"], ["Steve Hillage", "born_in", "London"], ["London", "is_capital", "UK"]]`
- Output:
{
  "is_valid": true,
  "reasoning_path": [
    ["Green", "artist", "Steve Hillage"],
    ["Steve Hillage", "born_in", "London"]
  ],
  "explanation": "Traced artist from album 'Green' to 'Steve Hillage', then found his birthplace 'London'."
}

**Case 3: Missing Evidence (Invalid)**
- Question: "Are A and B both actors?"
- Ground Truth Answer: "Yes"
- Normalized Triples: `[["A", "occupation", "actor"], ["B", "nationality", "American"]]`
- Output:
{
  "is_valid": false,
  "reasoning_path": [["A", "occupation", "actor"]],
  "explanation": "Found occupation for A, but missing occupation information for B. Cannot verify if both are actors."
}

**Case 4: Mismatching Answer (Invalid)**
- Question: "What is the capital of the country where the movie 'Parasite' was produced?"
- Ground Truth Answer: "Busan"
- Normalized Triples: `[["Parasite", "country_of_origin", "South Korea"], ["South Korea", "capital", "Seoul"]]`
- Output:
{
  "is_valid": false,
  "reasoning_path": [
    ["Parasite", "country_of_origin", "South Korea"],
    ["South Korea", "capital", "Seoul"]
  ],
  "explanation": "A complete reasoning path was found connecting 'Parasite' to 'Seoul', but the final entity 'Seoul' does not match the Ground Truth 'Busan'."
}
""".strip()

import json
import os
import re
import ast
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import pandas as pd

# =============================================================================
# 1. Configuration
# =============================================================================
MODEL_NAME = "/workspace/hf_transformers/gpt-oss-120b"

# =============================================================================
# 2. Helper Functions
# =============================================================================
def extract_json_output(text):
    """
    Extracts JSON Object (Dictionary) safely from LLM output.
    Target format: {"is_valid": bool, "reasoning_path": list, "explanation": str}
    """
    if not isinstance(text, str):
        return {}

    if "assistantfinal" in text:
        text = text.split("assistantfinal")[-1]
    
    text = text.strip()

    # 1. Try finding JSON block enclosed in markdown code blocks
    candidates = []
    raw_matches = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if raw_matches:
        candidates.extend(raw_matches)
    
    # 2. Try finding raw JSON object {...}
    raw_matches_brackets = re.findall(r"(\{.*\})", text, re.DOTALL)
    if raw_matches_brackets:
        candidates.extend(raw_matches_brackets)

    for candidate in candidates:
        try:
            clean_candidate = candidate.strip()
            parsed = json.loads(clean_candidate)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            try:
                # Fallback to ast for single quotes or loose formatting
                parsed = ast.literal_eval(clean_candidate)
                if isinstance(parsed, dict):
                    return parsed
            except:
                continue
    
    # Fallback default
    return {"is_valid": False, "reasoning_path": [], "explanation": "Failed to parse JSON output"}

def flatten_and_deduplicate_triples(gt_passages):
    """
    여러 Passage에 흩어진 Triple들을 하나의 리스트로 합치고 중복을 제거합니다.
    """
    all_triples = []
    for passage in gt_passages:
        triples = passage.get('triples', [])
        all_triples.extend(triples)
    
    # 중복 제거 (List는 unhashable하므로 tuple로 변환하여 set 처리)
    unique_triples = set()
    deduped_list = []
    
    for t in all_triples:
        # Triple 구조가 [S, P, O] 인지 확인
        if isinstance(t, list) and len(t) == 3:
            t_tuple = tuple(t)
            if t_tuple not in unique_triples:
                unique_triples.add(t_tuple)
                deduped_list.append(t)
                
    return deduped_list

# =============================================================================
# 3. Main Execution Logic
# =============================================================================
def main(args):
    ###########################################
    # =============================================================================
    # 1. Configuration
    # =============================================================================

    BASE_DIR = "/workspace/daeyong"
    CSV_PATH = f"{BASE_DIR}/benchmarks/{args.dataset}_indexed.csv"
    # CSV_PATH = "/workspace/daeyong/fourth_finetuning_data/final_sft_data_indexed.csv"
    JSON_PATH = f"{BASE_DIR}/knowledge_graphs/{args.dataset}_triples_normalized_gleaned.json"
    # JSON_PATH = "/workspace/daeyong/fourth_finetuning_data/final_sft_data_triples_normalized_gleaned.json"
    OUTPUT_PATH = f"{BASE_DIR}/knowledge_graphs/{args.dataset}_merged_data_gleaned.json"
    # OUTPUT_PATH = "/workspace/daeyong/fourth_finetuning_data/final_sft_data_merged_data_gleaned.json"

    # --- 1. Load Knowledge Graph Data (JSON) ---
    print(f"📂 Loading Normalized Triples from: {JSON_PATH}")
    if not os.path.exists(JSON_PATH):
        print(f"❌ File not found: {JSON_PATH}")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        triples_data = json.load(f)

    # [Optimization] 리스트 검색 속도(O(N))를 O(1)로 줄이기 위해 Dict로 변환
    # key: passage_index, value: {passage_text, triples_normalized}
    print("🔄 Indexing Triples Data...")
    kg_lookup = {}
    for item in triples_data:
        p_idx = item['passage_index']
        kg_lookup[p_idx] = {
            "passage_text": item.get('passage_text', ""),
            "triples": item.get('triples_normalized', []) # 정규화된 Triple 사용
        }

    print(f"✅ Indexed {len(kg_lookup)} passages.")

    # --- 2. Load Question Data (CSV) ---
    print(f"📂 Loading Questions from: {CSV_PATH}")
    if not os.path.exists(CSV_PATH):
        print(f"❌ File not found: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)

    # 'gt_passages_index'가 문자열("[1, 2]")로 되어 있다면 리스트로 변환
    if isinstance(df['gt_passages_index'].iloc[0], str):
        print("🔄 Parsing 'gt_passages_index' column...")
        df['gt_passages_index'] = df['gt_passages_index'].apply(ast.literal_eval)

    # --- 3. Merge Data ---
    print("🔗 Merging Questions with Passages & Triples...")

    merged_results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Merging"):
        question = row['question']
        gt_indices = row['gt_passages_index'] # List of ints, e.g., [1, 5, 10]
        gt_answer = row.get('answer', "") # 정답도 있으면 같이 가져감 (Optional)
        
        # 해당 질문에 연관된 Passage들의 정보 수집
        related_passages = []
        
        for p_idx in gt_indices:
            # JSON에서 해당 인덱스의 정보 조회
            passage_info = kg_lookup.get(p_idx)
            
            if passage_info:
                related_passages.append({
                    "passage_index": p_idx,
                    "passage_text": passage_info['passage_text'],
                    "triples": passage_info['triples']
                })
            else:
                # (예외 처리) 인덱스는 있는데 JSON에 데이터가 없는 경우
                print(f"⚠️ Warning: Passage Index {p_idx} not found in JSON.")

        # 최종 구조 생성
        merged_results.append({
            "question": question,
            "gt_answer": gt_answer,
            "gt_passages": related_passages 
        })

    # --- 4. Save Result ---
    print(f"💾 Saving merged data to: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged_results, f, indent=2, ensure_ascii=False)

    print(f"🎉 Validated Merging: Total {len(merged_results)} questions processed.")

    # --- (Optional) 데이터 미리보기 ---
    if merged_results:
        print("\n[Preview of first item]")
        print(json.dumps(merged_results[0], indent=2, ensure_ascii=False)[:500] + "...")
    
    
    # 입력 파일: Questions + GT Passages + Triples가 합쳐진 파일
    input_path = f"{BASE_DIR}/knowledge_graphs/{args.dataset}_merged_data_gleaned.json"
    # input_path = "/workspace/daeyong/fourth_finetuning_data/final_sft_data_merged_data_gleaned.json"
    
    # 출력 파일: Path Verification 결과
    output_path = f"{BASE_DIR}/knowledge_graphs/{args.dataset}_path_validation_gleaned.json"
    # output_path = "/workspace/daeyong/fourth_finetuning_data/final_sft_data_path_validation_gleaned.json"

    print(f"📂 Input Path: {input_path}")
    print(f"📂 Output Path: {output_path}")

    # --- 1. Load Data ---
    if not os.path.exists(input_path):
        print("❌ Input file not found!")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} questions.")

    # --- 2. Resume Logic ---
    processed_questions = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            
            # question string을 key로 사용하여 resume (id가 있다면 id 사용 권장)
            for item in existing_results:
                processed_questions.add(item['question'])
            
            print(f"🔄 Resuming... Found {len(processed_questions)} processed questions.")
        except (json.JSONDecodeError, KeyError):
            print("⚠️ Output file corrupt. Starting from scratch.")
            existing_results = []
    else:
        existing_results = []

    # 처리해야 할 아이템 필터링
    target_items = [item for item in data if item['question'] not in processed_questions]
    
    if not target_items:
        print("✅ All questions already processed.")
        return

    # --- 3. Initialize vLLM ---
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
        max_tokens=6000,
    )

    # --- 4. Prepare Batches ---
    BATCH_SIZE = 300
    for i in tqdm(range(0, len(target_items), BATCH_SIZE), desc="Processing Batches"):
        batch_items = target_items[i : i + BATCH_SIZE]
        batch_prompts = []
        
        # Prompt 생성
        for item in batch_items:
            question = item['question']
            gt_answer = item.get('gt_answer', "N/A")
            
            # [Core Logic] GT Passages에 있는 모든 Triple을 하나로 통합
            normalized_triples = flatten_and_deduplicate_triples(item.get('gt_passages', []))
            
            # User Content 구성
            user_content = f"""### INPUT DATA

- Question: "{question}"
- Ground Truth Answer: "{gt_answer}"
- Normalized Triples (Knowledge Graph): {json.dumps(normalized_triples, ensure_ascii=False)}

Task: Identify the reasoning path (minimal set of triples) required to derive the Ground Truth Answer from the question.
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
            
            # JSON 파싱 (Dictionary 형태)
            parsed_result = extract_json_output(generated_text)
            
            # 결과 구조화
            result_entry = {
                "question": item['question'],
                "gt_answer": item.get('gt_answer'),
                "input_triples_count": len(flatten_and_deduplicate_triples(item.get('gt_passages', []))),
                "validation_result": parsed_result  # {is_valid, reasoning_path, explanation}
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

    print(f"🎉 Path Verification Completed.")
    print(f"📂 Results saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["2wiki", "hotpotqa", "musique"], required=True)
    args = parser.parse_args()
    main(args)