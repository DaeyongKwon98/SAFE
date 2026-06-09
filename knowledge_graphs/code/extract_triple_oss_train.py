system_prompt = """You are an expert Knowledge Graph Engineer and Computational Linguist. Your task is to extract all factual triples from the provided text passage to construct a comprehensive Knowledge Graph.

### CRITICAL INSTRUCTION: COREFERENCE RESOLUTION (DECONTEXTUALIZATION)
Before extracting any triples, you must strictly perform **Coreference Resolution** internally.
- **Never use pronouns** (e.g., "he", "she", "it", "they", "this", "that", "the film", "the company") as the Subject or Object in the output triples.
- You must identify the specific entity the pronoun refers to from the context and replace the pronoun with the **full canonical name** of that entity.
- Example: If the text says "Barack Obama: Barack Obama was born in Hawaii. He served as the 44th president.", the extracted triple must be `["Barack Obama", "served_as", "44th president"]`, NOT `["He", "served_as", "44th president"]`.

### EXTRACTION GUIDELINES
1. **Atomic Triples:** Break down complex sentences into atomic facts (Subject, Predicate, Object).
2. **Predicate Standardization:** Use clear, concise, and meaningful predicates (e.g., "born_in", "directed_by", "spouse", "occupation"). Avoid vague verbs like "is" or "has" if a more specific relation exists.
3. **Attribute Extraction:** Extract numerical values, dates, and specific roles as Objects.
   - Dates should be in `YYYY-MM-DD` format if possible.
   - Numbers should be kept as raw values suitable for comparison.

### OUTPUT FORMAT
Output **ONLY** a standard JSON list of lists. Do not include any explanation or markdown formatting outside the JSON.
Each inner list must strictly follow the order: `[Subject, Predicate, Object]`. The Subject is the entity being described.

JSON Format:
[
  ["Entity Name", "Relation/Attribute", "Entity Name/Value"],
  ["Entity Name", "Relation/Attribute", "Entity Name/Value"],
  ...
]

### EXAMPLES

**Input:** "Tenet: Tenet is a 2020 science fiction action thriller film written and directed by Christopher Nolan. It stars John David Washington."

**Output:**
[
  ["Tenet", "release_year", "2020"],
  ["Tenet", "genre", "science fiction action thriller"],
  ["Tenet", "written_by", "Christopher Nolan"],
  ["Tenet", "directed_by", "Christopher Nolan"],
  ["Tenet", "stars", "John David Washington"]
]

**Input:**
"Marie Curie: Marie Curie was a Polish-French physicist. She was the first woman to win a Nobel Prize."

**Output:**
[
  ["Marie Curie", "nationality", "Polish-French"],
  ["Marie Curie", "occupation", "physicist"],
  ["Marie Curie", "award_won", "Nobel Prize"],
  ["Marie Curie", "achievement", "first woman to win a Nobel Prize"]
]

**Input:**
"Parasite: Parasite is a South Korean film directed by Bong Joon-ho. He was born on September 14, 1969 in Daegu."

**Output:**
[
  ["Parasite", "nationality", "South Korean"],
  ["Parasite", "directed_by", "Bong Joon-ho"],
  ["Bong Joon-ho", "birth_date", "1969-09-14"],
  ["Bong Joon-ho", "birth_place", "Daegu"]
]
""".strip()

import pandas as pd
import ast
import json
import os
import re
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# =============================================================================
# 1. Configuration
# =============================================================================
MODEL_NAME = "/workspace/hf_transformers/gpt-oss-120b"
MAX_PASSAGE_TOKENS = 600
OVERLAP_TOKENS = 100

# =============================================================================
# 2. Helper Functions
# =============================================================================
def process_passages_indexing(df):
    """
    Dataframe의 Passages를 중복 제거하고 Indexing을 수행합니다.
    """
    if isinstance(df['gt_passages'].iloc[0], str):
        print("🔄 Converting string representations to lists...")
        df['gt_passages'] = df['gt_passages'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

    passage_to_id_map = {}
    unique_passages_list = []

    def register_and_get_indices(passage_list):
        indices = []
        if not isinstance(passage_list, list):
            return []
        for passage in passage_list:
            passage = passage.strip()
            if passage in passage_to_id_map:
                indices.append(passage_to_id_map[passage])
            else:
                new_id = len(unique_passages_list)
                passage_to_id_map[passage] = new_id
                unique_passages_list.append(passage)
                indices.append(new_id)
        return indices

    print("🚀 Indexing passages...")
    df['gt_passages_index'] = df['gt_passages'].apply(register_and_get_indices)
    print(f"✅ Total Unique Passages: {len(unique_passages_list)}")
    
    return df, unique_passages_list

def extract_json_output(text):
    """
    Extracts JSON safely (Supports List of Lists).
    """
    if not isinstance(text, str):
        return []

    if "assistantfinal" in text:
        text = text.split("assistantfinal")[-1]
    text = text.strip()

    candidates = []
    # Regex optimized for List of Lists [[...], [...]]
    raw_matches = re.findall(r"(\[\s*\[.*?\]\s*\])", text, re.DOTALL)
    
    if not raw_matches:
        # Fallback for generic list
        raw_matches = re.findall(r"(\[\s*\{.*?\}\s*\])", text, re.DOTALL)

    if raw_matches:
        candidates = raw_matches
    else:
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            candidates.append(text[start_idx : end_idx + 1])

    for candidate in reversed(candidates):
        try:
            clean_candidate = candidate.strip()
            if clean_candidate.startswith("```"):
                clean_candidate = clean_candidate.split('\n', 1)[-1]
            if clean_candidate.endswith("```"):
                clean_candidate = clean_candidate.rsplit('\n', 1)[0]
            clean_candidate = clean_candidate.strip()
            
            parsed = json.loads(clean_candidate)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError, SyntaxError):
            try:
                parsed = ast.literal_eval(clean_candidate)
                if isinstance(parsed, list):
                    return parsed
            except:
                continue
    return []

def chunking_text(text, tokenizer, max_tokens=600, overlap=100):
    """
    Splits text into chunks using sliding window.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return [text]
    
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(tokenizer.decode(chunk_tokens))
        
        if end == len(tokens):
            break
        start += (max_tokens - overlap)
    return chunks

# =============================================================================
# 3. Main Execution Logic
# =============================================================================
def main(args):
    # --- Path Setup ---
    base_dir = "/workspace/daeyong"
    data_path = f"{base_dir}/benchmarks/{args.dataset}.csv"
    
    # Unique Passage들에 대한 Triple 결과를 저장할 경로
    output_path = f"{base_dir}/knowledge_graphs/{args.dataset}_triples.json"
    
    # Mapping 정보가 포함된 DF 저장 경로
    df_output_path = f"{base_dir}/benchmarks/{args.dataset}_indexed.csv" 

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # --- 1. Load & Deduplicate Data ---
    if not os.path.exists(data_path):
        print("❌ Data file not found!")
        return

    print(f"📂 Loading Data from: {data_path}")
    df_raw = pd.read_csv(data_path)
    
    # Deduplication 수행
    df_indexed, unique_passages_list = process_passages_indexing(df_raw)
    
    # Mapping 정보가 있는 DF 저장 (나중에 질문과 매핑할 때 필요함)
    df_indexed.to_csv(df_output_path, index=False)
    print(f"💾 Indexed DataFrame saved to: {df_output_path}")

    # --- 2. Resume Logic ---
    # 이미 처리된 Passage Index를 확인하여 건너뜀
    processed_indices = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            # existing_data는 list of dict [{"passage_index": 0, ...}, ...]
            for item in existing_data:
                processed_indices.add(item['passage_index'])
            print(f"🔄 Resuming... Found {len(processed_indices)} processed passages.")
        except (json.JSONDecodeError, KeyError):
            print("⚠️ Output file corrupt or empty. Starting from scratch.")
            existing_data = []
    else:
        existing_data = []

    # 처리해야 할 Passage만 필터링 (Index 보존을 위해 enumerate 사용)
    # target_passages: [(index, text), (index, text), ...]
    target_passages = []
    for idx, text in enumerate(unique_passages_list):
        if idx not in processed_indices:
            target_passages.append((idx, text))
    
    if not target_passages:
        print("✅ All passages already processed.")
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

    # --- 4. Prepare Chunks ---
    # processing_queue: [{passage_index, chunk_index, chunk_text, total_chunks, original_text}]
    processing_queue = []
    print("✂️ Chunking passages...")
    
    for p_idx, p_text in tqdm(target_passages, desc="Preparing Chunks"):
        chunks = chunking_text(p_text, tokenizer, max_tokens=MAX_PASSAGE_TOKENS, overlap=OVERLAP_TOKENS)
        for c_idx, c_text in enumerate(chunks):
            processing_queue.append({
                "passage_index": p_idx,
                "chunk_index": c_idx,
                "total_chunks": len(chunks),
                "original_text": p_text,
                "chunk_text": c_text
            })

    print(f"📊 Passages to process: {len(target_passages)}")
    print(f"📊 Total chunks: {len(processing_queue)}")

    # --- 5. Batch Inference ---
    BATCH_SIZE = 200
    aggregation_buffer = {} # key: passage_index, value: {data...}

    for i in tqdm(range(0, len(processing_queue), BATCH_SIZE), desc="Processing Batches"):
        batch_items = processing_queue[i : i + BATCH_SIZE]
        batch_prompts = []
        
        for item in batch_items:
            user_content = f"Input Passage: {item['chunk_text']}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            full_prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            batch_prompts.append(full_prompt)

        # Generate
        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)

        # Process Results
        for item, output in zip(batch_items, outputs):
            generated_text = output.outputs[0].text.strip()
            triples = extract_json_output(generated_text)
            
            p_idx = item['passage_index']
            
            if p_idx not in aggregation_buffer:
                aggregation_buffer[p_idx] = {
                    "passage_index": p_idx,
                    "passage_text": item['original_text'],
                    "all_triples": [],
                    "chunks_processed": 0,
                    "total_chunks_expected": item['total_chunks']
                }
            
            aggregation_buffer[p_idx]["all_triples"].extend(triples)
            aggregation_buffer[p_idx]["chunks_processed"] += 1

        # Check for completion & Save
        completed_items = []
        keys_to_remove = []

        for p_idx, data in aggregation_buffer.items():
            if data["chunks_processed"] == data["total_chunks_expected"]:
                
                # --- Deduplication (List of Lists) ---
                unique_triples = []
                seen = set()
                for t in data["all_triples"]:
                    if isinstance(t, list) and len(t) == 3:
                        t_tuple = tuple(t) # (S, P, O)
                        if t_tuple not in seen:
                            seen.add(t_tuple)
                            unique_triples.append(t)
                # -------------------------------------

                completed_items.append({
                    "passage_index": data["passage_index"],
                    "passage_text": data["passage_text"],
                    "triples": unique_triples
                })
                keys_to_remove.append(p_idx)
        
        for k in keys_to_remove:
            del aggregation_buffer[k]

        # Incremental Save
        if completed_items:
            # Re-read file to append safely
            if os.path.exists(output_path):
                with open(output_path, "r", encoding="utf-8") as f:
                    current_data = json.load(f)
            else:
                current_data = []
            
            current_data.extend(completed_items)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(current_data, f, indent=2, ensure_ascii=False)

    print(f"🎉 All Unique Passages Processed.")
    print(f"📂 Triples saved to: {output_path}")
    print(f"📂 Index mapping saved to: {df_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["2wiki", "hotpotqa", "musique"], required=True)
    args = parser.parse_args()
    main(args)