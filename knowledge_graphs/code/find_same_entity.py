system_prompt = """You are an expert Knowledge Graph Engineer specializing in Entity Resolution (Coreference Resolution).
Your task is to identify and group entities that refer to the **same real-world object** (Named Entities) from the provided list of entities and context triples.

### INSTRUCTIONS
1. **Analyze Entities:** Review the provided list of entities.
2. **Check Context:** Use the "Context Triples" to verify if similar names refer to the exact same unique entity.
3. **Group Synonyms:** Group variations (abbreviations, acronyms, partial names, typos) of the same **Specific Named Entity** into a single list.
4. **Filter Noise:** **STRICTLY EXCLUDE** generic terms, common nouns, numbers, dates, and abstract concepts.

### OUTPUT FORMAT
Output **ONLY** a JSON list of lists. Each inner list represents a group of synonymous entities.
Example: `[["USA", "United States", "U.S."], ["JFK", "John F. Kennedy"], ["The Green Album", "Green"]]`

### RULES
1. **Target Proper Nouns Only:** Only group Specific People, Places, Organizations, Events, and Works.
2. **Exclude Generics:** Do NOT include or group common nouns or roles.
   - **Bad:** "president", "mother", "clothes", "the album", "the band", "different stage names", "two sons".
   - **Good:** "President Obama", "Mother Teresa", "The White Album", "System 7".
3. **Exclude Literals:** Do NOT include or group Values.
   - **Dates:** "2020-01-01", "1990s", "May 5th".
   - **Numbers:** "3", "100", "first", "one".
   - **Measurements:** "3 years", "10kg", "50%".
4. **Avoid Super Nodes:** Never group a specific entity with a generic category (e.g., DO NOT group `["Barack Obama", "President"]`). "President" is a title/role, not the person's unique identifier.
5. **Context is King:** If "Jordan" appears in a triple about basketball and another about the Middle East, DO NOT group them.
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
# 2. Helper Functions
# =============================================================================
def extract_json_output(text):
    """
    Extracts JSON List of Lists from model output safely.
    """
    if not isinstance(text, str):
        return []

    if "assistantfinal" in text:
        text = text.split("assistantfinal")[-1]
    
    text = text.strip()
    
    # Try to find the outer list structure [[...]]
    match = re.search(r"(\[\s*\[.*?\]\s*\])", text, re.DOTALL)
    
    if match:
        content = match.group(1)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(content)
            except:
                pass
    
    # Fallback: sometimes models output just the list without outer brackets if prompted poorly, 
    # but the regex above covers the standard case.
    return []

def get_entities_from_triples(triples):
    """
    Extracts a set of all entities (subjects and objects) from triples.
    Filters out non-string objects (like dates/numbers) to save tokens.
    """
    entities = set()
    for t in triples:
        # t = [Subject, Predicate, Object]
        if len(t) < 3: continue
        
        entities.add(t[0]) # Subject
        
        # Object: Include only if string and reasonable length
        if isinstance(t[2], str) and len(t[2]) < 100: 
            entities.add(t[2])
            
    return list(entities)

# =============================================================================
# 3. Main Execution Logic
# =============================================================================
def main(args):
    BASE_DIR = "/workspace/daeyong"
    MODEL_NAME = "/workspace/hf_transformers/gpt-oss-120b"
    
    # Input Paths
    CSV_PATH = f"{BASE_DIR}/benchmarks/{args.dataset}_indexed.csv"
    # CSV_PATH = "/workspace/daeyong/fourth_finetuning_data/final_sft_data_indexed.csv"
    TRIPLES_PATH = f"{BASE_DIR}/knowledge_graphs/{args.dataset}_triples_gleaned.json"
    # TRIPLES_PATH = "/workspace/daeyong/fourth_finetuning_data/final_sft_data_triples_gleaned.json"
    OUTPUT_PATH = f"{BASE_DIR}/knowledge_graphs/{args.dataset}_same_entity_gleaned.json"
    # OUTPUT_PATH = "/workspace/daeyong/fourth_finetuning_data/final_sft_data_same_entity_gleaned.json"
    
    # --- 1. Load Data ---
    print(f"📂 Loading Questions from: {CSV_PATH}")
    if not os.path.exists(CSV_PATH):
        print(f"❌ File not found: {CSV_PATH}")
        return
    df = pd.read_csv(CSV_PATH)
    
    print(f"📂 Loading Triples from: {TRIPLES_PATH}")
    if not os.path.exists(TRIPLES_PATH):
        print(f"❌ File not found: {TRIPLES_PATH}")
        return
        
    with open(TRIPLES_PATH, "r", encoding="utf-8") as f:
        triples_data = json.load(f)
    
    # Index Triples by passage_index for O(1) lookup
    # kg_lookup = { passage_index: { "triples": [...], "passage_text": "..." } }
    print("🔄 Indexing Triples Data...")
    kg_lookup = {}
    for item in triples_data:
        kg_lookup[item['passage_index']] = item

    # --- 2. Prepare Target Items (Aggregation) ---
    print("🔗 Aggregating Passages per Question...")
    
    # Resume Logic
    processed_questions = set()
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            for item in existing_data:
                processed_questions.add(item['question'])
            print(f"🔄 Resuming... Found {len(processed_questions)} processed questions.")
        except:
            print("⚠️ Output file corrupt. Starting from scratch.")
            existing_data = []
    else:
        existing_data = []

    target_items = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing Data"):
        question = row['question']
        
        if question in processed_questions:
            continue
            
        # Parse passage indices
        try:
            p_indices = ast.literal_eval(row['gt_passages_index']) if isinstance(row['gt_passages_index'], str) else row['gt_passages_index']
        except:
            print(f"⚠️ Failed to parse indices for question: {question}")
            continue

        # Aggregate data from all related passages
        aggregated_triples = []
        aggregated_entities = set()
        
        for p_idx in p_indices:
            if p_idx in kg_lookup:
                p_data = kg_lookup[p_idx]
                p_triples = p_data.get('triples_updated', []) # 수정됨!
                
                # Merge Triples
                aggregated_triples.extend(p_triples)
                
                # Merge Entities
                p_entities = get_entities_from_triples(p_triples)
                aggregated_entities.update(p_entities)
        
        if not aggregated_entities:
            continue

        target_items.append({
            "question": question,
            "passage_indices": p_indices,
            "entities": list(aggregated_entities),
            "triples": aggregated_triples
        })

    print(f"📊 Items to process: {len(target_items)}")
    
    if not target_items:
        print("✅ No new items to process.")
        return

    # --- 3. Initialize vLLM ---
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
        max_tokens=7000,
    )

    # --- 4. Batch Processing ---
    BATCH_SIZE = 150
    
    for i in tqdm(range(0, len(target_items), BATCH_SIZE), desc="Processing Batches"):
        batch_items = target_items[i : i + BATCH_SIZE]
        batch_prompts = []
        
        for item in batch_items:            
            user_content = f"""
### ENTITIES TO ANALYZE:
{json.dumps(item['entities'], ensure_ascii=False)}

### CONTEXT TRIPLES (EVIDENCE):
{json.dumps(item['triples'], ensure_ascii=False)}
""".strip()
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

        # Process Outputs
        new_results = []
        for item, output in zip(batch_items, outputs):
            generated_text = output.outputs[0].text.strip()
            synonym_groups = extract_json_output(generated_text)
            
            # Validation
            if not isinstance(synonym_groups, list) or (synonym_groups and not isinstance(synonym_groups[0], list)):
                synonym_groups = []
            
            # Meaningless groups filtering (len > 1)
            valid_synonym_groups = [group for group in synonym_groups if len(group) > 1]

            new_results.append({
                "question": item['question'],
                "passage_indices": item['passage_indices'],
                "synonym_groups": valid_synonym_groups
            })

        # Save Incrementally
        if os.path.exists(OUTPUT_PATH):
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                current_data = json.load(f)
        else:
            current_data = []
        
        current_data.extend(new_results)
        
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(current_data, f, indent=2, ensure_ascii=False)

    print(f"🎉 Cross-Passage Entity Resolution Completed.")
    print(f"📂 Saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["2wiki", "hotpotqa", "musique"], required=True)
    args = parser.parse_args()
    
    main(args)