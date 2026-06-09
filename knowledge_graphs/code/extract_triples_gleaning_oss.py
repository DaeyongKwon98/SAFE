system_prompt = """You are a meticulous Knowledge Graph Auditor and QA Specialist.
Your goal is to review the extraction results from a previous step and identify **MISSING** factual triples that were overlooked.

### INPUTS
You will be provided with:
1. **Original Text Passage**: The source text.
2. **Existing Triples**: A list of triples already extracted from the text.

### TASK
Compare the **Original Text** against the **Existing Triples**. Extract **NEW** factual triples that are present in the text but **NOT** in the Existing Triples list.

### CRITICAL RULES
1. **NO DUPLICATES**: Do not output any triple that is semantically identical to one in the "Existing Triples" list.
2. **Focus on Detail**: Look specifically for:
   - Secondary entities mentioned in the text (not just the main subject).
   - Specific dates, numbers, or measurements previously missed.
   - Relationships between secondary entities.
   - Adjectives or roles acting as attributes (e.g., "Polish-French" -> nationality).
3. **Coreference Resolution (Mandatory)**: Just like the first step, you MUST resolve pronouns (he, she, it, they) to their full canonical entity names. **Never** use pronouns in your output.
4. **Atomic & Standardized**: Follow the same extraction standards (Atomic facts, standardized predicates).
5. **Aliases & Alternate Names:**
   - If an entity has multiple names (e.g., "better known as", "born as", "pseudonym", "nickname"), you MUST extract a triple linking them.
   - Use predicates like `same_as`, `alias`, `birth_name`, or `alternative_name`.
   - Example: "Donna Paige Helmintoller, better known as Paige O'Hara" -> Extract `["Donna Paige Helmintoller", "same_as", "Paige O'Hara"]`.

### OUTPUT FORMAT
Output **ONLY** a standard JSON list of lists containing the **NEWLY EXTRACTED** triples.
If no new triples are found, output an empty list `[]`.

JSON Format:
[
  ["Subject", "Predicate", "Object"],
  ["Subject", "Predicate", "Object"],
  ...
]
""".strip()

import json
import os
import re
import argparse
import ast
from copy import deepcopy
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# =============================================================================
# 1. Configuration & System Prompt
# =============================================================================
MODEL_NAME = "/workspace/hf_transformers/gpt-oss-120b"
MAX_STEPS = 2
BATCH_SIZE = 256

# =============================================================================
# 2. Helper Functions
# =============================================================================
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

def construct_user_prompt(passage_text, current_triples):
    """
    Constructs the User Prompt for Gleaning.
    """
    triples_str = json.dumps(current_triples, ensure_ascii=False)
    
    prompt = f"""### Text Passage:
{passage_text}

### Existing Triples (Already Extracted):
{triples_str}

Please find any MISSING factual triples that were overlooked in the Existing Triples.
If no new triples are found, output an empty list [].
"""
    return prompt

# =============================================================================
# 3. Main Execution Logic
# =============================================================================
def main(args):
    # --- Path Setup ---
    base_dir = "/workspace/daeyong/knowledge_graphs"
    input_path = f"{base_dir}/{args.dataset}_triples.json" # dev 다 뺐음! 나중에 확인할것
    output_path = f"{base_dir}/{args.dataset}_triples_gleaned.json"
    stats_output_path = f"{base_dir}/{args.dataset}_gleaning_stats.json"
    # input_path = "/workspace/daeyong/fourth_finetuning_data/final_sft_data_triples.json"
    # output_path = "/workspace/daeyong/fourth_finetuning_data/final_sft_data_triples_gleaned.json"
    # stats_output_path = "/workspace/daeyong/fourth_finetuning_data/final_sft_data_gleaning_stats.json"

    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return

    print(f"📂 Loading Data from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # --- Initialize Data & Statistics ---
    # active_indices: indices of items that still need gleaning
    # statistics_list: stores stats for each item, initialized with Step 1/2 added = 0
    active_indices = []
    statistics_list = []

    for i, item in enumerate(data):
        # Deep copy original triples
        original_triples = item.get('triples', [])
        item['triples_updated'] = deepcopy(original_triples)
        
        # Initialize stats structure for this item
        stats_entry = {
            "passage_index": item.get("passage_index", i), # Fallback to list index if key missing
            "initial_triples_count": len(original_triples),
            "gleaning_step_1_added": 0,
            "gleaning_step_2_added": 0,
            "total_final_triples": 0 
        }
        statistics_list.append(stats_entry)
        
        active_indices.append(i)

    print(f"📊 Total items to process: {len(data)}")

    # --- Initialize vLLM ---
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
        max_tokens=3000,
    )

    # --- Gleaning Loop (Max Steps) ---
    for step in range(1, MAX_STEPS + 1):
        if not active_indices:
            print("✅ No more items to glean. Stopping early.")
            break

        print(f"\n🔄 Gleaning Step {step}/{MAX_STEPS} | Active items: {len(active_indices)}")
        
        # 1. Prepare Batches for Active Items
        prompts = []
        batch_mapping = [] # (index in 'data', original_active_idx)

        for idx in active_indices:
            item = data[idx]
            user_content = construct_user_prompt(item['passage_text'], item['triples_updated'])
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            full_prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            prompts.append(full_prompt)
            batch_mapping.append(idx)

        # 2. Inference
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

        # 3. Process Outputs & Update State
        next_active_indices = []
        
        for idx_in_data, output in zip(batch_mapping, outputs):
            generated_text = output.outputs[0].text.strip()
            new_triples = extract_json_output(generated_text)
            
            added_count = 0
            
            # Check if valid new triples are found
            if new_triples and isinstance(new_triples, list) and len(new_triples) > 0:
                # Deduplication logic
                current_set = set(tuple(t) for t in data[idx_in_data]['triples_updated'])
                
                for t in new_triples:
                    if isinstance(t, list) and len(t) == 3:
                        t_tuple = tuple(t)
                        if t_tuple not in current_set:
                            data[idx_in_data]['triples_updated'].append(t)
                            current_set.add(t_tuple)
                            added_count += 1
                
                # Update Statistics for this step
                if step == 1:
                    statistics_list[idx_in_data]["gleaning_step_1_added"] = added_count
                elif step == 2:
                    statistics_list[idx_in_data]["gleaning_step_2_added"] = added_count
                
                # If we actually added something, keep this item active for next step
                if added_count > 0:
                    next_active_indices.append(idx_in_data)
            
            # If empty list or invalid output, this item converges (leaves active list)

        print(f"   End of Step {step}: {len(active_indices) - len(next_active_indices)} items converged.")
        active_indices = next_active_indices

    # --- Final Statistics Calculation ---
    for i, stats in enumerate(statistics_list):
        stats["total_final_triples"] = len(data[i]['triples_updated'])

    # --- Final Save ---
    print(f"💾 Saving gleaned results to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"📊 Saving statistics to: {stats_output_path}")
    with open(stats_output_path, "w", encoding="utf-8") as f:
        json.dump(statistics_list, f, indent=2, ensure_ascii=False)
    
    print("🎉 Gleaning Process Completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["2wiki", "hotpotqa", "musique"], required=True)
    args = parser.parse_args()
    main(args)