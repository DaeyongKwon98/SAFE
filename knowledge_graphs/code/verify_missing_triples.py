import json
import os
import re
import ast
import argparse
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

system_prompt = """You are a strict Fact Verification Assistant.
Your task is to determine if a given "Hypothesized Triple" is **supported by** or **logically entailed by** the provided "Reference Passages".

### INSTRUCTIONS
1. **Analyze the Context:** Read the Reference Passages carefully.
2. **Verify the Triple:** Check if the relationship described in the triple `[Subject, Predicate, Object]` is explicitly stated OR strongly implied by the passages.
3. **Implicit Support:** If the triple is not written word-for-word but the information is clearly inferable from the passages (e.g., "Paris is in France" implies "Paris is a French city"), mark it as **SUPPORTED**.
4. **Contradiction/Neutral:** If the passages contradict the triple or do not contain enough information to confirm it, mark it as **UNSUPPORTED**.

### INPUT DATA
- **Reference Passages:** Extracts from the ground truth passages.
- **Hypothesized Triples:** A list of triples to verify.

### OUTPUT FORMAT
Output **ONLY** a JSON list of objects.
[
  {
    "triple": ["Subject", "Predicate", "Object"],
    "is_supported": true, // or false
    "reasoning": "Explanation citing the text evidence."
  },
  ...
]

### EXAMPLES

**Case 1: Explicit Support (Directly Stated)**
- **Reference Passages:**
Passage 1: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
- **Hypothesized Triples:** `[["Eiffel Tower", "located_in", "Paris"], ["Gustave Eiffel", "role", "engineer"]]`
- **Output:**
[
  {
    "triple": ["Eiffel Tower", "located_in", "Paris"],
    "is_supported": true,
    "reasoning": "Passage 1 explicitly states the tower is 'on the Champ de Mars in Paris, France'."
  },
  {
    "triple": ["Gustave Eiffel", "role", "engineer"],
    "is_supported": true,
    "reasoning": "Passage 1 refers to Gustave Eiffel as 'the engineer Gustave Eiffel'."
  }
]

**Case 2: Implicit Support (Logical Entailment)**
- **Reference Passages:**
Passage 1: Barack Obama served as the 44th president of the United States from 2009 to 2017. He was born in Honolulu, Hawaii.
- **Hypothesized Triples:** `[["Barack Obama", "nationality", "American"], ["Honolulu", "located_in", "United States"]]`
- **Output:**
[
  {
    "triple": ["Barack Obama", "nationality", "American"],
    "is_supported": true,
    "reasoning": "Supported implicitly. Being the 'president of the United States' logically entails American citizenship/nationality."
  },
  {
    "triple": ["Honolulu", "located_in", "United States"],
    "is_supported": true,
    "reasoning": "Supported implicitly. The text says he was born in 'Honolulu, Hawaii' and served as US president. Common knowledge connects Hawaii to the US, but structurally, the text implies Honolulu is a location within the context of his US presidency and birth."
  }
]

**Case 3: Unsupported (Missing Information)**
- **Reference Passages:**
Passage 1: 'Inception' is a 2010 science fiction action film written and directed by Christopher Nolan.
Passage 2: In the movie 'Inception', it stars Leonardo DiCaprio as a professional thief.
- **Hypothesized Triples:** `[["Inception", "budget", "$160 million"], ["Leonardo DiCaprio", "stars", "professional thief"]]`
- **Output:**
[
  {
    "triple": ["Inception", "budget", "$160 million"],
    "is_supported": false,
    "reasoning": "The passage mentions the genre, director, and star, but contains no information about the film's budget."
  },
  {
    "triple": ["Leonardo DiCaprio", "stars", "professional thief"],
    "is_supported": true,
    "reasoning": "The passage states that Leonardo DiCaprio stars as a professional thief, directly supporting the triple."
  }
]

**Case 4: Contradiction (Direct Conflict)**
- **Reference Passages:**
Passage 1: The 2022 World Cup was hosted by Qatar. It was the first World Cup to be held in the Arab world.
- **Hypothesized Triples:** `[["2022 World Cup", "host_country", "France"]]`
- **Output:**
[
  {
    "triple": ["2022 World Cup", "host_country", "France"],
    "is_supported": false,
    "reasoning": "Contradiction. The passage 1 explicitly states the 2022 World Cup was hosted by 'Qatar', not France."
  }
]

**Case 5: Contextual/Bridging Support (Multi-hop Inference)**
- **Reference Passages:**
Passage 1: The iPhone is a smartphone line designed by Apple Inc.
Passage 2: Steve Jobs co-founded Apple Inc. in 1976.
- **Hypothesized Triples:** `[["iPhone", "manufacturer", "Apple Inc."], ["Steve Jobs", "affiliated_with", "iPhone"]]`
- **Output:**
[
  {
    "triple": ["iPhone", "manufacturer", "Apple Inc."],
    "is_supported": true,
    "reasoning": "Passage 1 states the iPhone is 'designed by Apple Inc.', which supports the manufacturer relationship."
  },
  {
    "triple": ["Steve Jobs", "affiliated_with", "iPhone"],
    "is_supported": true,
    "reasoning": "Supported by combining context. Passage 1 links iPhone to Apple, and Passage 2 links Steve Jobs to Apple. Therefore, a connection between Jobs and iPhone is contextually valid."
  }
]
""".strip()

# =============================================================================
# 2. Helper Functions
# =============================================================================
def extract_json_output(text):
    """
    Extracts JSON List of Objects from model output safely.
    Target: [{"triple": [], "is_supported": bool, "reasoning": str}, ...]
    """
    if not isinstance(text, str):
        return []

    if "assistantfinal" in text:
        text = text.split("assistantfinal")[-1]
    
    text = text.strip()
    
    # Try finding JSON block
    match = re.search(r"(\[\s*\{.*?\}\s*\])", text, re.DOTALL)
    if match:
        content = match.group(1)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(content)
            except:
                pass
    
    # Fallback for raw text without brackets (rare but possible)
    if text.startswith("[") and text.endswith("]"):
        try:
            return json.loads(text)
        except:
            pass
            
    return []

# =============================================================================
# 3. Main Execution Logic
# =============================================================================
def main(args):
    # --- Configuration ---
    BASE_DIR = "/workspace/daeyong"
    MODEL_NAME = "/workspace/hf_transformers/gpt-oss-120b"
    
    # Files
    MISSING_TRIPLES_PATH = f"{BASE_DIR}/knowledge_graphs/{args.dataset}_missing_triples_gleaned.json"
    # MISSING_TRIPLES_PATH = "/workspace/daeyong/fourth_finetuning_data/final_sft_data_missing_triples_gleaned.json"
    DATASET_CSV_PATH = f"{BASE_DIR}/benchmarks/{args.dataset}.csv"
    # DATASET_CSV_PATH = "/workspace/daeyong/fourth_finetuning_data/final_sft_data_indexed.csv"
    OUTPUT_PATH = f"{BASE_DIR}/knowledge_graphs/{args.dataset}_verified_triples_gleaned.json"
    # OUTPUT_PATH = "/workspace/daeyong/fourth_finetuning_data/final_sft_data_verified_triples_gleaned.json"

    print(f"📂 Loading Missing Triples from: {MISSING_TRIPLES_PATH}")
    if not os.path.exists(MISSING_TRIPLES_PATH):
        print(f"❌ File not found: {MISSING_TRIPLES_PATH}")
        return
    
    with open(MISSING_TRIPLES_PATH, "r", encoding="utf-8") as f:
        missing_data = json.load(f)

    print(f"📂 Loading Dataset CSV from: {DATASET_CSV_PATH}")
    if not os.path.exists(DATASET_CSV_PATH):
        print(f"❌ File not found: {DATASET_CSV_PATH}")
        return
    
    df = pd.read_csv(DATASET_CSV_PATH)
    
    # --- Preprocessing: Create Lookup Dictionary for GT Passages ---
    print("🔄 Indexing GT Passages by Question...")
    # Create a dictionary: { question_text: gt_passages_formatted_string }
    # Using 'question' column as key. ensuring whitespace consistency.
    question_to_passages = {}
    if isinstance(df['gt_passages'].iloc[0], str):
        df['gt_passages'] = df['gt_passages'].apply(ast.literal_eval)
    for _, row in df.iterrows():
        q_clean = row['question'].strip()
        question_to_passages[q_clean] = [f"Passage {i+1}: {p}" for i, p in enumerate(row['gt_passages'])]

    # --- Filter Items to Process ---
    target_items = []
    
    # Check for items that actually have missing triples generated
    for item in missing_data:
        question = item['question'].strip()
        recovery_result = item.get('recovery_result', [])
        
        # Skip if no triples to verify
        if not recovery_result:
            continue
            
        # Retrieve GT Passages
        passages_text = "\n".join(question_to_passages.get(question, []))
        if not passages_text:
            print(f"⚠️ Warning: No GT Passages found for question: {question[:50]}...")
            continue
            
        # Extract just the triples [S, P, O] for verification
        triples_to_verify = [entry['missing_triple'] for entry in recovery_result if 'missing_triple' in entry]
        
        if not triples_to_verify:
            continue

        target_items.append({
            "original_item": item, # Keep original data to merge later
            "question": question,
            "passages_text": passages_text,
            "triples_to_verify": triples_to_verify
        })

    print(f"📊 Items to Verify: {len(target_items)}")
    
    # --- Initialize vLLM ---
    print(f"🚀 Loading vLLM Model: {MODEL_NAME}")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=4,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        max_model_len=5000,
        enable_prefix_caching=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=3000,
    )

    # --- Batch Inference ---
    BATCH_SIZE = 100
    final_results = []
    
    # Output file handling (Clean start or Resume - Here assuming clean start for verification)
    if os.path.exists(OUTPUT_PATH):
         print(f"⚠️ Output file exists. Overwriting: {OUTPUT_PATH}")
    
    for i in tqdm(range(0, len(target_items), BATCH_SIZE), desc="Verifying Batches"):
        batch_items = target_items[i : i + BATCH_SIZE]
        batch_prompts = []
        
        for item in batch_items:
            user_content = f"""
### REFERENCE PASSAGES:
{item['passages_text']}

### HYPOTHESIZED TRIPLES TO VERIFY:
{json.dumps(item['triples_to_verify'], ensure_ascii=False)}
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
        for item, output in zip(batch_items, outputs):
            generated_text = output.outputs[0].text.strip()
            verification_list = extract_json_output(generated_text)
            
            # Merge verification result back into the original item structure
            processed_item = item['original_item'].copy()
            
            # Map verification results back to the recovery_result
            # We create a dictionary for quick lookup of verification by triple signature
            ver_dict = {}
            for v in verification_list:
                # Use tuple of triple as key
                t_key = tuple(v.get('triple', []))
                ver_dict[t_key] = v

            # Update the recovery_result list with verification status
            updated_recovery_result = []
            for rec_entry in processed_item['recovery_result']:
                triple = rec_entry.get('missing_triple')
                if triple:
                    t_key = tuple(triple)
                    ver_info = ver_dict.get(t_key, {"is_supported": False, "reasoning": "Verification failed or parsing error."})
                    
                    rec_entry['verification_status'] = {
                        "is_supported": ver_info.get('is_supported', False),
                        "reasoning": ver_info.get('reasoning', "No reasoning provided.")
                    }
                updated_recovery_result.append(rec_entry)
            
            processed_item['recovery_result'] = updated_recovery_result
            
            # Add summary flag
            all_supported = all(entry['verification_status']['is_supported'] for entry in updated_recovery_result)
            processed_item['all_triples_supported'] = all_supported
            
            final_results.append(processed_item)

        # Incremental Save
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"🎉 Verification Completed.")
    print(f"📂 Results saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Verify Missing Triples using vLLM")
    argparser.add_argument("--dataset", type=str, choices=["2wiki", "hotpotqa", "musique"], required=True)
    args = argparser.parse_args()
    main(args)