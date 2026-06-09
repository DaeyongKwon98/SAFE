import json
import os
import re
import argparse
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# =============================================================================
# 1. Configuration & Prompts
# =============================================================================
MODEL_NAME = "/workspace/hf_transformers/gpt-oss-120b"
INPUT_FILE = "/workspace/daeyong/fourth_finetuning_data/younger_older_reverse_generated.json"
OUTPUT_FILE = "/workspace/daeyong/fourth_finetuning_data/younger_older_reverse_feedback.json"

system_prompt = """You are an expert AI evaluator providing feedback on reasoning steps involving age and dates.
Your task is to evaluate a specific **Logical Step** in a reasoning chain where the user applies "Inverse Logic" regarding birth dates.

**Task:**
Identify the "Logical Fallacy" where the user correctly identifies the chronological order of years (e.g., 1950 < 1990) but incorrectly assigns age labels (e.g., claiming the earlier year is "younger" or the later year is "older").

**Requirements:**
1.  **error_type**: ALWAYS return "Logical Fallacy".
2.  **diagnosis**: Explain that birth years function inversely to age. Specifically state: "A smaller/earlier birth year implies an older age, not younger." or "A larger/later birth year implies a younger age, not older."
3.  **guidance**: Provide the correct logic for determining age based on birth years.

**Output Format:**
Provide ONLY a valid JSON object:
{
  "error_type": "Logical Fallacy",
  "diagnosis": "...",
  "guidance": "..."
}

**Few-Shot Examples:**

Example 1:

Input:
Question: Who is younger, Alice or Bob?
Reasoning Chain:
['Step 1: According to Passage 1, Alice was born in 1974. (Attribution)',
 'Step 2: According to Passage 2, Bob was born in 1979. (Attribution)',
 'Step 3: Since 1974 is before 1979, Alice is younger than Bob. (Logical)']

Output:
{
  "error_type": "Logical Fallacy",
  "diagnosis": "The reasoning correctly identifies that 1974 comes before 1979. However, Step 3 incorrectly concludes that the person born earlier (1974) is younger. In fact, being born earlier means having lived longer (older).",
  "guidance": "Since 1974 is an earlier date than 1979, specify the younger person as Bob."
}

Example 2:

Input:
Question: Who is older, Charlie or David?
Reasoning Chain:
['Step 1: According to Passage A, Charlie was born on May 10, 1990. (Attribution)',
 'Step 2: According to Passage B, David was born on June 15, 1950. (Attribution)',
 'Step 3: Since 1990 is a larger than 1950, Charlie is older than David. (Logical)']

Output:
{
  "error_type": "Logical Fallacy",
  "diagnosis": "The reasoning in Step 3 incorrectly associates a later birth date with being older. A person born in 1990 has lived fewer years than someone born in 1950.",
  "guidance": "Since 1990 is a later date than 1950, specify the older person as David."
}
""".strip()

# =============================================================================
# 2. Helper Functions
# =============================================================================
def extract_json(text):
    text = text.strip().split("assistantfinal")[-1].strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        content = match.group(1).strip()
    else:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            content = match.group(1).strip()
        else:
            return text

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content

# =============================================================================
# 3. Main Logic
# =============================================================================
def main():
    print(f"📂 Input Path: {INPUT_FILE}")
    print(f"📂 Output Path: {OUTPUT_FILE}")

    # Load Data
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        return

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df_input = pd.DataFrame(data)
    except ValueError:
        print("⚠️ JSON load failed. Trying pandas read_json...")
        df_input = pd.read_json(INPUT_FILE)
    
    print(f"✅ Loaded Data: {len(df_input)} records.")

    if 'generated_inverse_reasoning' not in df_input.columns:
        print("❌ Error: Input file must contain 'generated_inverse_reasoning' column.")
        return

    # Resume Logic
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding='utf-8') as f:
            try:
                existing_results = json.load(f)
            except json.JSONDecodeError:
                existing_results = []
        
        processed_ids = {str(item.get('question')) for item in existing_results}
        print(f"🔄 Resuming... Found {len(processed_ids)} processed items.")
        df_input = df_input[~df_input['question'].astype(str).isin(processed_ids)]
    else:
        existing_results = []

    if df_input.empty:
        print("✅ No new items to process.")
        return

    # Initialize vLLM
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
        max_tokens=6000, 
    )

    # Batch Processing
    BATCH_SIZE = 100
    records = df_input.to_dict('records')
    final_results = existing_results

    print(f"🚀 Starting execution in batches of {BATCH_SIZE}...")

    for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Processing Batches"):
        batch_records = records[i : i + BATCH_SIZE]
        batch_prompts = []
        valid_indices = []

        for idx, row in enumerate(batch_records):
            question = row.get('question', '')
            reasoning_steps = row.get('generated_inverse_reasoning', [])

            if not isinstance(reasoning_steps, list) or not reasoning_steps:
                continue
            
            reasoning_text = "\n".join(reasoning_steps)

            user_content = f"""Question: {question}

Reasoning Chain:
{reasoning_text}
""".strip()

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            
            full_prompt = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=False
            )
            batch_prompts.append(full_prompt)
            valid_indices.append(idx)

        if not batch_prompts:
            continue

        # Generate
        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)

        # Process Outputs
        for local_idx, output in zip(valid_indices, outputs):
            row = batch_records[local_idx]
            generated_text = output.outputs[0].text.strip()
            
            feedback_json = extract_json(generated_text)
            
            result_entry = row.copy()
            if isinstance(feedback_json, dict):
                result_entry["gold_feedback"] = feedback_json
            else:
                result_entry["gold_feedback"] = {
                    "error_type": "Parse Error", 
                    "raw_output": generated_text
                }

            final_results.append(result_entry)

        # Save
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"🎉 All Completed. Total items saved: {len(final_results)}")
    print(f"📂 Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()