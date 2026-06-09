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
INPUT_FILE = "/workspace/daeyong/fourth_finetuning_data/when_to_where_generated.json"
OUTPUT_FILE = "/workspace/daeyong/fourth_finetuning_data/when_to_where_feedback.json"

# System prompt optimized for "When -> Where" mismatch
MISMATCH_FEEDBACK_SYSTEM_PROMPT = """You are an expert AI evaluator providing feedback on reasoning steps involving attribute extraction.
Your task is to evaluate a specific **Attribution Step** in a reasoning chain where the user answers a specific question type with irrelevant information.

**Task:**
Identify the "Off-topic" error where the user extracts a Place/Location (Where) when the question explicitly asks for a Time/Date (When).

**Requirements:**
1.  **error_type**: ALWAYS return "Off-topic".
2.  **diagnosis**: Explain the mismatch between the requested attribute (Date/Time) and the provided attribute (Location). State clearly: "The question asks for 'When' (a date/time), but the step provides 'Where' (a location)."
3.  **guidance**: Provide a direct and precise instruction.
    - Explicitly state **which passage** to look into.
    - Explicitly state **what specific information** (e.g., "birth date", "death year") to extract for **which person**.
    - **Do NOT use "e.g." or vague phrases.** Be authoritative and specific.

**Output Format:**
Provide ONLY a valid JSON object:
{
  "error_type": "Off-topic",
  "diagnosis": "...",
  "guidance": "..."
}

**Few-Shot Examples:**

Input:
Question: When was the director of the film "Parasite" born?
Retrieved Passages:
Passage 1: Bong Joon-ho (born September 14, 1969 in Daegu, South Korea) is a South Korean film director.
Reasoning Chain:
['Step 1: According to Passage 1, the director of the film Parasite is Bong Joon-ho. (Attribution)',
 'Step 2: According to Passage 1, Bong Joon-ho was born in Daegu, South Korea. (Attribution)']

Output:
{
  "error_type": "Off-topic",
  "diagnosis": "The step extracts the birth place (Where), but the question asks for the birth date (When). This information is irrelevant to the specific question asked.",
  "guidance": "Extract Bong Joon-ho's specific birth date (September 14, 1969) from Passage 1."
}

Input:
Question: When did the star of "Jaws" die?
Retrieved Passages:
Passage 1: Roy Scheider (November 10, 1932 – February 10, 2008) was an American actor and best known for his role in Jaws. He died in Little Rock, Arkansas.
Reasoning Chain:
['Step 1: According to Passage 1, the star of Jaws is Roy Scheider. (Attribution)',
 'Step 2: According to Passage 1, Roy Scheider died in Little Rock, Arkansas. (Attribution)']

Output:
{
  "error_type": "Off-topic",
  "diagnosis": "The step provides the location of death (Little Rock, Arkansas), which answers 'Where'. The question explicitly asks 'When', requiring a specific date.",
  "guidance": "Find Roy Scheider's death date (February 10, 2008) from Passage 1."
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

    if 'generated_when_mismatch_reasoning' not in df_input.columns:
        print("❌ Error: Input file must contain 'generated_when_mismatch_reasoning' column.")
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
            passages = row.get('retrieved_passages', [])
            reasoning_steps = row.get('generated_when_mismatch_reasoning', [])

            # Skip if reasoning steps are missing or empty
            if not isinstance(reasoning_steps, list) or not reasoning_steps:
                continue
            
            # Format passages
            if isinstance(passages, list):
                # If passages are just strings, number them
                passages_text = "\n".join([f"Passage {j+1}: {p}" for j, p in enumerate(passages)])
            elif isinstance(passages, str):
                try:
                    passages_list = eval(passages)
                    passages_text = "\n".join([f"Passage {j+1}: {p}" for j, p in enumerate(passages_list)])
                except:
                    passages_text = passages
            else:
                passages_text = str(passages)
            
            reasoning_text = "\n".join(reasoning_steps)
            
            # The target step is the final step in the generated chain (the one with the error)
            target_step = reasoning_steps[-1]

            user_content = f"""Question: {question}
            
Retrieved Passages:
{passages_text}

Reasoning Chain:
{reasoning_text}
""".strip()

            messages = [
                {"role": "system", "content": MISMATCH_FEEDBACK_SYSTEM_PROMPT},
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

        # Incremental Save
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"🎉 All Completed. Total items saved: {len(final_results)}")
    print(f"📂 Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()