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
# 1. Configuration & System Prompt
# =============================================================================
MODEL_NAME = "/workspace/hf_transformers/gpt-oss-120b"
INPUT_FILE = "/workspace/daeyong/fourth_finetuning_data/younger_older_reverse.json"
OUTPUT_FILE = "/workspace/daeyong/fourth_finetuning_data/younger_older_reverse_generated.json"

system_prompt = """You are a data generation assistant designed to create high-quality reasoning error examples for training feedback models.
Your task is to generate a reasoning chain that demonstrates a specific **"Logical Fallacy"** while maintaining perfect attribution accuracy.

**Input:**
- A comparison question about age (e.g., "Who is younger?", "Who is older?", "Who was born earlier/later?").
- Retrieved passages containing birth dates.

**Output:**
- A python list of strings, representing the reasoning steps.

**Strict Generation Rules:**
1.  **Attribution Steps (Must be Accurate):**
    - **CRITICAL:** You must verify which passage actually contains the person's birth date. Do not guess the passage number.
    - If the person appears in Passage 5, you MUST cite Passage 5.
    - Format: `Step N: According to Passage X, the birth date of [Person] is [Date]. (Attribution)`

2.  **The Flawed Logical Step (The Error):**
    - Perform a mathematically correct comparison of the years/dates (e.g., "1920 is before 1950").
    - **APPLY THE LOGICAL FALLACY:** Deliberately map the numerical magnitude incorrectly to the age concept.
        - **Fallacy Type A (Smaller = Younger):** Claim that because the birth year is smaller (earlier), the person is "younger".
        - **Fallacy Type B (Larger = Older):** Claim that because the birth year is larger (later), the person is "older".
    - Format: `Step N: Since [Date A] is [before/after] [Date B], [Person] is [younger/older - INVERSE LOGIC]. (Logical)`

**Template Examples:**

**Example 1 (Fallacy: Earlier Year -> Younger):**
User Question: Who is younger, Person A (born 1920) or Person B (born 1950)?
Passage 3: Person A was born in 1920.
Passage 7: Person B was born in 1950.
**Your Output:**
[
"Step 1: According to Passage 3, Person A was born in 1920. (Attribution)",
"Step 2: According to Passage 7, Person B was born in 1950. (Attribution)",
"Step 3: Since 1920 is before 1950, Person A is younger than Person B. (Logical)"
]

**Example 2 (Fallacy: Later Year -> Older):**
User Question: Who is older, Person X (born 1990) or Person Y (born 1980)?
Passage 1: Person X was born in 1990.
Passage 2: Person Y was born in 1980.
**Your Output:**
[
"Step 1: According to Passage 1, Person X was born in 1990. (Attribution)",
"Step 2: According to Passage 2, Person Y was born in 1980. (Attribution)",
"Step 3: Since 1990 is a larger number than 1980, Person X is older than Person Y. (Logical)"
]
""".strip()

# =============================================================================
# 2. Helper Functions
# =============================================================================
def extract_python_list(text):
    """
    Extracts python list from text using regex and ast.
    """
    if not isinstance(text, str):
        return []

    text = text.strip().split("assistantfinal")[-1].strip()
    match = re.search(r"```(?:python)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if match:
        content = match.group(1).strip()
    else:
        match = re.search(r"(\[.*\])", text, re.DOTALL)
        if match:
            content = match.group(1).strip()
        else:
            return text 

    try:
        return ast.literal_eval(content)
    except (ValueError, SyntaxError):
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

    if 'question' not in df_input.columns or 'retrieved_passages' not in df_input.columns:
        print("❌ Error: Input file must columns 'question' and 'retrieved_passages'.")
        return

    # Resume Logic
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding='utf-8') as f:
            try:
                existing_results = json.load(f)
            except json.JSONDecodeError:
                existing_results = []
        
        processed_questions = {item['question'] for item in existing_results if 'question' in item}
        print(f"🔄 Resuming... Found {len(processed_questions)} processed items.")
        df_input = df_input[~df_input['question'].isin(processed_questions)]
    else:
        existing_results = []

    if df_input.empty:
        print("✅ No new questions to process.")
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
        
        for row in batch_records:
            question = row.get('question', '')
            passages = row.get('retrieved_passages', [])
            
            if isinstance(passages, list):
                passages_text = "\n".join(passages)
            else:
                passages_text = str(passages)

            user_content = f"Question: {question}\nPassages:\n{passages_text}"

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

        # Generate
        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)

        # Process Outputs
        new_results = []
        for row, output in zip(batch_records, outputs):
            generated_text = output.outputs[0].text.strip()
            
            # Parse List
            reasoning_steps = extract_python_list(generated_text)
            
            result_entry = row.copy()
            result_entry["generated_inverse_reasoning"] = reasoning_steps
            # result_entry["raw_generation"] = generated_text 
            
            new_results.append(result_entry)

        # Save
        final_results.extend(new_results)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"🎉 All Completed. Total items saved: {len(final_results)}")
    print(f"📂 Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()