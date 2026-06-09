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
MODEL_NAME = "/workspace/hf_transformers/gpt-oss-120b"
INPUT_FILE = "/workspace/daeyong/fourth_finetuning_data/grandfather.json"
OUTPUT_FILE = "/workspace/daeyong/fourth_finetuning_data/grandfather_gender_generated.json"

KINSHIP_SIDE_ERROR_SYSTEM_PROMPT = """You are a data generation assistant designed to create specific "Reasoning Error" examples for training feedback models.
Your task is to generate a chain of reasoning steps that demonstrates a **"Logical Fallacy (Maternal/Paternal Confusion)"**.

**Input:**
- A question asking for a specific side's Grandparent (e.g., "**Paternal** grandfather" or "**Maternal** grandmother").
- Retrieved passages describing the lineage.

**Output:**
- A python list of strings, representing the reasoning steps.

**Strict Generation Rules:**
1.  **Attribution Steps (Correct):**
    - Correctly identify the **Parent** (e.g., "According to Passage K, X is the **mother** of Y").
    - Correctly identify the **Grandparent** (e.g., "According to Passage K, Z is the father of X").
    In the Attribution Step, you have to specify the exact passage number as Passage K.
2.  **The Flawed Logical Step (The Error):**
    - Construct a logic that ignores the gender of the parent (Mother vs Father) when assigning the Grandparent title.
    - **CRITICAL ERROR:**
        - If the intermediate parent is the **Mother**, conclude it is the **Paternal** Grandparent.
        - If the intermediate parent is the **Father**, conclude it is the **Maternal** Grandparent.
    - Format: `Step N: Since [Grandparent] is the father of [Mother], [Grandparent] is the [Paternal Grandfather] of [Child]. (Logical)`

**Template Examples:**

**Example 1 (Mother's father -> Paternal Grandfather Error):**
Question: Who is the paternal grandfather of Prince William?
Retrieved Passages:
Passage 1: Prince William's mother was Princess Diana. Diana's father was Earl Spencer.

**Your Output:**
[
"Step 1: According to Passage 1, the mother of Prince William is Princess Diana. (Attribution)",
"Step 2: According to Passage 1, the father of Princess Diana is Earl Spencer. (Attribution)",
"Step 3: Since Earl Spencer is the father of Prince William's mother, Earl Spencer is the paternal grandfather of Prince William. (Logical)"
]

**Example 2 (Father's mother -> Maternal Grandmother Error):**
Question: Who is the maternal grandmother of King Felipe VI?
Retrieved Passages:
Passage 1: Felipe VI is the son of Juan Carlos I. Juan Carlos's mother was María de las Mercedes.

**Your Output:**
[
"Step 1: According to Passage 1, the father of King Felipe VI is Juan Carlos I. (Attribution)",
"Step 2: According to Passage 1, the mother of Juan Carlos I is María de las Mercedes. (Attribution)",
"Step 3: Since María de las Mercedes is the mother of King Felipe VI's father, she is the maternal grandmother of King Felipe VI. (Logical)"
]
""".strip()

# =============================================================================
# 2. Helper Functions
# =============================================================================
def extract_python_list(text):
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

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        return

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df_input = pd.DataFrame(data)
    except ValueError:
        df_input = pd.read_json(INPUT_FILE)
    
    print(f"✅ Loaded Data: {len(df_input)} records.")

    # 필수 컬럼 체크
    if 'question' not in df_input.columns or 'retrieved_passages' not in df_input.columns:
        print("❌ Error: Input file must contain 'question' and 'retrieved_passages'.")
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
            
            if isinstance(passages, str):
                passages = eval(passages)
            passages_text = "\n".join([f"Passage {idx+1}: {p}" for idx, p in enumerate(passages)])

            user_content = f"Question: {question}\nRetrieved Passages:\n{passages_text}"
            messages = [
                {"role": "system", "content": KINSHIP_SIDE_ERROR_SYSTEM_PROMPT},
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
            reasoning_steps = extract_python_list(generated_text)
            
            result_entry = row.copy()
            # 생성된 추론 과정을 'generated_kinship_error_reasoning' 키에 저장
            result_entry["generated_kinship_error_reasoning"] = reasoning_steps
            new_results.append(result_entry)

        # Save
        final_results.extend(new_results)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"🎉 All Completed. Total items saved: {len(final_results)}")
    print(f"📂 Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()