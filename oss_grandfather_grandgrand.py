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
OUTPUT_FILE = "/workspace/daeyong/fourth_finetuning_data/grandfather_grandgrand_generated.json"

KINSHIP_LOGIC_ERROR_SYSTEM_PROMPT = """You are a data generation assistant designed to create specific reasoning error examples for training feedback models.
Your task is to generate a chain of reasoning steps that demonstrates a **"Logical Fallacy"**.

**Input:**
- A question asking for a Grandparent (e.g., "Who is the paternal grandfather of X?").
- Retrieved passages describing the lineage (Child -> Parent -> Grandparent).

**Output:**
- A python list of strings, representing the reasoning steps.

**Strict Generation Rules:**
1.  **Attribution Steps (Correct):**
    - Correctly identify the **Parent** of the subject (e.g., "Step 1: According to Passage K, ... X is the son/daughter of Y.").
    - Correctly identify the **Grandparent** if mentioned (e.g., "Step 2: According to Passage K, ... Y is the son/daughter of Z.").
    In the Attribution Step, you have to specify the exact passage number as Passage K.
2.  **The Flawed Logical Step (The Error):**
    - Create a logical step that **confuses the Parent with the Grandparent**.
    - Claim that because Y is the parent of X, Y is the *Grandparent* of X. (Ignoring the definition of Grandparent).
    - Format: `Step N: Since [Parent] is the father/mother of [Child], [Parent] is the [paternal/maternal grandfather/grandmother] of [Child]. (Logical)`

**Template Examples:**

**Example 1:**
Question: Who is the paternal grandfather of Birger Brosa?
Retrieved Passages:
Passage 1: Birger Brosa was the son of Bengt Snivil.
Passage 2: Bengt was the son of Folke the Fat.

**Your Output:**
[
"Step 1: According to Passage 1, Birger Brosa is the son of Bengt Snivil. (Attribution)",
"Step 2: According to Passage 2, Bengt Snivil is the son of Folke the Fat. (Attribution)",
"Step 3: Since Bengt Snivil is the father of Birger Brosa, Bengt Snivil is the paternal grandfather of Birger Brosa. (Logical)"
]

**Example 2:**
Question: Who is the maternal grandmother of Princess Marie?
Retrieved Passages:
Passage 1: Princess Marie is the daughter of Princess Victoria.
Passage 2: Victoria is the daughter of Queen Alexandra.

**Your Output:**
[
"Step 1: According to Passage 1, Princess Marie is the daughter of Princess Victoria. (Attribution)",
"Step 2: According to Passage 2, Princess Victoria is the daughter of Queen Alexandra. (Attribution)",
"Step 3: Since Princess Victoria is the mother of Princess Marie, Princess Victoria is the maternal grandmother of Princess Marie. (Logical)"
]

**Example 3:**
Question: Who is the paternal grandmother of Sir Robert Long?
Retrieved Passages:
Passage 1: Sir Robert Long was the son of Sir Walter Long.
Passage 2: Walter's mother was Mary Cocks.

**Your Output:**
[
"Step 1: According to Passage 1, Sir Robert Long is the son of Sir Walter Long. (Attribution)",
"Step 2: According to Passage 2, the mother of Sir Walter Long is Mary Cocks. (Attribution)",
"Step 3: Since Sir Walter Long is the father of Sir Robert Long, Sir Walter Long is the paternal grandmother of Sir Robert Long. (Logical)"
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
                {"role": "system", "content": KINSHIP_LOGIC_ERROR_SYSTEM_PROMPT},
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