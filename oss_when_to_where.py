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

INPUT_FILE = "/workspace/daeyong/fourth_finetuning_data/when_to_where.json"
OUTPUT_FILE = "/workspace/daeyong/fourth_finetuning_data/when_to_where_generated.json"

WHEN_MISMATCH_SYSTEM_PROMPT = """You are a data generation assistant designed to create specific reasoning error examples for training feedback models.
Your task is to generate a chain of reasoning steps that demonstrates an **"Off-topic"** error.

**Input:**
- A question asking for a **Time/Date** ("When was X born?", "When did X die?", "What year...").
- Retrieved passages containing both dates and locations.

**Output:**
- A python list of strings, representing the reasoning steps.

**Strict Generation Rules:**
1.  **Identity Step (Correct):** Correctly identify the subject (Director, Composer, Father, etc.) from the passages.
    - Format: `Step 1: According to Passage X, the [Role] of [Work] is [Person Name]. (Attribution)`
2.  **Extraction Step (The Error):** Instead of extracting the *Date/Year*, deliberately extract the **Place/Location** (City, Country) of the event.
    - Format: `Step 2: According to Passage Y, [Person Name] was born/died in [City/Country]. (Attribution)`

**CRITICAL:** If there are no information about locations (city, country, etc.) in the passages, do not generate any steps. Instead, return an empty list: `[]`.

**Template Examples:**

**Example 1:**
User Question: When was the director of film "Parasite" born?
Passages: 
Passage 1: Bong Joon-ho (born September 14, 1969 in Daegu, South Korea) is a South Korean film director.

**Your Output:**
[
"Step 1: According to Passage 1, the director of the film Parasite is Bong Joon-ho. (Attribution)",
"Step 2: According to Passage 1, Bong Joon-ho was born in Daegu, South Korea. (Attribution)"
]

**Example 2:**
User Question: When did the star of "Jaws" die?
Passages: 
Passage 1: Steven Spielberg directed Jaws.
Passage 2: Roy Scheider (November 10, 1932 – February 10, 2008) starred in Jaws. He died in Little Rock, Arkansas.

**Your Output:**
[
"Step 1: According to Passage 2, the star of the film Jaws is Roy Scheider. (Attribution)",
"Step 2: According to Passage 2, Roy Scheider died in Little Rock, Arkansas. (Attribution)"
]

**Example 3:**
User Question: When was the performer of song "Thriller" born?
Passages: 
Passage 1: Michael Jackson (August 29, 1958 – June 25, 2009) was an American singer, songwriter, and dancer. He was born in Gary, Indiana.

**Your Output:**
[
"Step 1: According to Passage 1, the performer of the song Thriller is Michael Jackson. (Attribution)",
"Step 2: According to Passage 1, Michael Jackson was born in Gary, Indiana. (Attribution)"
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

            user_content = f"Question: {question}\nPassages:\n{passages_text}"

            messages = [
                {"role": "system", "content": WHEN_MISMATCH_SYSTEM_PROMPT},
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
            # 생성된 추론 과정을 'generated_when_mismatch_reasoning' 키에 저장
            result_entry["generated_when_mismatch_reasoning"] = reasoning_steps
            new_results.append(result_entry)

        # Save
        final_results.extend(new_results)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"🎉 All Completed. Total items saved: {len(final_results)}")
    print(f"📂 Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()