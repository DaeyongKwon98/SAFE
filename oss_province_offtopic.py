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
INPUT_FILE = "/workspace/daeyong/fourth_finetuning_data/province_finished.json"
OUTPUT_FILE = "/workspace/daeyong/fourth_finetuning_data/province_offtopic_finished.json"

system_prompt = """You are an expert AI Data Synthesizer for Error Correction Training.
Your task is to generate a **Specific Reasoning Error** and its corresponding **Corrective Feedback** based on the provided context.

**SCENARIO:**
- The Question asks if two entities are in the **same country**.
- The provided `Previous Steps` correctly identify that both entities are in the **same country** but **different provinces/states/villages/etc**.

**YOUR TASK:**
1. **Generate a Flawed Logical Step (Step 3)**:
   - Create a reasoning step that **incorrectly focuses on the difference in Provinces/States/Villages/etc**.
   - It should compare the provinces (e.g., "Province A is not Province B") instead of comparing the countries.
   - It may typically lead to a wrong conclusion (e.g., "Therefore they are not in the same location") or simply state an irrelevant comparison.
2. **Generate Feedback for this Flawed Step**:
   - **error_type**: "Off-topic"
   - **diagnosis**: Explain that the step compares administrative regions (provinces) which is irrelevant because the question asks about the **country**.
   - **guidance**: Explicitly instruct to compare the **Countries** of the two entities, ignoring the province differences.

**INPUT:**
- Question
- Retrieved Passages
- Previous Steps (Correct Attributions)

**OUTPUT FORMAT (JSON ONLY):**
{
    "flawed_step": "Step 3: Since [Entity A] is in [Province A] and [Entity B] is in [Province B], and these are different provinces, they are not located in the same region. (Logical)",
    "feedback": {
        "error_type": "Off-topic",
        "diagnosis": "This step compares the provinces of the two entities, but the question specifically asks about the country. Comparing provinces is irrelevant to the core question.",
        "guidance": "Compare the country of [Entity A] and [Entity B] directly to check whether they are located in the same country. Do not focus on the difference in provinces."
    }
}
""".strip()

# =============================================================================
# 2. Helper Functions
# =============================================================================
def extract_json(text):
    """
    Extracts a JSON object from the text. 
    Handles cases with markdown code blocks or extraneous text.
    """
    if not isinstance(text, str):
        return None
    
    # Clean up standard markdown json wrappers
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").split("assistantfinal")[-1].strip()
    
    # Try to find the outermost curly braces
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        json_str = match.group(0).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass # Fallback to trying the whole text or other methods if needed
            
    # Try parsing the text directly if regex didn't work or failed to decode
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

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
    
    df_input['previous_steps'] = df_input.apply(lambda row: row['previous_steps'] + [row['current_step']], axis=1)

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
            previous_steps = row.get('previous_steps', [])
            
            # Passages Text Formatting
            if isinstance(passages, list):
                passages_text = "\n".join([f"Passage {idx+1}: {p}" for idx, p in enumerate(passages)])
            elif isinstance(passages, str):
                try:
                    passages_list = eval(passages)
                    passages_text = "\n".join([f"Passage {idx+1}: {p}" for idx, p in enumerate(passages_list)])
                except:
                    passages_text = passages
            else:
                passages_text = str(passages)

            user_content = f"Question: {question}\nPassages:\n{passages_text}\nPrevious Steps: {previous_steps}"

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
            
            # Extract JSON
            parsed_json = extract_json(generated_text)
            
            result_entry = row.copy()
            
            # Unpack the JSON structure into specific columns
            if parsed_json:
                result_entry["current_step"] = parsed_json.get("flawed_step", [])
                
                feedback_obj = parsed_json.get("feedback", {})
                result_entry["error_type"] = feedback_obj.get("error_type", None)
                result_entry["diagnosis"] = feedback_obj.get("diagnosis", None)
                result_entry["guidance"] = feedback_obj.get("guidance", None)
            else:
                # Fallback if parsing failed
                result_entry["current_step"] = ''
                result_entry["error_type"] = None
                result_entry["diagnosis"] = None
                result_entry["guidance"] = None
                
            # Optionally save raw output for debugging
            result_entry["raw_output"] = generated_text
            
            new_results.append(result_entry)

        # Save
        final_results.extend(new_results)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"🎉 All Completed. Total items saved: {len(final_results)}")
    print(f"📂 Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()