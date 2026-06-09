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
INPUT_FILE = "/workspace/daeyong/fourth_finetuning_data/final_swap_generated_2.json"
OUTPUT_FILE = "/workspace/daeyong/fourth_finetuning_data/final_swap_feedback_2.json"

FEEDBACK_SYSTEM_PROMPT = """You are an expert AI evaluator providing feedback on multi-hop reasoning steps.
Your task is to evaluate the **Final Step** of a reasoning chain where the user deliberately makes a "Wrong Conclusion".

**Task:**
Analyze the provided Question, Passages, and the full Reasoning Chain.
Focus on the **Final Step** (the answer step). Identify that it contradicts the logical conclusion derived in the immediately preceding steps.

**Requirements:**
1.  **error_type**: Use "Wrong Conclusion".
2.  **diagnosis**: Clearly explain the mismatch. Mention that the logic established X (e.g., Director A is younger), but the final answer chose Y (Film linked to Director B).
3.  **guidance**: Provide the specific correct mapping. State which entity should have been chosen based on the logic.

**Output Format:**
Provide ONLY a valid JSON object:
{
  "error_type": "Wrong Conclusion",
  "diagnosis": "...",
  "guidance": "..."
}

---
**Examples:**

Example 1:
Question: Which film whose director is younger, The Devil (1915 Film) or Minor Mishaps?

Retrieved Passages:
Passage 1: The Devil (1915 Film) was directed by Reginald Barker (born 1886).
Passage 2: Minor Mishaps was directed by Annette K. Olesen (born 1965).

Reasoning Chain:
Step 1: According to Passage 1, the director of The Devil is Reginald Barker. (Attribution)
Step 2: According to Passage 1, Reginald Barker was born in 1886. (Attribution)
Step 3: According to Passage 2, the director of Minor Mishaps is Annette K. Olesen. (Attribution)
Step 4: According to Passage 2, Annette K. Olesen was born in 1965. (Attribution)
Step 5: Since Annette K. Olesen was born in 1965 and Reginald Barker was born in 1886, Annette K. Olesen is the younger director. (Logical)
Step 6: ####ANSWER: The Devil (1915 Film) (Final Answer)

**Output:**
{
  "error_type": "Wrong Conclusion",
  "diagnosis": "The reasoning in Step 5 correctly identified that Annette K. Olesen is the younger director because she was born in 1965 (compared to 1886). However, the final answer incorrectly selected 'The Devil (1915 Film)', which was directed by the older director, Reginald Barker.",
  "guidance": "Since the logical conclusion is that Annette K. Olesen is younger, you must select the film directed by her. Generate final answer as ####ANSWER: Minor Mishaps (Final Answer)."
}

---

Example 2:
Question: Which film has the director who died earlier, Twist Of Fate or The Wakefield Case?

Retrieved Passages:
Passage 1: Twist Of Fate is directed by David Miller, who died in 1992.
Passage 2: The Wakefield Case is directed by George Brook, who died in 1974.

Reasoning Chain:
Step 1: According to Passage 1, David Miller directed Twist Of Fate and died in 1992. (Attribution)
Step 2: According to Passage 2, George Brook directed The Wakefield Case and died in 1974. (Attribution)
Step 3: Comparing the death years, 1974 is earlier than 1992. (Logical)
Step 4: Therefore, George Brook died earlier than David Miller. (Logical)
Step 5: ####ANSWER: Twist Of Fate (Final Answer)

**Output:**
{
  "error_type": "Wrong Conclusion",
  "diagnosis": "The reasoning correctly established in Step 4 that George Brook died earlier (1974 < 1992). However, the final answer incorrectly maps this conclusion to 'Twist Of Fate', which is David Miller's film.",
  "guidance": "You identified that George Brook died earlier. You must select the film directed by George Brook, which is 'The Wakefield Case'. Generate final answer as ####ANSWER: The Wakefield Case (Final Answer)."
}

---

Example 3:
Question: Which film has the director who was born later, Passkey to Danger or Monkey Shines?

Retrieved Passages:
Passage 1: Passkey to Danger was directed by Lesley Selander (born 1900).
Passage 2: Monkey Shines was directed by George A. Romero (born 1940).

Reasoning Chain:
Step 1: The director of Passkey to Danger is Lesley Selander (born 1900). (Attribution)
Step 2: The director of Monkey Shines is George A. Romero (born 1940). (Attribution)
Step 3: Since 1940 is a larger number than 1900, George A. Romero was born later. (Logical)
Step 4: ####ANSWER: Passkey to Danger (Final Answer)

**Output:**
{
  "error_type": "Wrong Conclusion",
  "diagnosis": "The logical step correctly determined that George A. Romero (born 1940) was born later than Lesley Selander (born 1900). The final answer, however, incorrectly points to 'Passkey to Danger', which belongs to the director born earlier.",
  "guidance": "Map the correct director to their film. Since George A. Romero was born later, the correct final answer is his film, 'Monkey Shines'. Generate final answer as ####ANSWER: Monkey Shines (Final Answer)."
}

---

Example 4:
Question: Who was born earlier, Funa Tonaki or Yeldos Smetov?

Retrieved Passages:
Passage 1: Funa Tonaki (born 1 August 1995) is a Japanese judoka.
Passage 2: Yeldos Smetov (born 9 September 1992) is a Kazakhstani judoka.

Reasoning Chain:
Step 1: According to Passage 1, Funa Tonaki was born on 1 August 1995. (Attribution)
Step 2: According to Passage 2, Yeldos Smetov was born on 9 September 1992. (Attribution)
Step 3: Since 1992 is earlier than 1995, Yeldos Smetov was born earlier than Funa Tonaki. (Logical)
Step 4: ####ANSWER: Funa Tonaki (Final Answer)

**Output:**
{
  "error_type": "Wrong Conclusion",
  "diagnosis": "The reasoning correctly identified that Yeldos Smetov was born earlier (1992 < 1995) in Step 3. However, the final answer incorrectly selected 'Funa Tonaki', who was born later.",
  "guidance": "Since Yeldos Smetov was born earlier, the correct final answer is 'Yeldos Smetov'. Generate final answer as ####ANSWER: Yeldos Smetov (Final Answer)."
}
""".strip()

# =============================================================================
# 2. Helper Functions
# =============================================================================
def extract_json(text):
    """
    Extracts JSON object from text.
    """
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

    # --- Load Data ---
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

    # 필수 컬럼 확인 ('generated_wrong_reasoning'이 있어야 함)
    if 'generated_wrong_reasoning' not in df_input.columns:
        print("❌ Error: Input file must contain 'generated_wrong_reasoning' column.")
        return

    # --- Resume Logic ---
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
        max_tokens=6000, 
    )

    # --- Batch Processing ---
    BATCH_SIZE = 100
    records = df_input.to_dict('records')
    final_results = existing_results

    print(f"🚀 Starting execution in batches of {BATCH_SIZE}...")

    for i in tqdm(range(0, len(records), BATCH_SIZE), desc="Processing Batches"):
        batch_records = records[i : i + BATCH_SIZE]
        batch_prompts = []
        valid_indices = [] # 유효한(reasoning이 있는) 레코드의 인덱스 추적

        for idx, row in enumerate(batch_records):
            question = row.get('question', '')
            passages = row.get('retrieved_passages', [])
            reasoning_steps = row.get('generated_wrong_reasoning', [])

            # Reasoning Steps가 리스트가 아니거나 비어있으면 스킵
            if not isinstance(reasoning_steps, list) or not reasoning_steps:
                continue
            
            # Passages 포맷팅
            if isinstance(passages, list):
                passages_text = "\n".join(passages)
            else:
                passages_text = str(passages)
            
            # Reasoning Steps 포맷팅
            reasoning_text = "\n".join(reasoning_steps)
            
            # 마지막 스텝 추출 (Evaluation 대상)
            final_step = reasoning_steps[-1]

            user_content = f"""Question: {question}
            
Retrieved Passages:
{passages_text}

Reasoning Chain:
{reasoning_text}
""".strip()

            messages = [
                {"role": "system", "content": FEEDBACK_SYSTEM_PROMPT},
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
            
            # JSON Parsing
            feedback_json = extract_json(generated_text)
            
            result_entry = row.copy()
            
            # 파싱 성공 여부에 따라 저장
            if isinstance(feedback_json, dict):
                result_entry["gold_feedback"] = feedback_json
            else:
                # 파싱 실패 시 fallback (텍스트 저장)
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