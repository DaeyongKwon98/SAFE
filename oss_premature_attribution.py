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

INPUT_FILE = "/workspace/daeyong/fourth_finetuning_data/premature_attribution.json"
OUTPUT_FILE = "/workspace/daeyong/fourth_finetuning_data/premature_attribution_generated.json"

PREMATURE_ATTRIBUTION_SYSTEM_PROMPT = """You are a data generation assistant designed to create specific reasoning error examples for training feedback models.
Your task is to generate a reasoning chain that demonstrates a **"Premature Attribution"** error.

**Input:**
- A multi-hop comparison question (e.g., "Which film's director died later?", "Who is older?", "Which song's performer was born first?").
- Retrieved passages containing the necessary information.

**Output:**
- A python list of strings, representing the reasoning steps.

**Strict Generation Rules:**
1.  **OMIT Linking Steps (The Error):** You must **NOT** generate steps that explicitly link the subject in the question to the target entity.
    - **FORBIDDEN:** "The director of Film A is [Person Name]."
    - **FORBIDDEN:** "The performer of Song B is [Artist Name]."
    - **FORBIDDEN:** "The father of [Person] is [Father Name]."
2.  **Immediate Attribution:** Start the reasoning directly by extracting the attribute (date, place, birth year, etc.) of the entities found in the passages, as if the connection is already known or obvious.
    - Format: `Step N: According to Passage X, [Person Name] died on [Date]. (Attribution)`
    - Format: `Step N: According to Passage Y, [Person Name] was born in [Place]. (Attribution)`

You must generate only the first single step. This generated step should reflect the premature attribution error by skipping the linking step.

**Template Examples:**

**Example 1 (Film Director - Death Date):**
User Question: Which film has the director who died later, "The Great Escape" or "The Birds"?
Passages:
Passage 1: "The Great Escape" is a 1963 film directed by John Sturges.
Passage 2: John Sturges died on August 18, 1992.
Passage 3: "The Birds" is a horror film directed by Alfred Hitchcock.
Passage 4: Alfred Hitchcock passed away on April 29, 1980.

**Your Output:**
[
"Step 1: According to Passage 2, John Sturges died on August 18, 1992. (Attribution)"
]
(Notice: The steps skip explicitly stating "The director of The Great Escape is John Sturges".)

**Example 2 (Song Performer - Birth Date):**
User Question: Who was born earlier, the performer of "Respect" or the performer of "Imagine"?
Passages:
Passage 1: "Respect" is a song recorded by Aretha Franklin.
Passage 2: Aretha Franklin was born on March 25, 1942.
Passage 3: "Imagine" is a song written and performed by John Lennon.
Passage 4: John Lennon was born on October 9, 1940.

**Your Output:**
[
"Step 1: According to Passage 2, Aretha Franklin was born on March 25, 1942. (Attribution)"
]

**Example 3 (Book Author - Age Comparison):**
User Question: Who is older, the author of "Harry Potter" or the author of "The Shining"?
Passages:
Passage 1: J.K. Rowling is the author of the Harry Potter series. She was born in 1965.
Passage 2: Stephen King wrote "The Shining". He was born in 1947.

**Your Output:**
[
"Step 1: According to Passage 1, J.K. Rowling was born in 1965. (Attribution)"
]
""".strip()

# =============================================================================
# 2. Helper Functions
# =============================================================================
def extract_python_list(text):
    if not isinstance(text, str):
        return []
    text = text.strip().split("assistantfinal")[-1].strip() # 템플릿에 따라 조정
    
    # 1. Markdown Code Block 제거
    match = re.search(r"```(?:python)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if match:
        content = match.group(1).strip()
    else:
        # 2. 그냥 대괄호 찾기
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

            user_content = f"Question: {question}\nPassages:\n{passages_text}"

            messages = [
                {"role": "system", "content": PREMATURE_ATTRIBUTION_SYSTEM_PROMPT},
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
            result_entry["generated_premature_reasoning"] = reasoning_steps
            new_results.append(result_entry)

        # Save
        final_results.extend(new_results)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"🎉 All Completed. Total items saved: {len(final_results)}")
    print(f"📂 Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()