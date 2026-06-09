system_prompt_musique = """You are an expert AI assistant specializing in multi-hop reasoning. 
Your task is to generate the ideal, step-by-step reasoning that correctly follows a given reasoning plan, using only the provided ground truth context.

**Instructions:**
1. You will be given a `Question`, a `Plan` (as a list of instructions), and a `Ground Truth Context`.
2. Your goal is to "execute" the `Plan` to generate the final reasoning steps.
3. You must generate exactly one reasoning step for each instruction in the `Plan`.
4. **CRITICAL (Context):** Your reasoning *must* be based *only* on the facts provided in the `Ground Truth Context`. Do not add any external knowledge or information.
5. **CRITICAL (Citation):** For all `(Attribution)` steps, you **must** explicitly cite the passage number you used ("According to Passage 1, ...").
6. **CRITICAL (No Generic Citations):** You **MUST NOT** use generic phrases like "According to the ground truth context" or "According to the passages". Always cite the *specific* passage number.
7. **CRITICAL (Tags):** Each reasoning step you generate *must* end with the exact `(Attribution)` or `(Logical)` tag that appears in the corresponding plan step.
8. **CRITICAL (Dependencies):** When a plan step has a dependency (e.g., "...from Step 1"), your reasoning step must clearly show this. For example, if Step 1 found "Bengt Snivil", Step 2's reasoning should be "The father of Bengt Snivil (from Step 1) is..."
9. **CRITICAL (Format):** You MUST output **only** the list of steps in the specified format: `[Step 1: ..., Step 2: ..., ...]`. Do not include *any* other text, JSON formatting, explanations, or conversational chat before or after the list.

---

**Examples:**

Question: What is the the primary genre of the record label that has the performer of Keep the Faith?

Plan:
[Step 1: Find the performer of "Keep the Faith". (Attribution), 
 Step 2: Find the record label for the performer found in Step 1. (Attribution), 
 Step 3: Find the primary genre of the record label found in Step 2. (Attribution), 
 Step 4: Identify the genre found in Step 3 as the answer. (Logical)]

Ground Truth Context: 
Passage 1: The song "Keep the Faith" was released by the artist Bon Jovi. 
Passage 2: Island Records' primary genre is widely recognized as reggae, though it also signs artists in rock and pop.
Passage 3: Bon Jovi was signed to Island Records for much of their career.

Output: 
[Step 1: According to Passage 1, the performer of "Keep the Faith" is Bon Jovi. (Attribution), 
 Step 2: According to Passage 3, the record label for Bon Jovi (from Step 1) is Island Records. (Attribution), 
 Step 3: According to Passage 2, the primary genre of Island Records (from Step 2) is reggae. (Attribution), 
 Step 4: The genre found in Step 3 is reggae. (Logical)]
 
---
 
Question: Who was the mother of the person under whom the colonizer in the 1st century BC of Ahmed Temsah's country reached its greatest extent?

Plan:
[Step 1: Find the country Ahmed Temsah is from. (Attribution), 
 Step 2: Find what the country from Step 1 became a colony of in the 1st century BC. (Attribution), 
 Step 3: Find the person under whom the entity from Step 2 reached its greatest extent. (Attribution), 
 Step 4: Find the mother of the person found in Step 3. (Attribution), 
 Step 5: Identify the mother found in Step 4 as the answer. (Logical)]

Ground Truth Context: 
Passage 1: Ahmed Temsah is a noted scholar from Egypt. 
Passage 2: In the 1st century BC, Egypt (from Step 1) was annexed as a colony by the Roman Empire. 
Passage 3: Trajan's mother was a noblewoman named Marcia.
Passage 4: The Roman Empire achieved its greatest territorial extent under the rule of the emperor Trajan. 

Output: 
[Step 1: According to Passage 1, the country Ahmed Temsah is from is Egypt. (Attribution), 
 Step 2: According to Passage 2, Egypt (from Step 1) became a colony of the Roman Empire in the 1st century BC. (Attribution), 
 Step 3: According to Passage 4, the Roman Empire (from Step 2) reached its greatest extent under Trajan. (Attribution), 
 Step 4: According to Passage 3, the mother of Trajan (from Step 3) is Marcia. (Attribution), 
 Step 5: The mother found in Step 4 is Marcia. (Logical)]
 
---

Question: Where did the pizza style of the city that shares a border with Al Herman's place of death come from?

Plan: 
[Step 1: Find the place where Al Herman died. (Attribution), 
 Step 2: Find the city that shares a border with the place from Step 1. (Attribution), 
 Step 3: Find where the pizza style of the city from Step 2 originated from. (Attribution),
 Step 4: Identify the origin place found in Step 3 as the answer. (Logical)]

Ground Truth Context: 
Passage 1: The famous New Haven-style pizza originated at Frank Pepe Pizzeria Napoletana, which was founded by immigrants from Naples.
Passage 2: West Haven shares a border with the city of New Haven. 
Passage 3: Racing driver Al Herman died in West Haven, Connecticut.

Output: 
[Step 1: According to Passage 3, the place where Al Herman died is West Haven. (Attribution), 
 Step 2: According to Passage 2, the city that shares a border with West Haven (from Step 1) is New Haven. (Attribution), 
 Step 3: According to Passage 1, the pizza style of New Haven (from Step 2) originated from Naples. (Attribution), 
 Step 4: The origin place found in Step 3 is Naples. (Logical)]
""".strip()


import pandas as pd
from tqdm import tqdm
import json
import os
import re
import ast
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import argparse

# =======================================================
# 1. Helpers
# =======================================================
def extract_step_list(text: str) -> str:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        return m.group(0).strip()

    # Wrap in brackets if steps exist but no brackets
    steps = re.findall(r"Step\s*\d+:.*", text)
    if steps:
        return "[" + ", ".join(s.strip() for s in steps) + "]"

    return text.strip()

def safe_parse_plan(plan_str: str):
    """
    Parses the step string generated by the LLM into a list.
    Handles both comma and newline separators.
    """
    plan_str = plan_str.strip()
    
    # Extract content inside brackets
    if plan_str.startswith('[') and plan_str.endswith(']'):
        inner = plan_str[1:-1].strip()
    else:
        inner = plan_str

    if not inner:
        return []
    
    steps_raw = re.findall(r"(Step\s*\d+:.*?)(?=Step\s*\d+:|$)", inner, re.DOTALL)
    
    return [s.strip() for s in steps_raw if s.strip()]

# =======================================================
# 2. Main Execution
# =======================================================

def main(args):
    # ✅ Constants & Paths
    MODEL_NAME = "/workspace/hf_transformers/gpt-oss-120b"
    
    if args.dataset == "2wiki":
        CSV_PATH = "/workspace/daeyong/benchmarks/2wiki_20k_sample.csv"  # Q, Context
        PLAN_PATH = "/workspace/daeyong/reasoning_plans/2wiki_20k_sample_plan.json"  # Q, Plan
        RESULT_PATH = "/workspace/daeyong/ideal_steps/2wiki_20k_sample_ideal_steps.json"  # Output
        system_prompt = system_prompt_2wiki
    elif args.dataset == "hotpotqa":
        CSV_PATH = "/workspace/daeyong/benchmarks/hotpotqa_20k_sample.csv"  # Q, Context
        PLAN_PATH = "/workspace/daeyong/reasoning_plans/hotpotqa_20k_sample_plan.json"  # Q, Plan
        RESULT_PATH = "/workspace/daeyong/ideal_steps/hotpotqa_20k_sample_ideal_steps.json"  # Output
        system_prompt = system_prompt_hotpotqa
    elif args.dataset == "musique":
        CSV_PATH = "/workspace/daeyong/benchmarks/musique.csv"  # Q, Context
        PLAN_PATH = "/workspace/daeyong/reasoning_plans/musique_plan.json"  # Q, Plan
        RESULT_PATH = "/workspace/daeyong/ideal_steps/musique_20k_ideal_steps.json"  # Output
        system_prompt = system_prompt_musique
    
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)

    # ✅ 1. Initialize vLLM
    print(f"🚀 Loading vLLM model: {MODEL_NAME}")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=4,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        max_model_len=4096,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=4000,
    )
    
    # Load tokenizer for chat template application
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ✅ 2. Load Data
    print("📂 Loading data...")
    # Load Context CSV
    try:
        df_csv = pd.read_csv(CSV_PATH)
        if 'question' not in df_csv.columns or 'gt_context' not in df_csv.columns:
            raise ValueError("CSV must contain 'question' and 'gt_context' columns.")
        # Create a map for fast lookup
        csv_map = {row["question"].strip(): row["gt_context"].strip() for _, row in df_csv.iterrows()}
    except Exception as e:
        print(f"⚠️ Error loading {CSV_PATH}: {e}")
        return

    # Load Plan JSON
    try:
        with open(PLAN_PATH, "r") as f:
            plan_data = json.load(f)
    except Exception as e:
        print(f"⚠️ Error loading {PLAN_PATH}: {e}")
        return

    # ✅ 3. Resume Logic
    if os.path.exists(RESULT_PATH):
        with open(RESULT_PATH, "r") as f:
            existing_results = json.load(f)
        
        # Valid results have matching step counts
        valid_results = []
        processed_questions = set()
        
        for r in existing_results:
            if "plan" in r and "ideal_steps" in r:
                # Basic validation: Check if step count matches or if ERROR occurred
                if len(r["plan"]) == len(r["ideal_steps"]) or any("ERROR:" in s for s in r["ideal_steps"]):
                    valid_results.append(r)
                    processed_questions.add(r["question"])
        
        print(f"🔄 Resuming... Found {len(existing_results)} total, {len(valid_results)} valid.")
        results = valid_results
    else:
        results, processed_questions = [], set()

    # Filter targets
    # We iterate over plan_data as the source of truth for items to process
    targets = [item for item in plan_data if item["question"].strip() not in processed_questions]
    print(f"🎯 Total items to process: {len(targets)}")

    if not targets:
        print("✅ No new items to process.")
        return

    # ✅ 4. Batch Inference Loop
    BATCH_SIZE = 500  # Adjust based on memory
    
    # Process in chunks
    for i in tqdm(range(0, len(targets), BATCH_SIZE), desc="Processing Batches"):
        batch_items = targets[i : i + BATCH_SIZE]
        batch_prompts = []
        batch_metadata = [] # Store metadata to map outputs back to items

        # --- Prepare Prompts for Batch ---
        for item in batch_items:
            question = item["question"].strip()
            plan_list = item["plan"]
            
            # Retrieve Context
            context_str = csv_map.get(question)
            if not context_str:
                # If context missing, skip this item but keep index alignment safely
                # (Ideally log this)
                continue
            
            context_str = context_str.strip()

            # Parse Context List
            try:
                context_list = ast.literal_eval(context_str)
                if not isinstance(context_list, list):
                    context_list = [str(context_list)]
            except Exception:
                context_list = [context_str]

            # Format Context for Prompt
            formatted_context_lines = [f"Passage {idx}: {txt.strip()}" for idx, txt in enumerate(context_list, 1)]
            formatted_context = "\n".join(formatted_context_lines)
            
            # Format Plan
            plan_str = "\n".join(plan_list)

            # Construct Prompt
            user_content = f"""
Question: {question}

Plan:
{plan_str}

Ground Truth Context:
{formatted_context}
"""
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
            batch_metadata.append({
                "question": question,
                "plan": plan_list,
                "gt_context": context_list
            })

        if not batch_prompts:
            continue

        # --- Run Inference ---
        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)

        # --- Process Outputs ---
        new_results = []
        for meta, output in zip(batch_metadata, outputs):
            generated_text = output.outputs[0].text.split("assistantfinal")[-1].strip()
            
            try:
                steps_str = extract_step_list(generated_text)
                steps_list = safe_parse_plan(steps_str)
                
                # Validation: Check step count mismatch
                if len(steps_list) != len(meta["plan"]):
                    # Log mismatch and don't save
                    print(f"⚠️ Mismatch Q={meta['question'][:30]}... Plan:{len(meta['plan'])} vs Gen:{len(steps_list)}")
                    continue
                    
            except Exception as e:
                print(f"⚠️ Parsing failed for Q={meta['question'][:30]}...: {e}")
                steps_list = [f"ERROR: {e}"]

            # Append result
            new_results.append({
                "question": meta["question"],
                "gt_context": meta["gt_context"],
                "plan": meta["plan"],
                "ideal_steps": steps_list
            })

        # --- Save Intermediate ---
        results.extend(new_results)
        with open(RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Completed. Total items saved: {len(results)}")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Generate Ideal Reasoning Steps using vLLM")
    arg_parser.add_argument("--dataset", type=str, choices=["2wiki", "hotpotqa", "musique"], required=True)
    args = arg_parser.parse_args()
    main(args)