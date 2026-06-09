import pandas as pd
from tqdm import tqdm
import json
import os
import re
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from typing import List, Dict, Any, Tuple
import random

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

# system_prompt = """You are an expert in logical reasoning, tasked with intentionally introducing a specific logical error into a reasoning steps.

# Your Goal: Replace a single, correct reasoning step with a 'Contradictory' error.

# Error Definition: 'Contradictory'
# An 'Contradictory' error is an (Attribution) step that makes a factual claim that **directly conflicts** or **contradicts** information explicitly stated in the `Retrieved Passages`.
# - The new step's content (the "fact") MUST be **demonstrably false** based on the provided passages (e.g., if Passage 1 says "X is in France," the error MUST say "X is in Germany").
# - The topic of the step (e.g., a person's name, a location, a date) MUST be present in the passages, but the *value* or *fact* about it MUST be wrong.
# - This error MUST NOT be 'Unsupported' (i.e., do not invent a fact about something *not* mentioned at all).
# - The new step MUST still look like a plausible attribution (e.g., "According to Passage X...").
# - MUST maintain the original step's `(Attribution)` label.
# - MUST NOT be a repetition of a previous step (that is a Redundancy error).

# Input Format:
# You will receive:
# 1. Question: The user's original question.
# 2. Retrieved Passages: Contextual information.
# 3. Ideal Reasoning Steps: The correct, multi-step reasoning.
# 4. Target Step to Corrupt: The specific (Attribution) step from the ideal steps that you must replace.

# Output Format:
# - You MUST output only the single, new, erroneous reasoning step.
# - The new step MUST be formatted exactly like the target step, including the "Step X:" prefix and the "(Label)" suffix.

# ---
# EXAMPLES
# ---

# Question: "What was the first EP of the singer who recorded What Ifs?"

# Retrieved Passages:
# "Passage 1: \"What Ifs\" is a song recorded by American country music singer Kane Brown for his self-titled debut album..."
# "Passage 2: Kane Brown: Kane Allen Brown (born October 21, 1993) is an American country music singer and songwriter... He released his first EP, titled \"Closer\", in June 2015..."

# Ideal Reasoning Steps:
# [
#  "Step 1: According to Passage 1, the singer who recorded \"What Ifs\" is Kane Brown. (Attribution)",
#  "Step 2: According to Passage 2, the first EP of Kane Brown (from Step 1) is titled \"Closer\". (Attribution)",
#  "Step 3: Therefore, the EP found in Step 2, \"Closer\", is the answer. (Logical)"
# ]

# Target Step to Corrupt:
# Step 2

# Output:
# Step 2: According to Passage 2, the first EP of Kane Brown (from Step 1) is titled "Wide open". (Attribution)

# ---

# Question: "What Bengali political film was censored by one of the most powerful film censor boards in the world?"

# Retrieved Passages: 
# "Passage 1: Central Board of Film Certification: ... (CBFC) ... considered to be one of the most powerful film censor boards in the world..."
# "Passage 2: Kangal Malsat: Kangal Malsat ... is a Bengali political film ... based on the novel with same title written by Nabarun Bhattacharya. ... the Central Board of Film Certification denied approval to the film..."

# Ideal Reasoning Steps: 
# [ 
#  "Step 1: According to Passage 1, the Central Board of Film Certification (CBFC) is one of the most powerful film censor boards in the world. (Attribution)", 
#  "Step 2: According to Passage 2, the Bengali political film 'Kangal Malsat' was censored by the Central Board of Film Certification (from Step 1). (Attribution)", 
#  "Step 3: Therefore, the film found in Step 2, 'Kangal Malsat', is the answer. (Logical)" 
# ]

# Target Step to Corrupt: 
# Step 2

# Output: 
# Step 2: According to Passage 2, the Central Board of Film Certification (from Step 1) immediately approved the film 'Kangal Malsat'. (Attribution)

# ---

# Question: "Are David Nixon and Charlie Chaplin from the same country originally?"

# Retrieved Passages: 
# "Passage 1: David Nixon (director): David Nixon is an American film director and film producer."
# "Passage 2: Charlie Chaplin: Sir Charles Spencer 'Charlie' Chaplin... was an English comic actor, filmmaker, and composer..."

# Ideal Reasoning Steps: 
# [ 
#  "Step 1: According to Passage 1, David Nixon is an American film director. (Attribution)", 
#  "Step 2: According to Passage 2, Charlie Chaplin was an English comic actor. (Attribution)", 
#  "Step 3: Based on Step 1 (American) and Step 2 (English), their countries of origin are not the same. (Logical)" 
# ]

# Target Step to Corrupt: 
# Step 1

# Output: 
# Step 1: According to Passage 1, David Nixon is an German film director. (Attribution)

# ---

# Question: "Who is the paternal grandfather of James Tuchet, 3rd Earl of Castlehaven?"

# Retrieved Passages: 
# "Passage 1: James Tuchet, 3rd Earl of Castlehaven (c. 1617 – 11 October 1684) was the son of Mervyn Tuchet, 2nd Earl of Castlehaven..." 
# "Passage 2: Mervyn Tuchet, 2nd Earl of Castlehaven (1593 – 14 May 1631)... A son of George Tuchet, 1st Earl of Castlehaven and 11th Baron Audley..."

# Ideal Reasoning Steps: 
# [ 
#  "Step 1: According to Passage 1, the father of James Tuchet, 3rd Earl of Castlehaven is Mervyn Tuchet, 2nd Earl of Castlehaven. (Attribution)", 
#  "Step 2: According to Passage 2, the father of Mervyn Tuchet, 2nd Earl of Castlehaven (from Step 1) is George Tuchet, 1st Earl of Castlehaven. (Attribution)", 
#  "Step 3: Therefore, the person found in Step 2, George Tuchet, 1st Earl of Castlehaven, is the paternal grandfather of James Tuchet, 3rd Earl of Castlehaven. (Logical)" 
# ]

# Target Step to Corrupt: 
# Step 2

# Output: 
# Step 2: According to Passage 1, the father of Mervyn Tuchet, 2nd Earl of Castlehaven (from Step 1) is James Tuchet, 3rd Earl of Castlehaven. (Attribution)

# ---

# Question: "Who lived longer, Ignace Matondo Kwa Nzambi or Claudia Zobel?"

# Retrieved Passages: 
# "Passage 1: Ignace Matondo Kwa Nzambi (12 April 1932 – 9 September 2011) was a Congolese politician..." 
# "Passage 2: Claudia Zobel (February 27, 1964 – February 10, 1984) was a Filipina actress..."

# Ideal Reasoning Steps: 
# [ 
#  "Step 1: According to Passage 1, Ignace Matondo Kwa Nzambi was born on 12 April 1932 and died on 9 September 2011. (Attribution)", 
#  "Step 2: Based on the dates from Step 1, Ignace Matondo Kwa Nzambi lived for 79 years. (Logical)", 
#  "Step 3: According to Passage 2, Claudia Zobel was born on February 27, 1964 and died on February 10, 1984. (Attribution)", 
#  "Step 4: Based on the dates from Step 3, Claudia Zobel lived for 19 years. (Logical)", 
#  "Step 5: Comparing the lifespans from Step 2 (79 years) and Step 4 (19 years), Ignace Matondo Kwa Nzambi lived longer. (Logical)" 
# ]

# Target Step to Corrupt: 
# Step 3

# Output: 
# Step 3: According to Passage 2, Claudia Zobel was born on February 27, 1952. (Attribution)
# """.strip()

system_prompt = """You are an expert in logical reasoning, tasked with intentionally introducing a specific logical error into a reasoning steps.

Your Goal: Replace a single, correct reasoning step with a 'Contradictory' error.

Error Definition: 'Contradictory'
A 'Contradictory' error is an (Attribution) step that makes a factual claim that **directly conflicts** with information explicitly stated in the `Retrieved Passages`.
To create a Hard Negative, do not just invent random falsehoods. Instead, try to **swap entities, dates, or relations** found within the retrieved passages to make the error look plausible but demonstrably false.

Your generated step must reflect one of the following behaviors:
1. **Entity Swap:** Attribute an action or property to the *wrong person/place* mentioned in the same or other passages (e.g., If Passage says "A directed X," claim "B directed X" where B is another person in the retrieved passages).
2. **Relation Flip:** Reverse the relationship between two entities (e.g., If "A is the father of B," claim "B is the father of A").
3. **Date/Number Shift:** Change a specific date or number mentioned in the text to a different one (preferably one that appears elsewhere in the text, or is close to the original).
4. **Direct Negation:** State the opposite of a clear action (e.g., "approved" instead of "denied").

Constraints:
- The new claim MUST be **explicitly contradicted** by the passages. Do not create 'Unsupported' claims about things not mentioned.
- MUST maintain the original step's `(Attribution)` label.
- MUST NOT be a repetition of a previous step.
- The tone must remain confident and cite the passage ("According to Passage X, ...").

Input Format:
You will receive:
1. Question: The user's original question.
2. Retrieved Passages: Contextual information.
3. Ideal Reasoning Steps: The correct, multi-step reasoning.
4. Target Step to Corrupt: The specific (Attribution) step from the ideal steps that you must replace.

Output Format:
- You MUST output only the single, new, erroneous reasoning step.
- The new step MUST be formatted exactly like the target step, including the "Step X:" prefix and the "(Label)" suffix.

---
EXAMPLES
---

Question: "What was the first EP of the singer who recorded What Ifs?"

Retrieved Passages:
"Passage 1: \"What Ifs\" is a song recorded by American country music singer Kane Brown for his self-titled debut album..."
"Passage 2: Kane Brown: Kane Allen Brown (born October 21, 1993) is an American country music singer and songwriter... He released his first EP, titled \"Closer\", in June 2015..."

Ideal Reasoning Steps:
[
 "Step 1: According to Passage 1, the singer who recorded \"What Ifs\" is Kane Brown. (Attribution)",
 "Step 2: According to Passage 2, the first EP of Kane Brown (from Step 1) is titled \"Closer\". (Attribution)",
 "Step 3: Therefore, the EP found in Step 2, \"Closer\", is the answer. (Logical)"
]

Target Step to Corrupt:
Step 2

Output:
Step 2: According to Passage 2, the first EP of Kane Brown (from Step 1) is titled "What Ifs". (Attribution)

---

Question: "What Bengali political film was censored by one of the most powerful film censor boards in the world?"

Retrieved Passages: 
"Passage 1: Central Board of Film Certification: ... (CBFC) ... considered to be one of the most powerful film censor boards in the world..."
"Passage 2: Kangal Malsat: Kangal Malsat ... is a Bengali political film ... based on the novel with same title written by Nabarun Bhattacharya. ... the Central Board of Film Certification denied approval to the film..."

Ideal Reasoning Steps: 
[ 
 "Step 1: According to Passage 1, the Central Board of Film Certification (CBFC) is one of the most powerful film censor boards in the world. (Attribution)", 
 "Step 2: According to Passage 2, the Bengali political film 'Kangal Malsat' was censored by the Central Board of Film Certification (from Step 1). (Attribution)", 
 "Step 3: Therefore, the film found in Step 2, 'Kangal Malsat', is the answer. (Logical)" 
]

Target Step to Corrupt: 
Step 2

Output: 
Step 2: According to Passage 2, the Central Board of Film Certification (from Step 1) actually granted full approval to the film 'Kangal Malsat'. (Attribution)
---

Question: "Are David Nixon and Charlie Chaplin from the same country originally?"

Retrieved Passages: 
"Passage 1: David Nixon (director): David Nixon is an American film director and film producer."
"Passage 2: Charlie Chaplin: Sir Charles Spencer 'Charlie' Chaplin... was an English comic actor, filmmaker, and composer..."

Ideal Reasoning Steps: 
[ 
 "Step 1: According to Passage 1, David Nixon is an American film director. (Attribution)", 
 "Step 2: According to Passage 2, Charlie Chaplin was an English comic actor. (Attribution)", 
 "Step 3: Based on Step 1 (American) and Step 2 (English), their countries of origin are not the same. (Logical)" 
]

Target Step to Corrupt: 
Step 1

Output: 
Step 1: According to Passage 1, David Nixon is described as an English film director. (Attribution)
---

Question: "Who is the paternal grandfather of James Tuchet, 3rd Earl of Castlehaven?"

Retrieved Passages: 
"Passage 1: James Tuchet, 3rd Earl of Castlehaven (c. 1617 – 11 October 1684) was the son of Mervyn Tuchet, 2nd Earl of Castlehaven..." 
"Passage 2: Mervyn Tuchet, 2nd Earl of Castlehaven (1593 – 14 May 1631)... A son of George Tuchet, 1st Earl of Castlehaven and 11th Baron Audley..."

Ideal Reasoning Steps: 
[ 
 "Step 1: According to Passage 1, the father of James Tuchet, 3rd Earl of Castlehaven is Mervyn Tuchet, 2nd Earl of Castlehaven. (Attribution)", 
 "Step 2: According to Passage 2, the father of Mervyn Tuchet, 2nd Earl of Castlehaven (from Step 1) is George Tuchet, 1st Earl of Castlehaven. (Attribution)", 
 "Step 3: Therefore, the person found in Step 2, George Tuchet, 1st Earl of Castlehaven, is the paternal grandfather of James Tuchet, 3rd Earl of Castlehaven. (Logical)" 
]

Target Step to Corrupt: 
Step 2

Output: 
Step 2: According to Passage 2, Mervyn Tuchet is the father of George Tuchet, 1st Earl of Castlehaven. (Attribution)

---

Question: "Who lived longer, Ignace Matondo Kwa Nzambi or Claudia Zobel?"

Retrieved Passages: 
"Passage 1: Ignace Matondo Kwa Nzambi (12 April 1932 – 9 September 2011) was a Congolese politician..." 
"Passage 2: Claudia Zobel (February 27, 1964 – February 10, 1984) was a Filipina actress..."

Ideal Reasoning Steps: 
[ 
 "Step 1: According to Passage 1, Ignace Matondo Kwa Nzambi was born on 12 April 1932 and died on 9 September 2011. (Attribution)", 
 "Step 2: Based on the dates from Step 1, Ignace Matondo Kwa Nzambi lived for 79 years. (Logical)", 
 "Step 3: According to Passage 2, Claudia Zobel was born on February 27, 1964 and died on February 10, 1984. (Attribution)", 
 "Step 4: Based on the dates from Step 3, Claudia Zobel lived for 19 years. (Logical)", 
 "Step 5: Comparing the lifespans from Step 2 (79 years) and Step 4 (19 years), Ignace Matondo Kwa Nzambi lived longer. (Logical)" 
]

Target Step to Corrupt: 
Step 3

Output:
Step 3: According to Passage 2, Claudia Zobel's birth date is February 10, 1984. (Attribution)
""".strip()

def generate_response(tokenizer, llm, messages):
    """Chat template 기반 gpt-oss-120b 응답 생성"""
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
    )

    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
    response = outputs[0].outputs[0].text
    return response.split("assistantfinal")[-1].strip()

def parse_step(step_str: str) -> Tuple[str, str]:
    """
    "Step 1: Do something. (Attribution)" -> ("Step 1:", "(Attribution)")
    """
    step_str = step_str.strip()
    
    # Regex 수정: (Attribution) 또는 (Logical) 레이블을 정확히 찾음
    match = re.match(r"^(Step\s*\d+:)(.*)(\((Attribution|Logical)\))$", step_str, re.DOTALL)
    
    if match:
        prefix = match.group(1).strip()
        label = match.group(3).strip()
        return prefix, label
    else:
        # 레이블 파싱 실패 시 기본값
        print(f"⚠️ Warning: Could not parse label for step: {step_str}. Defaulting to (Logical).")
        prefix_match = re.match(r"^(Step\s*\d+:)", step_str)
        prefix = prefix_match.group(1).strip() if prefix_match else f"Step {step_str.split(':')[0]}:"
        return prefix, "(Logical)"

def save_results(data: List[Dict[str, Any]], filepath: str):
    """결과를 JSON 파일로 저장"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.9,
        max_model_len=3000,
        dtype="bfloat16",
        enable_prefix_caching=True,
    )
    input_filepath = f"/workspace/daeyong/ideal_steps/{args.dataset}_ideal_steps_passage_mapped.json"
    output_filepath = f"/workspace/daeyong/ideal_steps/{args.dataset}_contradictory_rewriting.json"
    
    try:
        with open(input_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} items from {input_filepath}")
    except Exception as e:
        print(f"Error loading data from {input_filepath}: {e}")
        return

    # 1: Load existing results
    results = []
    processed_questions = set() # Store processed Question strings
    
    if os.path.exists(output_filepath):
        try:
            with open(output_filepath, "r", encoding="utf-8") as f:
                results = json.load(f)
            # 2. Add processed questions to Set
            for res in results:
                if 'question' in res:
                    processed_questions.add(res['question'])
            print(f"Loaded {len(results)} existing results from {output_filepath}. Resuming...")
        except Exception as e:
            print(f"Warning: Could not load existing results from {output_filepath}. Starting fresh. Error: {e}")
            results = []
            processed_questions = set()
    
    # =======================================================
    # 🔹 Loop: Error Injection (Once per Question)
    # =======================================================
    for item in tqdm(data, desc="Injecting 'Contradictory' errors"):
        try:
            question = item['question']
            
            # 1. Skip if question already processed
            if question in processed_questions:
                continue

            passages = item['retrieved_passages']
            ideal_steps = item['ideal_steps']
            
            # 2. Find all candidate steps that are "(Attribution)"
            # Store tuples of (index, step_text) where index is 1-based
            candidates = []
            for i, step_text in enumerate(ideal_steps):
                if "(Attribution)" in step_text:
                    candidates.append(i + 1) # 1-based index
            
            # If no Attribution steps exist, skip this question
            if not candidates:
                continue

            # 3. Weighted Sampling
            # Apply squared weights (i^2) to prioritize later steps (which are rarer)
            # Example: If candidates are [1, 2, 4], weights become [1, 4, 16]
            weights = [idx**2 for idx in candidates]
            
            # Select one target index based on weights
            target_index = random.choices(candidates, weights=weights, k=1)[0]

            # 4. Generate Error Step
            try:
                # Format passages for context
                passages_context = "\n".join(f"Passage {i+1}: {p}" for i, p in enumerate(passages))

                # Construct Prompt
                user_prompt = f"""Question: {question}

Retrieved Passages:
{passages_context}

Ideal Reasoning Steps:
{json.dumps(ideal_steps, indent=2, ensure_ascii=False)}

Target Step to Corrupt:
Step {target_index}
""".strip()

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]

                # Generate Corrupted Step
                corrupted_step = generate_response(tokenizer, llm, messages).strip().strip(",")

                # Validate Output Format
                # It must start with "Step X:" and end with "(Attribution)"
                # (Note: Contradictory definition says it must maintain Attribution label)
                if not (corrupted_step.startswith(f"Step {target_index}:") and 
                        "(Attribution)" in corrupted_step):
                    print(f"\n⚠️ Warning: Model output format mismatch for Q: {question[:50]}... (Step {target_index})")
                    print(f"  Expected prefix: 'Step {target_index}:'")
                    print(f"  Expected suffix: '(Attribution)'")
                    print(f"  Got: {corrupted_step}")
                    continue 

                # Create Corrupted Step List (Truncate after error)
                base_steps = ideal_steps[:target_index-1] 
                corrupted_steps = base_steps + [corrupted_step]

                # Save Result
                new_item = item.copy()
                new_item['corrupted_steps'] = corrupted_steps
                new_item['corrupted_step_index'] = target_index # 1-based
                new_item['error_type'] = 'Contradictory'
                results.append(new_item)

                # Mark question as processed
                processed_questions.add(question)

                # Intermediate Save (every 5 items)
                if len(results) % 5 == 0:
                    save_results(results, output_filepath)

            except Exception as e:
                print(f"\nFailed to process question {question[:50]}... on Step {target_index}: {e}")
                import traceback
                traceback.print_exc()
                continue

        except Exception as e:
            # Handle outer loop errors (e.g., item loading)
            print(f"\nFailed to process item {question[:50]}...: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final Save
    save_results(results, output_filepath)
    print(f"✅ Completed error injection. Total {len(results)} corrupted items saved to {output_filepath}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject 'Contradictory' errors into reasoning steps.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., '2wiki') to determine input/output filenames.")
    parser.add_argument("--model_name", type=str, 
                        default="/workspace/hf_transformers/gpt-oss-120b",
                        help="Path to the HuggingFace model directory.")
    
    args = parser.parse_args()
    main(args)
