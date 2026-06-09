import argparse
import copy
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from inject_errors_false_triples_oss120b import (
    ERROR_CONFIGS,
    build_feedback_system_prompts,
    feedback_parse_passages_to_text,
    robust_json_parse,
)


DEFAULT_INPUT_JSON = (
    "/workspace/daeyong/fourth_finetuning_data/"
    "final_answer_with_wrong_conclusion_off_topic_first_premature_attribution.json"
)
DEFAULT_OUTPUT_JSON = (
    "/workspace/daeyong/fourth_finetuning_data/"
    "final_answer_with_wrong_conclusion_off_topic_first_premature_attribution_gpt_oss20b_errors.json"
)
DEFAULT_EXISTING_ERRORS_JSON = DEFAULT_OUTPUT_JSON
DEFAULT_MODEL_PATH = (
    "/workspace/hf_transformers/models--openai--gpt-oss-20b/"
    "snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
)


PREMATURE_CONCLUSION_FEEDBACK_SYSTEM_PROMPT = """You are an expert reasoning evaluator.
Your task is to analyze a reasoning step that contains a "Premature Conclusion" error and provide critical feedback.

A "Premature Conclusion" error occurs when the current step submits a final answer before the required explicit logical inference or comparison step has been made from the previous attributed facts.

You will receive:
1. Question: The user's original question.
2. Retrieved Passages: Contextual information and the only source of truth.
3. Previous Steps: The reasoning steps before the error.
4. Current Step: The premature final-answer step.

Your feedback MUST:
1. Diagnose why the final answer was submitted too early.
2. Provide exactly one immediate next step that should be generated before submitting the final answer.
3. Be concise and grounded only in the passages and previous steps.

Output ONLY a valid JSON object:
{
  "error_type": "Premature Conclusion",
  "diagnosis": "...",
  "guidance": "..."
}

Example:
Question: Who is older, Alice or Bob?
Retrieved Passages:
Passage 1: Alice was born in 1980.
Passage 2: Bob was born in 1990.
Previous Steps:
Step 1: According to Passage 1, Alice was born in 1980. (Attribution)
Step 2: According to Passage 2, Bob was born in 1990. (Attribution)
Current Step:
Step 3: ####ANSWER: Alice (Final Answer)
Output:
{
  "error_type": "Premature Conclusion",
  "diagnosis": "The step submits the answer immediately after extracting birth years, but it has not yet made the explicit comparison that 1980 is earlier than 1990.",
  "guidance": "Generate a logical step stating that Alice is older than Bob because 1980 is earlier than 1990."
}
""".strip()


STRICT_RETRY_SYSTEM_PROMPT = """You are an expert reasoning evaluator.
The current reasoning step has the assigned error category "{error_type}".

Regenerate only the feedback for this assigned category.
Do not debate whether the assigned category is correct. Write feedback for the assigned category.
Output ONLY one valid JSON object with exactly these keys:
{{
  "diagnosis": "...",
  "guidance": "..."
}}

Do not output markdown, commentary, analysis, or code fences.
Keep both fields concise, specific, and grounded in the provided passages and previous steps.
The diagnosis must be at least 10 words, mention the concrete fact or inference being evaluated, and must not include guidance instructions or the word "Guidance".
The guidance must be a direct instruction with at least 10 words.
Do not use generic diagnosis text such as "No error detected" or "The answer is correct."
Do not use "we need to", "we should", "I need to", "let's", "maybe", "perhaps", "seems", "appears", "not sure", "known error", "error label", or "error_type".
Do not include section headers inside field values.
""".strip()


CORRECT_FEEDBACK_SYSTEM_PROMPT = """You are an expert reasoning evaluator and verifier.
The current reasoning step is already assigned the category "Correct (No Error)".

Use the Retrieved Passages as the only source of truth. Do not use external knowledge.
Do not reclassify the step or debate the assigned category.

Write:
1. diagnosis: a concise, self-contained explanation of why the Current Step is logically sound, relevant to the question, and supported by the retrieved passages or previous established steps.
2. guidance: exactly one immediate next action that should follow this correct step. If the current step already completes the reasoning or gives the final answer, instruct the model to stop reasoning rather than adding another step.

Strict style rules:
- diagnosis must explain why the step is correct; do not include next-step instructions in diagnosis.
- diagnosis must be at least 10 words and mention the concrete fact, entity, comparison, or inference that makes the step correct.
- guidance must be a direct instruction with at least 10 words and no more than two sentences.
- Do not use generic diagnosis text such as "No error detected" or "The answer is correct."
- Do not write first-person or deliberation phrases such as "we need to", "we should", "I need to", "let's", "maybe", "perhaps", "seems", "appears", or "not sure".
- Do not mention the data-generation category, error label, "known error", or "error_type" in either field.
- Do not include section headers inside field values.

Output ONLY a valid JSON object with keys "diagnosis" and "guidance".
Do not output markdown, code fences, analysis, or extra prose.
""".strip()


ERROR_DEFINITIONS = {
    "Correct (No Error)": (
        "The current step is logically sound, factually supported by the Retrieved Passages "
        "or validly follows from previous established steps, and moves the reasoning forward correctly."
    ),
    "Wrong Conclusion": (
        "The current step is a final answer that contradicts the conclusion supported by the preceding reasoning "
        "or gives the wrong answer value for the question."
    ),
    "Overthinking": (
        "The current step continues with unnecessary reasoning after the useful path is already clear, or adds "
        "extra information that does not help recover the correct answer."
    ),
    "Off-topic": (
        "The current step introduces information or an inference irrelevant to the question or to the necessary "
        "sub-goal at this point in the reasoning chain."
    ),
    "Redundancy": (
        "The current step repeats information or a conclusion already present in previous steps without adding "
        "new meaningful progress."
    ),
    "Inefficiency": (
        "The current step is procedural meta-talk, planning, or placeholder reasoning instead of performing an "
        "actual evidence extraction or logical inference."
    ),
    "Logical Fallacy": (
        "The current step uses facts or prior statements to draw an invalid inference, comparison, calculation, "
        "or causal/attribution conclusion."
    ),
    "Unsupported": (
        "The current step makes a factual claim that cannot be found in the Retrieved Passages and is therefore "
        "hallucinated or insufficiently grounded."
    ),
    "Contradictory": (
        "The current step makes a factual claim that directly conflicts with information explicitly stated in "
        "the Retrieved Passages."
    ),
    "Information Miss": (
        "The current step incorrectly claims that needed information is unavailable or unknown even though it is "
        "present in the Retrieved Passages."
    ),
    "Premature Attribution": (
        "The current step extracts an attribute of an entity before first establishing the necessary linking "
        "relationship between that entity and the subject of the question."
    ),
    "Premature Conclusion": (
        "The current step submits a final answer before the required explicit logical inference or comparison "
        "has been made from the previous attributed facts."
    ),
}


KNOWN_ERROR_FEEDBACK_SYSTEM_PROMPT_TEMPLATE = """You are an expert reasoning evaluator and verifier.
Your task is to regenerate feedback for a reasoning step with a known error type.

Assigned error category: "{error_type}"
Definition: {definition}

Use the Retrieved Passages as the only source of truth. Do not use external knowledge.
Do not assume the Previous Steps are factually correct; treat them as reasoning history and verify any relevant claim against the passages.
Do not debate whether the assigned category is correct. The category is fixed for this data-generation task.
If the category is ambiguous, still write the best concise diagnosis/guidance consistent with the assigned category.

Write:
1. diagnosis: a concise, self-contained explanation of why the Current Step has the known error. Mention the concrete mismatch, repetition, unsupported claim, invalid inference, or missing bridge.
2. guidance: exactly one immediate next action that moves the reasoning toward the passage-supported answer. If a previous step contains an unresolved mistake, the guidance should correct or bypass that mistaken assumption rather than endorsing it.

Strict style rules:
- diagnosis must explain the problem only; do not include the word "Guidance" or next-step instructions in diagnosis.
- guidance must be a direct instruction, at least 10 words, and no more than two sentences.
- Do not write first-person or deliberation phrases such as "we need to", "we should", "I need to", "let's", "maybe", "perhaps", "seems", "appears", or "not sure".
- Do not mention the data-generation category, error label, "known error", or "error_type" in either field.
- Do not include section headers inside field values.

For final-answer guidance, include the required strict format `####ANSWER: <answer_value>` only when the answer value is directly supported by the passages and satisfies the question.

Output ONLY a valid JSON object with keys "diagnosis" and "guidance".
Do not output markdown, code fences, analysis, or extra prose.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate diagnosis/guidance for non-correct training records "
            "with gpt-oss-20b, using existing per-error feedback prompts."
        )
    )
    parser.add_argument("--input_json", default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output_json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output_jsonl", default=None)
    parser.add_argument("--metadata_json", default=None)
    parser.add_argument("--failures_json", default=None)
    parser.add_argument(
        "--existing_errors_json",
        default=None,
        help=(
            "Optional gpt-oss output containing non-Correct rows. When producing a full dataset, "
            "these rows are reused instead of regenerated."
        ),
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=12000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--reasoning_effort",
        choices=["low", "medium", "high"],
        default="low",
        help="gpt-oss chat-template reasoning effort. Low reduces analysis-channel spillover.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_retries", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--repair_quality_issues",
        action="store_true",
        help=(
            "When resuming from an existing output file, regenerate rows whose "
            "diagnosis/guidance contain quality artifacts instead of skipping them."
        ),
    )
    parser.add_argument(
        "--min_guidance_words",
        type=int,
        default=10,
        help="Minimum word count required for guidance in --repair_quality_issues mode.",
    )
    parser.add_argument(
        "--prompt_style",
        choices=["definition", "legacy_feedback"],
        default="definition",
        help=(
            "definition=use concise per-error definitions from the evaluator taxonomy; "
            "legacy_feedback=use the older few-shot feedback prompt builder."
        ),
    )
    parser.add_argument(
        "--include_correct",
        action="store_true",
        help="Include Correct (No Error) rows in the output.",
    )
    parser.add_argument(
        "--generate_correct",
        action="store_true",
        help="Generate diagnosis/guidance for Correct (No Error) rows instead of copying them.",
    )
    parser.add_argument(
        "--save_every_batches",
        type=int,
        default=5,
        help="Checkpoint output files every N batches. 0 disables intermediate writes.",
    )
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_output_jsonl(output_json: str) -> str:
    path = Path(output_json)
    return str(path.with_suffix(".jsonl"))


def default_metadata_path(output_json: str) -> str:
    path = Path(output_json)
    return str(path.with_name(path.stem + "_metadata.json"))


def default_failures_path(output_json: str) -> str:
    path = Path(output_json)
    return str(path.with_name(path.stem + "_failures.json"))


def load_json_list(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def write_json(path: str, payload) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def has_feedback(item: dict) -> bool:
    return (
        isinstance(item.get("diagnosis"), str)
        and item["diagnosis"].strip()
        and isinstance(item.get("guidance"), str)
        and item["guidance"].strip()
    )


QUALITY_PATTERNS = {
    "analysis_leak": re.compile(
        r"\b(we need to|we should|i need to|let'?s|let us)\b|assistantfinal",
        re.IGNORECASE,
    ),
    "section_header_inside": re.compile(
        r"\b(?:Guidance|Diagnosis)\s*:",
        re.IGNORECASE,
    ),
    "guidance_mixed_into_diagnosis": re.compile(
        r"\b(?:the guidance|next action|so guidance|guidance should)\b",
        re.IGNORECASE,
    ),
    "label_debate": re.compile(
        r"\b(?:known error|error_type|error type|error label|data label|"
        r"labeled as|but the error type|seems correct|appears correct|"
        r"actually correct|the step is correct|chain is correct|"
        r"no [a-z -]+fallacy is present|no error is present|no error detected)\b",
        re.IGNORECASE,
    ),
    "uncertainty": re.compile(
        r"\b(?:maybe|perhaps|not sure)\b",
        re.IGNORECASE,
    ),
}


def count_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", str(text)))


def feedback_quality_issues(item: dict, min_guidance_words: int = 10) -> List[str]:
    diagnosis = str(item.get("diagnosis", "") or "")
    guidance = str(item.get("guidance", "") or "")
    combined = f"{diagnosis}\n{guidance}"
    issues: List[str] = []
    truncated_end = re.compile(
        r"[,;:]\s*$|"
        r"\b(?:and|or|but|because|with|while|to|from|into|by|for|that|which|"
        r"therefore|also|if|when|where|as|than|then)\s*$",
        re.IGNORECASE,
    )

    if not diagnosis.strip():
        issues.append("missing_diagnosis")
    if not guidance.strip():
        issues.append("missing_guidance")
    if count_words(guidance) < min_guidance_words:
        issues.append("guidance_too_short")
    if count_words(diagnosis) < 10:
        issues.append("diagnosis_too_short")
    if truncated_end.search(diagnosis.rstrip()):
        issues.append("diagnosis_likely_truncated")
    if truncated_end.search(guidance.rstrip()):
        issues.append("guidance_likely_truncated")

    for issue_name, pattern in QUALITY_PATTERNS.items():
        target = diagnosis if issue_name == "guidance_mixed_into_diagnosis" else combined
        if pattern.search(target):
            issues.append(issue_name)

    return issues


def record_key(item: dict) -> Tuple[str, str, str]:
    return (
        str(item.get("question", "")).strip(),
        str(item.get("current_step", "")).strip(),
        str(item.get("error_type", "")).strip(),
    )


def build_existing_map(output_json: str) -> Dict[Tuple[str, str, str], dict]:
    if not output_json:
        return {}
    path = Path(output_json)
    if not path.exists():
        return {}
    try:
        existing = load_json_list(str(path))
    except Exception:
        return {}
    return {
        record_key(item): item
        for item in existing
        if isinstance(item, dict) and has_feedback(item)
    }


def first_existing(
    item: dict,
    existing_maps: List[Dict[Tuple[str, str, str], dict]],
) -> Optional[dict]:
    key = record_key(item)
    for existing_map in existing_maps:
        existing = existing_map.get(key)
        if existing is not None:
            return existing
    return None


def make_error_type_maps() -> Tuple[Dict[str, str], Dict[str, str]]:
    display_to_key = {
        config.record_error_type: key for key, config in ERROR_CONFIGS.items()
    }
    display_to_key["Correct (No Error)"] = "correct"
    display_to_key["Premature Conclusion"] = "premature_conclusion"
    normalized_to_display = {
        re.sub(r"[^a-z0-9]+", " ", display.lower()).strip(): display
        for display in display_to_key
    }
    return display_to_key, normalized_to_display


def normalize_display_error_type(error_type: str, normalized_to_display: Dict[str, str]) -> str:
    raw = str(error_type).strip()
    if raw in normalized_to_display.values():
        return raw
    key = re.sub(r"[^a-z0-9]+", " ", raw.lower()).strip()
    if key in normalized_to_display:
        return normalized_to_display[key]
    raise ValueError(f"Unsupported error_type: {error_type}")


def build_system_prompts(prompt_style: str) -> Dict[str, str]:
    if prompt_style == "definition":
        prompts = {"correct": CORRECT_FEEDBACK_SYSTEM_PROMPT}
        for key, config in ERROR_CONFIGS.items():
            display = config.record_error_type
            prompts[key] = KNOWN_ERROR_FEEDBACK_SYSTEM_PROMPT_TEMPLATE.format(
                error_type=display,
                definition=ERROR_DEFINITIONS[display],
            )
        prompts["premature_conclusion"] = KNOWN_ERROR_FEEDBACK_SYSTEM_PROMPT_TEMPLATE.format(
            error_type="Premature Conclusion",
            definition=ERROR_DEFINITIONS["Premature Conclusion"],
        )
        return prompts

    prompts = build_feedback_system_prompts()
    prompts["correct"] = CORRECT_FEEDBACK_SYSTEM_PROMPT
    prompts["premature_conclusion"] = PREMATURE_CONCLUSION_FEEDBACK_SYSTEM_PROMPT
    augmented: Dict[str, str] = {}
    for key, prompt in prompts.items():
        display = (
            "Premature Conclusion"
            if key == "premature_conclusion"
            else ERROR_CONFIGS[key].record_error_type
        )
        augmented[key] = (
            prompt.strip()
            + "\n\n"
            + f'The Current Step is already labeled "{display}". Keep this error type fixed; regenerate only diagnosis and guidance.'
            + "\nIf Ideal Reasoning Steps are unavailable, infer the immediate corrective step using only the Retrieved Passages and Previous Steps."
            + "\nDo not assume the Previous Steps are factually correct. Treat them as reasoning history, then independently verify the relevant facts against the Retrieved Passages."
            + "\nThe guidance must move the chain toward the passage-supported answer, even when that requires correcting an unresolved mistaken assumption in the Previous Steps."
            + "\nDo not instruct submitting a final answer unless that answer is directly supported by the Retrieved Passages and satisfies the Question."
            + '\nReturn JSON only. The fields "diagnosis" and "guidance" are required.'
        )
    return augmented


def format_steps(steps) -> str:
    if isinstance(steps, list) and steps:
        return "\n".join(str(step).strip() for step in steps if str(step).strip())
    return "(No previous steps.)"


def build_feedback_user_prompt(item: dict, error_type: str) -> str:
    question = str(item.get("question", "")).strip()
    passages_text = feedback_parse_passages_to_text(item.get("retrieved_passages", []))
    previous_steps = item.get("previous_steps", [])
    if not isinstance(previous_steps, list):
        previous_steps = []
    current_step = str(item.get("current_step", "")).strip()
    previous_text = format_steps(previous_steps)
    chain_steps = previous_steps + ([current_step] if current_step else [])
    chain_text = format_steps(chain_steps)

    return (
        f"Question:\n{question}\n\n"
        f"Retrieved Passages:\n{passages_text}\n\n"
        "Ideal Reasoning Steps:\n(Unavailable for this dataset.)\n\n"
        f"Previous Steps:\n{previous_text}\n\n"
        f"Current Step:\n{current_step}\n\n"
        f"Full Reasoning Chain:\n{chain_text}\n\n"
        f"Assigned Error Category:\n{error_type}\n\n"
        'Generate a new JSON object with keys "diagnosis" and "guidance". '
        "The field values must not contain section names, private reasoning, "
        "or debate about the assigned category. The guidance must be a corrective "
        "instruction sentence, not only a final answer."
    )


def build_training_input(item: dict) -> str:
    question = str(item.get("question", "")).strip()
    passages_text = feedback_parse_passages_to_text(item.get("retrieved_passages", []))
    previous_text = format_steps(item.get("previous_steps", []))
    current_step = str(item.get("current_step", "")).strip()
    return (
        "### Task: Evaluate the Correctness of the Reasoning Step\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved Passages:\n{passages_text}\n\n"
        f"Previous Steps:\n{previous_text}\n\n"
        f"Step to evaluate:\n{current_step}"
    )


def write_jsonl(path: str, items: List[dict]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            if not has_feedback(item):
                continue
            output = {
                "error_type": item.get("error_type", ""),
                "diagnosis": item.get("diagnosis", ""),
                "guidance": item.get("guidance", ""),
            }
            row = {
                "input": build_training_input(item),
                "output": json.dumps(output, ensure_ascii=False),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def apply_chat_template(
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    reasoning_effort: str = "low",
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        reasoning_effort=reasoning_effort,
    )


def clean_field(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = text.strip("`").strip()
    return re.sub(r"\s+", " ", text)


def regex_section(text: str, name: str) -> Optional[str]:
    pattern = rf"(?is)\b{name}\b\s*[:\-]\s*(.+?)(?=\n\s*(?:diagnosis|guidance|error_type)\b\s*[:\-]|\Z)"
    match = re.search(pattern, text)
    if not match:
        return None
    value = match.group(1).strip().strip('"').strip()
    return clean_field(value) if value else None


def extract_feedback_fields(raw_output: str) -> Tuple[Optional[str], Optional[str], Optional[dict]]:
    parsed = robust_json_parse(raw_output)
    if isinstance(parsed, dict):
        diagnosis = parsed.get("diagnosis")
        guidance = parsed.get("guidance")
        if isinstance(diagnosis, str) and isinstance(guidance, str):
            return clean_field(diagnosis), clean_field(guidance), parsed
        for nested_key in ("feedback", "gold_feedback"):
            nested = parsed.get(nested_key)
            if isinstance(nested, dict):
                diagnosis = nested.get("diagnosis")
                guidance = nested.get("guidance")
                if isinstance(diagnosis, str) and isinstance(guidance, str):
                    return clean_field(diagnosis), clean_field(guidance), parsed

    text = raw_output.split("assistantfinal")[-1].strip()
    diagnosis = regex_section(text, "diagnosis")
    guidance = regex_section(text, "guidance")
    if diagnosis and guidance:
        return diagnosis, guidance, parsed
    return None, None, parsed


def load_llm_and_tokenizer(args: argparse.Namespace):
    print(f"Loading gpt-oss-20b with vLLM: {args.model_path}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        enforce_eager=False,
        enable_prefix_caching=True,
        seed=args.seed,
    )
    return llm, tokenizer


def save_all(
    output_json: str,
    output_jsonl: str,
    metadata_json: str,
    failures_json: str,
    output_items: List[dict],
    metadata: dict,
    failures: List[dict],
) -> None:
    write_json(output_json, output_items)
    write_jsonl(output_jsonl, output_items)
    write_json(metadata_json, metadata)
    write_json(failures_json, failures)


def main() -> None:
    args = parse_args()
    args.output_jsonl = args.output_jsonl or default_output_jsonl(args.output_json)
    args.metadata_json = args.metadata_json or default_metadata_path(args.output_json)
    args.failures_json = args.failures_json or default_failures_path(args.output_json)

    display_to_key, normalized_to_display = make_error_type_maps()
    system_prompts = build_system_prompts(args.prompt_style)
    input_items = load_json_list(args.input_json)
    existing_maps: List[Dict[Tuple[str, str, str], dict]] = []
    if not args.overwrite:
        existing_maps.append(build_existing_map(args.output_json))
        if args.existing_errors_json and args.existing_errors_json != args.output_json:
            existing_maps.append(build_existing_map(args.existing_errors_json))

    output_items: List[dict] = []
    pending: List[dict] = []
    input_counts = Counter()
    selected_counts = Counter()
    repair_issue_counts = Counter()
    skipped_existing = 0

    for source_index, item in enumerate(input_items):
        if not isinstance(item, dict):
            continue
        raw_error_type = str(item.get("error_type", "")).strip()
        input_counts[raw_error_type] += 1
        if raw_error_type == "Correct (No Error)" and not (args.include_correct or args.generate_correct):
            continue

        out_item = copy.deepcopy(item)
        if raw_error_type == "Correct (No Error)" and not args.generate_correct:
            output_items.append(out_item)
            continue

        display_error_type = normalize_display_error_type(raw_error_type, normalized_to_display)
        key = display_to_key[display_error_type]
        existing = first_existing(item, existing_maps)
        existing_issues: List[str] = []
        if existing is not None and args.repair_quality_issues:
            existing_issues = feedback_quality_issues(existing, args.min_guidance_words)

        if existing is not None and not existing_issues:
            output_items.append(existing)
            skipped_existing += 1
            continue

        if existing is not None:
            out_item = copy.deepcopy(existing)
            repair_issue_counts.update(existing_issues)
        else:
            out_item.pop("diagnosis", None)
            out_item.pop("guidance", None)

        output_index = len(output_items)
        output_items.append(out_item)
        pending.append(
            {
                "source_index": source_index,
                "output_index": output_index,
                "prompt_key": key,
                "error_type": display_error_type,
                "item": item,
                "repair_quality_issues": existing_issues,
            }
        )
        selected_counts[display_error_type] += 1
        if args.limit > 0 and len(pending) >= args.limit:
            break

    metadata = {
        "created_at_utc": utc_now_iso(),
        "updated_at_utc": utc_now_iso(),
        "input_json": args.input_json,
        "existing_errors_json": args.existing_errors_json,
        "output_json": args.output_json,
        "output_jsonl": args.output_jsonl,
        "model_path": args.model_path,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "args": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "batch_size": args.batch_size,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "max_retries": args.max_retries,
            "include_correct": args.include_correct,
            "generate_correct": args.generate_correct,
            "overwrite": args.overwrite,
            "limit": args.limit,
            "prompt_style": args.prompt_style,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "reasoning_effort": args.reasoning_effort,
            "repair_quality_issues": args.repair_quality_issues,
            "min_guidance_words": args.min_guidance_words,
        },
        "prompt_sources": {
            "logical_fallacy": "inject_errors_false_triples_oss120b.py + feedback_fewshot_examples.py",
            "information_miss": "inject_errors_false_triples_oss120b.py + feedback_fewshot_examples.py",
            "inefficiency": "inject_errors_false_triples_oss120b.py + feedback_fewshot_examples.py",
            "redundancy": "inject_errors_false_triples_oss120b.py + feedback_fewshot_examples.py",
            "off_topic": "inject_errors_false_triples_oss120b.py + feedback_fewshot_examples.py",
            "overthinking": "inject_errors_false_triples_oss120b.py + feedback_fewshot_examples.py",
            "contradictory": "inject_errors_false_triples_oss120b.py + feedback_fewshot_examples.py",
            "unsupported": "inject_errors_false_triples_oss120b.py + feedback_fewshot_examples.py",
            "premature_attribution": "inject_errors_false_triples_oss120b.py::PREMATURE_ATTRIBUTION_FEEDBACK_SYSTEM_PROMPT",
            "wrong_conclusion": "oss_final_answer_swap_feedback.py::FEEDBACK_SYSTEM_PROMPT",
            "premature_conclusion": "prompts.py Premature Conclusion example adapted into this script",
            "correct": "regenerate_error_feedback_gpt_oss20b.py::CORRECT_FEEDBACK_SYSTEM_PROMPT",
        },
        "input_error_type_counts": dict(input_counts),
        "selected_error_type_counts": dict(selected_counts),
        "repair_issue_counts": dict(repair_issue_counts),
        "skipped_existing": skipped_existing,
        "generated": 0,
        "failed": 0,
    }

    print(f"Input rows: {len(input_items)}")
    print(f"Output rows selected: {len(output_items)}")
    print(f"Pending regeneration: {len(pending)}")
    print(f"Skipped existing valid feedback: {skipped_existing}")
    print(f"Selected counts: {dict(selected_counts)}")
    if repair_issue_counts:
        print(f"Repair issue counts: {dict(repair_issue_counts)}")

    failures: List[dict] = []
    if not pending:
        metadata["updated_at_utc"] = utc_now_iso()
        metadata["generated"] = sum(1 for item in output_items if has_feedback(item))
        metadata["failed"] = 0
        metadata["output_error_type_counts"] = dict(
            Counter(item.get("error_type") for item in output_items)
        )
        metadata["valid_feedback_rows"] = sum(1 for item in output_items if has_feedback(item))
        save_all(
            args.output_json,
            args.output_jsonl,
            args.metadata_json,
            args.failures_json,
            output_items,
            metadata,
            failures,
        )
        return

    llm, tokenizer = load_llm_and_tokenizer(args)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    remaining = pending
    total_success = 0
    for attempt in range(args.max_retries + 1):
        next_remaining: List[dict] = []
        print(f"Generation attempt {attempt + 1}/{args.max_retries + 1}: {len(remaining)} rows")
        for batch_no, start in enumerate(range(0, len(remaining), args.batch_size), start=1):
            batch = remaining[start : start + args.batch_size]
            prompts = []
            for rec in batch:
                if attempt == 0:
                    system_prompt = system_prompts[rec["prompt_key"]]
                else:
                    system_prompt = STRICT_RETRY_SYSTEM_PROMPT.format(
                        error_type=rec["error_type"]
                    )
                user_prompt = build_feedback_user_prompt(rec["item"], rec["error_type"])
                if args.repair_quality_issues and rec.get("repair_quality_issues"):
                    user_prompt += (
                        "\n\nPrevious feedback was rejected for these quality issues: "
                        + ", ".join(rec["repair_quality_issues"])
                        + ". Avoid these artifacts in both fields."
                    )
                prompts.append(
                    apply_chat_template(
                        tokenizer,
                        system_prompt,
                        user_prompt,
                        reasoning_effort=args.reasoning_effort,
                    )
                )

            generations = llm.generate(prompts, sampling_params, use_tqdm=False)
            for rec, generation in zip(batch, generations):
                raw_output = generation.outputs[0].text if generation.outputs else ""
                diagnosis, guidance, parsed = extract_feedback_fields(raw_output)
                parsed_feedback = {
                    "diagnosis": diagnosis or "",
                    "guidance": guidance or "",
                }
                generated_issues = (
                    feedback_quality_issues(parsed_feedback, args.min_guidance_words)
                    if args.repair_quality_issues
                    else []
                )

                if not diagnosis or not guidance or generated_issues:
                    if attempt >= args.max_retries:
                        failures.append(
                            {
                                "source_index": rec["source_index"],
                                "output_index": rec["output_index"],
                                "error_type": rec["error_type"],
                                "question": str(rec["item"].get("question", ""))[:300],
                                "reason": (
                                    "generated_quality_issues"
                                    if generated_issues
                                    else "parse_or_missing_feedback_failed"
                                ),
                                "quality_issues": generated_issues,
                                "previous_quality_issues": rec.get("repair_quality_issues"),
                                "parsed": parsed,
                                "raw_output": raw_output,
                            }
                        )
                    else:
                        retry_rec = dict(rec)
                        retry_rec["repair_quality_issues"] = generated_issues or [
                            "parse_or_missing_feedback_failed"
                        ]
                        next_remaining.append(retry_rec)
                    continue

                out_item = output_items[rec["output_index"]]
                out_item["error_type"] = rec["error_type"]
                out_item["diagnosis"] = diagnosis
                out_item["guidance"] = guidance
                total_success += 1

            metadata["updated_at_utc"] = utc_now_iso()
            metadata["generated"] = total_success
            metadata["failed"] = len(failures)
            metadata["remaining_for_retry"] = len(next_remaining)
            if args.save_every_batches and batch_no % args.save_every_batches == 0:
                save_all(
                    args.output_json,
                    args.output_jsonl,
                    args.metadata_json,
                    args.failures_json,
                    output_items,
                    metadata,
                    failures,
                )
                print(
                    f"  checkpoint attempt={attempt + 1} batch={batch_no} "
                    f"generated={total_success} failures={len(failures)}"
                )

        remaining = next_remaining
        if not remaining:
            break

    metadata["updated_at_utc"] = utc_now_iso()
    metadata["generated"] = sum(1 for item in output_items if has_feedback(item))
    metadata["failed"] = len(failures)
    metadata["output_error_type_counts"] = dict(Counter(item.get("error_type") for item in output_items))
    metadata["valid_feedback_rows"] = sum(1 for item in output_items if has_feedback(item))

    save_all(
        args.output_json,
        args.output_jsonl,
        args.metadata_json,
        args.failures_json,
        output_items,
        metadata,
        failures,
    )

    print(f"Done. Generated valid feedback rows: {metadata['valid_feedback_rows']}")
    print(f"Failures: {len(failures)}")
    print(f"JSON: {args.output_json}")
    print(f"JSONL: {args.output_jsonl}")
    print(f"Metadata: {args.metadata_json}")
    print(f"Failures: {args.failures_json}")


if __name__ == "__main__":
    main()
