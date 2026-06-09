import argparse
import ast
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_INPUT_ROOT = "/workspace/daeyong/filtering_noise_data/false_triples_oss120b_seed42"
DEFAULT_MODEL_PATH = "/workspace/hf_transformers/gpt-oss-120b"
DATASETS = ("2wiki", "hotpotqa", "musique")


@dataclass(frozen=True)
class ErrorTypeConfig:
    key: str
    record_error_type: str
    prompt_file: str
    prompt_var: str
    select_policy: str
    expected_label: str


ERROR_CONFIGS: Dict[str, ErrorTypeConfig] = {
    "logical_fallacy": ErrorTypeConfig(
        key="logical_fallacy",
        record_error_type="Logical Fallacy",
        prompt_file="/workspace/daeyong/inject_error_logical_fallacy.py",
        prompt_var="system_prompt",
        select_policy="logical",
        expected_label="Logical",
    ),
    "information_miss": ErrorTypeConfig(
        key="information_miss",
        record_error_type="Information Miss",
        prompt_file="/workspace/daeyong/inject_error_information_miss.py",
        prompt_var="system_prompt",
        select_policy="attribution",
        expected_label="Attribution",
    ),
    "inefficiency": ErrorTypeConfig(
        key="inefficiency",
        record_error_type="Inefficiency",
        prompt_file="/workspace/daeyong/inject_error_inefficiency.py",
        prompt_var="system_prompt",
        select_policy="all",
        expected_label="Either",
    ),
    "redundancy": ErrorTypeConfig(
        key="redundancy",
        record_error_type="Redundancy",
        prompt_file="/workspace/daeyong/inject_error_redundancy.py",
        prompt_var="system_prompt",
        select_policy="from2",
        expected_label="Either",
    ),
    "off_topic": ErrorTypeConfig(
        key="off_topic",
        record_error_type="Off-topic",
        prompt_file="/workspace/daeyong/inject_error_off_topic.py",
        prompt_var="system_prompt",
        select_policy="all",
        expected_label="Either",
    ),
    "overthinking": ErrorTypeConfig(
        key="overthinking",
        record_error_type="Overthinking",
        prompt_file="/workspace/daeyong/inject_error_overthinking.py",
        prompt_var="system_prompt",
        select_policy="overthinking",
        expected_label="Either",
    ),
    "contradictory": ErrorTypeConfig(
        key="contradictory",
        record_error_type="Contradictory",
        prompt_file="/workspace/daeyong/inject_error_attribution_contradictory.py",
        prompt_var="system_prompt",
        select_policy="attribution",
        expected_label="Attribution",
    ),
    "unsupported": ErrorTypeConfig(
        key="unsupported",
        record_error_type="Unsupported",
        prompt_file="/workspace/daeyong/inject_error_attribution_unsupported.py",
        prompt_var="system_prompt",
        select_policy="attribution",
        expected_label="Attribution",
    ),
    "premature_attribution": ErrorTypeConfig(
        key="premature_attribution",
        record_error_type="Premature Attribution",
        prompt_file="/workspace/daeyong/oss_premature_attribution.py",
        prompt_var="PREMATURE_ATTRIBUTION_SYSTEM_PROMPT",
        select_policy="premature_step1",
        expected_label="Either",
    ),
    "wrong_conclusion": ErrorTypeConfig(
        key="wrong_conclusion",
        record_error_type="Wrong Conclusion",
        prompt_file="/workspace/daeyong/generate_wrong_conclusion.py",
        prompt_var="system_prompt",
        select_policy="final_answer_last",
        expected_label="Final Answer",
    ),
}


ERROR_ALIASES = {
    "logical fallacy": "logical_fallacy",
    "logical-fallacy": "logical_fallacy",
    "logical_fallacy": "logical_fallacy",
    "information miss": "information_miss",
    "information-miss": "information_miss",
    "information_miss": "information_miss",
    "inefficiency": "inefficiency",
    "redundancy": "redundancy",
    "off-topic": "off_topic",
    "off topic": "off_topic",
    "off_topic": "off_topic",
    "overthinking": "overthinking",
    "contradictory": "contradictory",
    "contradictiory": "contradictory",
    "unsupported": "unsupported",
    "premature attribution": "premature_attribution",
    "premature-attribution": "premature_attribution",
    "premature_attribution": "premature_attribution",
    "wrong conclusion": "wrong_conclusion",
    "wrong-conclusion": "wrong_conclusion",
    "wrong_conclusion": "wrong_conclusion",
}


DEFAULT_ERROR_TYPES = [
    "logical_fallacy",
    "information_miss",
    "inefficiency",
    "redundancy",
    "off_topic",
    "overthinking",
    "contradictory",
    "unsupported",
    "premature_attribution",
    "wrong_conclusion",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified gpt-oss-120b error-injection pipeline for false-triple ideal steps."
    )
    parser.add_argument("--input_root", type=str, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument(
        "--error_types",
        nargs="+",
        default=DEFAULT_ERROR_TYPES,
        help="Error types. Aliases/spaces/hyphens are accepted.",
    )
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max_items_per_dataset",
        type=int,
        default=0,
        help="If > 0, process at most this many items per dataset.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=10000)
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Only build normalized inputs and summary; skip model generation.",
    )

    # Feedback mode options (replacement for generate_feedback_*.py scripts)
    parser.add_argument(
        "--task",
        choices=["inject", "feedback"],
        default="inject",
        help="inject: generate error steps, feedback: generate diagnosis/guidance for existing error data.",
    )
    parser.add_argument("--input_dir_name", type=str, default="error_data_reindexed")
    parser.add_argument("--output_dir_name", type=str, default="feedback_data_reindexed")
    parser.add_argument(
        "--max_items_per_file",
        type=int,
        default=0,
        help="(feedback) Process at most this many non-resumed items per file. 0 means all.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--save_raw_feedback_output",
        action="store_true",
        help="(feedback) Save raw model output in each successful item.",
    )
    parser.add_argument(
        "--feedback_max_tokens",
        type=int,
        default=700,
        help="(feedback) max tokens for diagnosis/guidance generation.",
    )
    parser.add_argument(
        "--feedback_input_file",
        type=str,
        default=None,
        help="(feedback) Optional single input JSON file. If set, directory mode is skipped.",
    )
    parser.add_argument(
        "--feedback_output_file",
        type=str,
        default=None,
        help="(feedback) Output JSON path for --feedback_input_file mode.",
    )
    parser.add_argument(
        "--feedback_error_type",
        type=str,
        default=None,
        help="(feedback) Error type for single-file mode (e.g., premature_attribution).",
    )
    parser.add_argument(
        "--feedback_output_field",
        choices=["top_level", "feedback", "gold_feedback"],
        default="top_level",
        help="(feedback single-file mode) Where to write generated diagnosis/guidance.",
    )
    parser.add_argument(
        "--feedback_write_top_level",
        action="store_true",
        help="(feedback single-file mode) Also write top-level diagnosis/guidance/error_type when output field is nested.",
    )
    return parser.parse_args()


def normalize_error_type(raw: str) -> str:
    key = raw.strip().lower().replace("_", " ").replace("-", " ")
    key = re.sub(r"\s+", " ", key)
    normalized = ERROR_ALIASES.get(key)
    if normalized is None:
        raise ValueError(f"Unsupported error type: {raw}")
    return normalized


def unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def ensure_dirs(base: Path) -> Dict[str, Path]:
    out = {
        "root": base / "error_data",
        "logs": base / "error_data" / "logs",
    }
    out["root"].mkdir(parents=True, exist_ok=True)
    out["logs"].mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: Path, payload, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_text_constant(node) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "strip"
        and not node.args
        and not node.keywords
    ):
        inner = parse_text_constant(node.func.value)
        if inner is None:
            return None
        return inner.strip()
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = parse_text_constant(node.left)
        right = parse_text_constant(node.right)
        if left is None or right is None:
            return None
        return left + right
    return None


def load_prompt_from_file(file_path: str, var_name: str) -> str:
    src = Path(file_path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    candidates: List[str] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == var_name for t in node.targets):
            continue
        value = parse_text_constant(node.value)
        if value is not None:
            candidates.append(value)
    if not candidates:
        raise ValueError(f"Cannot find prompt variable '{var_name}' in {file_path}")
    # Use the latest assignment (needed for inject_error_off_topic.py)
    return candidates[-1]


def step_label(step_text: str) -> Optional[str]:
    m = re.search(r"\((Attribution|Logical|Final Answer)\)\s*$", step_text.strip())
    if not m:
        return None
    return m.group(1)


def parse_passages(value) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed]
        except Exception:
            return [text]
    return []


def normalize_ideal_steps(raw_steps) -> Optional[List[str]]:
    if not isinstance(raw_steps, list) or not raw_steps:
        return None
    converted = []
    for item in raw_steps:
        if isinstance(item, dict):
            step_text = item.get("ideal_step")
        else:
            step_text = item
        if not isinstance(step_text, str):
            return None
        step_text = step_text.strip()
        if not step_text:
            return None
        converted.append(step_text)
    if not converted:
        return None
    return converted


def parse_answer_candidate(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        for item in value:
            text = str(item).strip()
            if text:
                return text
        return ""

    text = str(value).strip()
    if not text:
        return ""

    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return text

    if isinstance(parsed, list):
        for item in parsed:
            candidate = str(item).strip()
            if candidate:
                return candidate
        return ""

    return str(parsed).strip()


def normalize_final_answer_value(raw_answer) -> str:
    answer = parse_answer_candidate(raw_answer)
    if not answer:
        return ""
    answer = re.sub(r"^Step\s*\d+\s*:\s*", "", answer).strip()
    answer = re.sub(r"^####ANSWER:\s*", "", answer).strip()
    answer = re.sub(r"\s*\(Final Answer\)\s*$", "", answer).strip()

    if (answer.startswith('"') and answer.endswith('"')) or (
        answer.startswith("'") and answer.endswith("'")
    ):
        answer = answer[1:-1].strip()
    return answer


def build_correct_final_step(ideal_steps: List[str], raw_answer) -> Optional[str]:
    answer = normalize_final_answer_value(raw_answer)
    if not answer:
        return None
    step_idx = len(ideal_steps) + 1
    return f"Step {step_idx}: ####ANSWER: {answer} (Final Answer)"


def extract_sample_size_tag(sample_path: Path) -> str:
    m = re.search(r"_sampled_(\d+)_input\.json$", sample_path.name)
    if not m:
        return "200"
    return m.group(1)


def load_dataset_records(
    input_root: Path,
    dataset: str,
    max_items: int,
) -> Tuple[List[dict], Dict[str, int], str]:
    ideal_path = input_root / "ideal_steps" / f"{dataset}_sampled_200_ideal_steps.json"
    sample_path = input_root / "samples" / f"{dataset}_sampled_200_input.json"
    sample_tag = extract_sample_size_tag(sample_path)

    ideal_data = json.loads(ideal_path.read_text(encoding="utf-8"))
    sample_data = json.loads(sample_path.read_text(encoding="utf-8"))

    sample_map = {}
    for row in sample_data:
        question = str(row.get("question", "")).strip()
        if question and question not in sample_map:
            sample_map[question] = row

    records = []
    stats = {
        "ideal_total": len(ideal_data),
        "sample_total": len(sample_data),
        "joined_total": 0,
        "missing_sample_total": 0,
        "invalid_ideal_total": 0,
        "invalid_passages_total": 0,
        "missing_answer_total": 0,
    }

    for row in ideal_data:
        question = str(row.get("question", "")).strip()
        if not question:
            stats["invalid_ideal_total"] += 1
            continue

        sample_row = sample_map.get(question)
        if sample_row is None:
            stats["missing_sample_total"] += 1
            continue

        ideal_steps = normalize_ideal_steps(row.get("ideal_steps"))
        if ideal_steps is None:
            stats["invalid_ideal_total"] += 1
            continue

        retrieved_passages = parse_passages(sample_row.get("gt_passages"))
        if not retrieved_passages:
            stats["invalid_passages_total"] += 1
            continue

        if not normalize_final_answer_value(sample_row.get("gt_answer")):
            stats["missing_answer_total"] += 1

        records.append(
            {
                "question": question,
                "dataset": dataset,
                "sample_index": sample_row.get("sample_index"),
                "retrieved_passages": retrieved_passages,
                "ideal_steps": ideal_steps,
                "gt_answer": sample_row.get("gt_answer"),
            }
        )
        stats["joined_total"] += 1

    if max_items > 0:
        records = records[:max_items]

    return records, stats, sample_tag


def choose_target_index(
    policy: str, ideal_steps: List[str], rng: random.Random
) -> Tuple[Optional[int], List[int]]:
    total = len(ideal_steps)
    if total == 0:
        return None, []

    if policy == "logical":
        candidates = [i + 1 for i, step in enumerate(ideal_steps) if step_label(step) == "Logical"]
    elif policy == "attribution":
        candidates = [
            i + 1 for i, step in enumerate(ideal_steps) if step_label(step) == "Attribution"
        ]
    elif policy == "all":
        candidates = list(range(1, total + 1))
    elif policy == "from2":
        candidates = list(range(2, total + 1))
    elif policy == "overthinking":
        return total + 1, [total + 1]
    elif policy == "premature_step1":
        return 1, [1]
    elif policy == "final_answer_last":
        return total + 1, [total + 1]
    else:
        raise ValueError(f"Unknown target policy: {policy}")

    if not candidates:
        return None, []
    return rng.choice(candidates), candidates


def build_user_prompt(error_type: str, record: dict, target_index: int) -> str:
    question = record["question"]
    passages = "\n".join(
        f"Passage {idx + 1}: {p}" for idx, p in enumerate(record["retrieved_passages"])
    )
    ideal_steps_json = json.dumps(record["ideal_steps"], indent=2, ensure_ascii=False)

    if error_type == "overthinking":
        return (
            f"Question: {question}\n\n"
            f"Retrieved Passages:\n{passages}\n\n"
            f"Ideal Reasoning Steps:\n{ideal_steps_json}"
        )
    if error_type == "premature_attribution":
        return f"Question: {question}\nPassages:\n{passages}"
    if error_type == "wrong_conclusion":
        reasoning_steps_text = "\n".join(record["ideal_steps"])
        return (
            f"## Question ##\n{question}\n\n"
            f"## Reasoning Steps ##\n{reasoning_steps_text}\n\n"
            f"## Correct Step ##\n{record['correct_final_step']}"
        )

    return (
        f"Question: {question}\n\n"
        f"Retrieved Passages:\n{passages}\n\n"
        f"Ideal Reasoning Steps:\n{ideal_steps_json}\n\n"
        f"Target Step to Corrupt:\n"
        f"Step {target_index}"
    )


def extract_step_text(raw_text: str, expected_index: int) -> Optional[str]:
    text = raw_text.split("assistantfinal")[-1].strip()
    text = re.sub(r"```(?:json|python)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()

    pattern = (
        rf"(Step\s*{expected_index}\s*:.*?(?:\((?:Attribution|Logical)\)))"
    )
    m = re.search(pattern, text, re.DOTALL)
    if m:
        out = m.group(1).strip()
        out = re.sub(r"\s+", " ", out)
        return out

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in lines:
        if re.match(rf"^Step\s*{expected_index}\s*:", line) and re.search(
            r"\((Attribution|Logical)\)\s*$", line
        ):
            return re.sub(r"\s+", " ", line)
    return None


def parse_python_list(text: str):
    text = text.strip().split("assistantfinal")[-1].strip()
    text = re.sub(r"```(?:json|python)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()
    m = re.search(r"(\[.*\])", text, re.DOTALL)
    if m:
        content = m.group(1).strip()
    else:
        content = text
    try:
        return ast.literal_eval(content)
    except Exception:
        return content


def extract_premature_step(raw_text: str) -> Optional[str]:
    parsed = parse_python_list(raw_text)
    if isinstance(parsed, list) and parsed:
        first = parsed[0]
        if isinstance(first, str):
            step = re.sub(r"\s+", " ", first.strip())
            return step
    if isinstance(parsed, str):
        m = re.search(r"(Step\s*1\s*:.*?(?:\((?:Attribution|Logical)\)))", parsed, re.DOTALL)
        if m:
            return re.sub(r"\s+", " ", m.group(1).strip())
    return None


def robust_json_parse(text: str) -> Optional[dict]:
    if not text:
        return None

    text = text.split("assistantfinal")[-1].strip()
    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)

    start_idx = text.find("{")
    if start_idx == -1:
        return None
    json_candidate = text[start_idx:].strip()

    candidates = [json_candidate]
    end_idx = json_candidate.rfind("}")
    if end_idx != -1:
        candidates.append(json_candidate[: end_idx + 1])

    completion_patterns = ["}", '"}', '"]', "}}", '"}}', "}]", '"}]']
    for pattern in completion_patterns:
        candidates.append(json_candidate + pattern)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    for candidate in candidates:
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (SyntaxError, ValueError):
            continue
    return None


def validate_step(step_text: str, expected_index: int, expected_label: str) -> Optional[str]:
    if not re.match(rf"^Step\s*{expected_index}\s*:", step_text):
        return "wrong_step_prefix"
    label = step_label(step_text)
    if label is None:
        return "missing_step_label"
    if expected_label != "Either" and label != expected_label:
        return f"wrong_step_label:{label}"
    return None


def validate_final_answer_step(step_text: str, expected_index: int) -> Tuple[Optional[str], str]:
    normalized = re.sub(r"\s+", " ", step_text.strip())
    m = re.match(
        r"^Step\s*(\d+)\s*:\s*####ANSWER:\s*(.+?)\s*\(Final Answer\)\s*$",
        normalized,
    )
    if not m:
        return "wrong_final_format", normalized

    actual_index = int(m.group(1))
    if actual_index != expected_index:
        return "step_index_mismatch", normalized

    if not m.group(2).strip():
        return "wrong_final_format", normalized
    return None, normalized


def init_llm_and_tokenizer(args: argparse.Namespace):
    from transformers import AutoTokenizer
    from vllm import LLM

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    return llm, tokenizer


def run_generation_for_type(
    args: argparse.Namespace,
    llm,
    tokenizer,
    dataset_records: List[dict],
    config: ErrorTypeConfig,
    system_prompt: str,
    rng: random.Random,
) -> Tuple[List[dict], List[dict], Dict[str, int], List[dict]]:
    from vllm import SamplingParams

    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    outputs = []
    failures = []
    generation_logs = []
    stats = {
        "input_records": len(dataset_records),
        "generated_records": 0,
        "failed_records": 0,
        "skipped_no_candidate": 0,
        "skipped_missing_answer": 0,
    }

    pending = []
    for record in dataset_records:
        target_index, candidates = choose_target_index(config.select_policy, record["ideal_steps"], rng)
        if target_index is None:
            stats["skipped_no_candidate"] += 1
            failures.append(
                {
                    "question": record["question"],
                    "reason": "no_target_candidates",
                    "candidates": candidates,
                }
            )
            continue

        prompt_record = record
        if config.key == "wrong_conclusion":
            correct_final_step = build_correct_final_step(record["ideal_steps"], record.get("gt_answer"))
            if correct_final_step is None:
                stats["skipped_missing_answer"] += 1
                failures.append(
                    {
                        "question": record["question"],
                        "reason": "missing_gt_answer",
                        "target_index": target_index,
                    }
                )
                continue
            prompt_record = dict(record)
            prompt_record["correct_final_step"] = correct_final_step

        user_prompt = build_user_prompt(config.key, prompt_record, target_index)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        full_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        pending.append((prompt_record, target_index, full_prompt))

    for i in range(0, len(pending), args.batch_size):
        batch = pending[i : i + args.batch_size]
        prompts = [x[2] for x in batch]
        generations = llm.generate(prompts, sampling_params, use_tqdm=False)

        for (record, target_index, _), generation in zip(batch, generations):
            raw_text = generation.outputs[0].text if generation.outputs else ""
            if config.key == "wrong_conclusion":
                parsed_json = robust_json_parse(raw_text)
                if parsed_json is None:
                    stats["failed_records"] += 1
                    failures.append(
                        {
                            "question": record["question"],
                            "reason": "cannot_parse_json",
                            "target_index": target_index,
                            "raw_output": raw_text,
                        }
                    )
                    continue

                required_keys = ("generated_wrong_step", "diagnosis", "guidance")
                missing_keys = [
                    key for key in required_keys if not isinstance(parsed_json.get(key), str)
                ]
                empty_keys = [
                    key for key in required_keys if isinstance(parsed_json.get(key), str) and not parsed_json.get(key).strip()
                ]
                if missing_keys or empty_keys:
                    stats["failed_records"] += 1
                    failures.append(
                        {
                            "question": record["question"],
                            "reason": "missing_keys",
                            "target_index": target_index,
                            "missing_keys": missing_keys,
                            "empty_keys": empty_keys,
                            "parsed_json": parsed_json,
                            "raw_output": raw_text,
                        }
                    )
                    continue

                diagnosis = parsed_json["diagnosis"].strip()
                guidance = parsed_json["guidance"].strip()
                validation_error, wrong_step = validate_final_answer_step(
                    parsed_json["generated_wrong_step"],
                    target_index,
                )
                if validation_error is not None:
                    stats["failed_records"] += 1
                    failures.append(
                        {
                            "question": record["question"],
                            "reason": validation_error,
                            "target_index": target_index,
                            "parsed_step": wrong_step,
                            "parsed_json": parsed_json,
                            "raw_output": raw_text,
                        }
                    )
                    continue

                ideal_steps = record["ideal_steps"] + [record["correct_final_step"]]
                corrupted_steps = record["ideal_steps"] + [wrong_step]
                if len(ideal_steps) != target_index or len(corrupted_steps) != target_index:
                    stats["failed_records"] += 1
                    failures.append(
                        {
                            "question": record["question"],
                            "reason": "step_index_mismatch",
                            "target_index": target_index,
                            "ideal_steps_len": len(ideal_steps),
                            "corrupted_steps_len": len(corrupted_steps),
                            "parsed_step": wrong_step,
                        }
                    )
                    continue

                outputs.append(
                    {
                        "question": record["question"],
                        "retrieved_passages": record["retrieved_passages"],
                        "ideal_steps": ideal_steps,
                        "corrupted_steps": corrupted_steps,
                        "corrupted_step_index": target_index,
                        "error_type": config.record_error_type,
                        "dataset": record["dataset"],
                        "sample_index": record["sample_index"],
                    }
                )
                generation_logs.append(
                    {
                        "question": record["question"],
                        "dataset": record["dataset"],
                        "sample_index": record["sample_index"],
                        "correct_final_step": record["correct_final_step"],
                        "generated_wrong_step": wrong_step,
                        "diagnosis": diagnosis,
                        "guidance": guidance,
                    }
                )
                stats["generated_records"] += 1
                continue

            if config.key == "premature_attribution":
                step_text = extract_premature_step(raw_text)
            else:
                step_text = extract_step_text(raw_text, target_index)

            if step_text is None:
                stats["failed_records"] += 1
                failures.append(
                    {
                        "question": record["question"],
                        "reason": "cannot_parse_step",
                        "target_index": target_index,
                        "raw_output": raw_text,
                    }
                )
                continue

            validation_error = validate_step(step_text, target_index, config.expected_label)
            if validation_error is not None:
                stats["failed_records"] += 1
                failures.append(
                    {
                        "question": record["question"],
                        "reason": validation_error,
                        "target_index": target_index,
                        "parsed_step": step_text,
                        "raw_output": raw_text,
                    }
                )
                continue

            if config.key == "overthinking":
                corrupted_steps = record["ideal_steps"] + [step_text]
            else:
                corrupted_steps = record["ideal_steps"][: target_index - 1] + [step_text]

            if len(corrupted_steps) != target_index:
                stats["failed_records"] += 1
                failures.append(
                    {
                        "question": record["question"],
                        "reason": "length_index_mismatch",
                        "target_index": target_index,
                        "corrupted_steps_len": len(corrupted_steps),
                        "parsed_step": step_text,
                    }
                )
                continue

            outputs.append(
                {
                    "question": record["question"],
                    "retrieved_passages": record["retrieved_passages"],
                    "ideal_steps": record["ideal_steps"],
                    "corrupted_steps": corrupted_steps,
                    "corrupted_step_index": target_index,
                    "error_type": config.record_error_type,
                    "dataset": record["dataset"],
                    "sample_index": record["sample_index"],
                }
            )
            stats["generated_records"] += 1

    return outputs, failures, stats, generation_logs



FEEDBACK_STANDARD_FEWSHOT_ERROR_TYPES = {
    "logical_fallacy",
    "information_miss",
    "inefficiency",
    "redundancy",
    "off_topic",
    "overthinking",
    "contradictory",
    "unsupported",
}

FEEDBACK_SYSTEM_PROMPT_TEMPLATE = """You are an expert reasoning evaluator and verifier.
Your task is to analyze a reasoning step that contains a specific error and provide critical feedback.

You will receive:
1. Question: The user's original question.
2. Retrieved Passages: Contextual information.
3. Ideal Reasoning Steps: The *correct, full* reasoning path to the answer.
4. Previous Steps: The reasoning steps *before* the error.
5. Current Step: The reasoning step which contains the error.

The error in the Current Step is a "{error_type}" error.

Your feedback MUST:
1.  **Diagnose the Problem:** Clearly explain *why* the Current Step is wrong (e.g., "it's off-topic," "it contradicts Passage 3," "it repeats Step 1," etc.).
2.  **Propose a Solution:** Provide a *clear, actionable instruction* for what the *next one* reasoning step should be. **Use the `Ideal Reasoning Steps` as the "answer key"** to identify the correct step that *should have* followed the `Previous Steps`.
3.  **Be Concise:** Deliver *only the core message* for the diagnosis and solution. Do not add verbose explanations or unnecessary conversational filler.

Output *only* the feedback text, and nothing else.

---
Examples
---
{fewshot_examples}
""".strip()

PREMATURE_ATTRIBUTION_FEEDBACK_SYSTEM_PROMPT = """You are an expert reasoning evaluator.
Your task is to analyze a reasoning step that contains a "Premature Attribution" error and provide critical feedback.

You will receive:
1. Question: The user's original question.
2. Retrieved Passages: Contextual information.
3. Ideal Reasoning Steps (if available): The correct full reasoning path.
4. Previous Steps (if available): The steps taken so far.
5. Current Step: The step containing the Premature Attribution error.

The error is that the step extracts an attribute too early, before establishing the identification link between the question subject and the referenced entity.

Output ONLY a valid JSON object with keys "error_type", "diagnosis", and "guidance".
- error_type: ALWAYS "Premature Attribution"
- diagnosis: explain the missing linking step.
- guidance: provide exactly one immediate next step to establish that link.
""".strip()


def load_dict_constant_from_file(file_path: str, var_name: str) -> Dict[str, str]:
    src = Path(file_path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == var_name for t in node.targets):
            continue
        try:
            parsed = ast.literal_eval(node.value)
        except Exception as e:
            raise ValueError(f"Failed to parse dict constant '{var_name}' in {file_path}: {e}")
        if not isinstance(parsed, dict):
            raise ValueError(f"Constant '{var_name}' in {file_path} is not dict.")
        return {str(k): str(v) for k, v in parsed.items()}
    raise ValueError(f"Cannot find dict constant '{var_name}' in {file_path}")


def feedback_convert_template_to_json(system_prompt: str) -> str:
    target = "Output *only* the feedback text, and nothing else."
    replacement = (
        'Output ONLY a valid JSON object with keys "diagnosis" and "guidance". '
        "Do not output markdown, code fences, or extra prose."
    )
    if target in system_prompt:
        system_prompt = system_prompt.replace(target, replacement)
    else:
        system_prompt = system_prompt.strip() + "\n\n" + replacement

    system_prompt = system_prompt.strip() + "\n\n" + 'JSON schema: {"diagnosis": "...", "guidance": "..."}'
    return system_prompt


def build_feedback_system_prompts() -> Dict[str, str]:
    fewshot_dict = load_dict_constant_from_file(
        "/workspace/daeyong/feedback_fewshot_examples.py",
        "fewshot_dict",
    )
    wrong_conclusion_prompt = load_prompt_from_file(
        "/workspace/daeyong/oss_final_answer_swap_feedback.py",
        "FEEDBACK_SYSTEM_PROMPT",
    )

    prompts: Dict[str, str] = {}
    for error_type in FEEDBACK_STANDARD_FEWSHOT_ERROR_TYPES:
        display = ERROR_CONFIGS[error_type].record_error_type
        prompts[error_type] = feedback_convert_template_to_json(
            FEEDBACK_SYSTEM_PROMPT_TEMPLATE.format(
                error_type=display,
                fewshot_examples=fewshot_dict.get(error_type, ""),
            )
        )

    prompts["premature_attribution"] = PREMATURE_ATTRIBUTION_FEEDBACK_SYSTEM_PROMPT
    prompts["wrong_conclusion"] = wrong_conclusion_prompt
    return prompts


def feedback_extract_diagnosis_guidance(parsed: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not isinstance(parsed, dict):
        return None, None, None

    diagnosis = parsed.get("diagnosis")
    guidance = parsed.get("guidance")
    error_type = parsed.get("error_type")

    if isinstance(diagnosis, str) and isinstance(guidance, str):
        return diagnosis.strip(), guidance.strip(), error_type if isinstance(error_type, str) else None

    for nested_key in ("feedback", "gold_feedback"):
        nested = parsed.get(nested_key)
        if isinstance(nested, dict):
            d = nested.get("diagnosis")
            g = nested.get("guidance")
            e = nested.get("error_type")
            if isinstance(d, str) and isinstance(g, str):
                return d.strip(), g.strip(), e if isinstance(e, str) else None

    return None, None, None


def feedback_parse_passages_to_text(passages_value: Any) -> str:
    if isinstance(passages_value, list):
        return "\n".join(f"Passage {idx + 1}: {p}" for idx, p in enumerate(passages_value))
    if isinstance(passages_value, str):
        text = passages_value.strip()
        if not text:
            return ""
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return "\n".join(f"Passage {idx + 1}: {p}" for idx, p in enumerate(parsed))
        except Exception:
            pass
        return text
    return str(passages_value)


def feedback_extract_step_context(item: dict) -> Tuple[Optional[List[str]], Optional[str], Optional[int], Optional[str]]:
    corrupted_steps = item.get("corrupted_steps")
    raw_idx = item.get("corrupted_step_index")

    if isinstance(corrupted_steps, list) and corrupted_steps:
        try:
            step_idx = int(raw_idx)
        except Exception:
            step_idx = len(corrupted_steps)

        if step_idx < 1 or step_idx > len(corrupted_steps):
            return None, None, None, "invalid_index"

        current_step = corrupted_steps[step_idx - 1]
        if not isinstance(current_step, str) or not current_step.strip():
            return None, None, None, "invalid_step"
        previous_steps = corrupted_steps[: step_idx - 1]
        return previous_steps, current_step, step_idx, None

    for field in ("generated_premature_reasoning", "corrupted_response"):
        seq = item.get(field)
        if isinstance(seq, str):
            try:
                parsed = ast.literal_eval(seq)
                if isinstance(parsed, list):
                    seq = parsed
            except Exception:
                seq = []
        if isinstance(seq, list) and seq:
            first = seq[0]
            if isinstance(first, str) and first.strip():
                return [], first.strip(), 1, None

    current_step = item.get("current_step")
    if isinstance(current_step, str) and current_step.strip():
        prev = item.get("previous_steps")
        if not isinstance(prev, list):
            prev = []
        return prev, current_step.strip(), len(prev) + 1, None

    return None, None, None, "invalid_index"


def feedback_build_user_prompt(item: dict, previous_steps: List[str], current_step: str) -> str:
    question = str(item.get("question", "")).strip()
    passages_text = feedback_parse_passages_to_text(item.get("retrieved_passages", []))
    ideal_steps = item.get("ideal_steps", [])
    if not isinstance(ideal_steps, list):
        ideal_steps = []

    return (
        f"Question: {question}\n\n"
        f"Retrieved Passages:\n{passages_text}\n\n"
        f"Ideal Reasoning Steps:\n{json.dumps(ideal_steps, indent=2, ensure_ascii=False)}\n\n"
        f"Previous Steps:\n{json.dumps(previous_steps, indent=2, ensure_ascii=False)}\n\n"
        f"Current Step:\n{json.dumps(current_step, ensure_ascii=False)}\n\n"
        'Output JSON with keys "diagnosis" and "guidance".'
    )


def feedback_load_json_list(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"JSON is not a list: {path}")
    return data


def feedback_save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def feedback_record_key(item: dict) -> str:
    question = str(item.get("question", "")).strip()
    _, current_step, _, _ = feedback_extract_step_context(item)
    current = current_step or ""
    return f"{question}|||{current}"


def feedback_merge_existing_by_key(base_items: List[dict], existing_items: List[dict]) -> None:
    existing_map = {}
    for item in existing_items:
        if isinstance(item, dict):
            existing_map[feedback_record_key(item)] = item

    for idx, item in enumerate(base_items):
        if not isinstance(item, dict):
            continue
        old = existing_map.get(feedback_record_key(item))
        if old is None:
            continue
        for key in ("diagnosis", "guidance", "feedback_error_type", "raw_feedback_output", "feedback", "gold_feedback"):
            if key in old:
                item[key] = old[key]


def feedback_has_valid_feedback(item: dict, output_field: str) -> bool:
    if output_field == "top_level":
        d = item.get("diagnosis")
        g = item.get("guidance")
        return isinstance(d, str) and d.strip() and isinstance(g, str) and g.strip()

    nested = item.get(output_field)
    if isinstance(nested, dict):
        d = nested.get("diagnosis")
        g = nested.get("guidance")
        return isinstance(d, str) and d.strip() and isinstance(g, str) and g.strip()
    return False


def feedback_apply_result(
    item: dict,
    diagnosis: str,
    guidance: str,
    feedback_error_type: str,
    raw_output: str,
    output_field: str,
    save_raw: bool,
    write_top_level: bool,
) -> None:
    if output_field == "top_level":
        item["diagnosis"] = diagnosis
        item["guidance"] = guidance
        if feedback_error_type:
            item["feedback_error_type"] = feedback_error_type
        if save_raw:
            item["raw_feedback_output"] = raw_output
        return

    payload = {
        "error_type": feedback_error_type if feedback_error_type else "",
        "diagnosis": diagnosis,
        "guidance": guidance,
    }
    item[output_field] = payload
    if write_top_level:
        item["diagnosis"] = diagnosis
        item["guidance"] = guidance
        if feedback_error_type:
            item["feedback_error_type"] = feedback_error_type
    if save_raw:
        item["raw_feedback_output"] = raw_output


def feedback_process_file(
    args: argparse.Namespace,
    llm,
    tokenizer,
    system_prompt: str,
    input_file: Path,
    output_file: Path,
    failure_log_file: Path,
    dataset: str,
    error_type: str,
) -> dict:
    input_items = feedback_load_json_list(input_file)
    output_items = [dict(item) if isinstance(item, dict) else {"_raw_item": item} for item in input_items]

    if output_file.exists() and not args.overwrite:
        try:
            existing_items = feedback_load_json_list(output_file)
            feedback_merge_existing_by_key(output_items, existing_items)
        except Exception:
            pass

    failures: List[dict] = []
    prompts = []
    meta = []
    skipped_existing = 0
    selected_count = 0

    output_field = args.feedback_output_field
    write_top_level = args.feedback_write_top_level

    for idx, item in enumerate(output_items):
        if not isinstance(item, dict):
            failures.append(
                {
                    "item_index": idx,
                    "dataset": dataset,
                    "error_type": error_type,
                    "reason": "invalid_item_type",
                }
            )
            continue

        if not args.overwrite and feedback_has_valid_feedback(item, output_field):
            skipped_existing += 1
            continue

        if args.max_items_per_file > 0 and selected_count >= args.max_items_per_file:
            continue
        selected_count += 1

        previous_steps, current_step, step_idx, parse_error = feedback_extract_step_context(item)
        if parse_error is not None or current_step is None:
            failures.append(
                {
                    "item_index": idx,
                    "dataset": dataset,
                    "error_type": error_type,
                    "sample_index": item.get("sample_index"),
                    "question": str(item.get("question", ""))[:200],
                    "reason": "invalid_index",
                    "corrupted_step_index": item.get("corrupted_step_index"),
                }
            )
            continue

        user_prompt = feedback_build_user_prompt(item, previous_steps or [], current_step)

        if not args.dry_run:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            full_prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            prompts.append(full_prompt)

        meta.append(
            {
                "item_index": idx,
                "sample_index": item.get("sample_index"),
                "question": str(item.get("question", ""))[:200],
                "step_idx": step_idx,
            }
        )

    succeeded = 0
    failed = len(failures)
    processed = selected_count

    if args.dry_run:
        feedback_save_json(failure_log_file, failures)
        feedback_save_json(output_file, output_items)
        return {
            "dataset": dataset,
            "error_type": error_type,
            "input_file": str(input_file),
            "output_file": str(output_file),
            "failure_log_file": str(failure_log_file),
            "input_count": len(input_items),
            "processed": processed,
            "succeeded": succeeded,
            "failed": failed,
            "skipped_existing": skipped_existing,
            "status": "dry_run",
        }

    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.feedback_max_tokens,
        seed=args.seed,
    )

    for start in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[start : start + args.batch_size]
        batch_meta = meta[start : start + args.batch_size]
        generations = llm.generate(batch_prompts, sampling_params, use_tqdm=False)

        for m, generation in zip(batch_meta, generations):
            raw_output = generation.outputs[0].text if generation.outputs else ""
            parsed = robust_json_parse(raw_output)
            if parsed is None:
                failed += 1
                failures.append(
                    {
                        "item_index": m["item_index"],
                        "dataset": dataset,
                        "error_type": error_type,
                        "sample_index": m["sample_index"],
                        "question": m["question"],
                        "reason": "cannot_parse_json",
                        "raw_output": raw_output,
                    }
                )
                continue

            diagnosis, guidance, feedback_error_type = feedback_extract_diagnosis_guidance(parsed)
            if diagnosis is None or guidance is None or not diagnosis.strip() or not guidance.strip():
                failed += 1
                failures.append(
                    {
                        "item_index": m["item_index"],
                        "dataset": dataset,
                        "error_type": error_type,
                        "sample_index": m["sample_index"],
                        "question": m["question"],
                        "reason": "missing_or_empty_fields",
                        "parsed_json": parsed,
                        "raw_output": raw_output,
                    }
                )
                continue

            out_item = output_items[m["item_index"]]
            feedback_apply_result(
                item=out_item,
                diagnosis=diagnosis,
                guidance=guidance,
                feedback_error_type=(feedback_error_type or ERROR_CONFIGS[error_type].record_error_type),
                raw_output=raw_output,
                output_field=output_field,
                save_raw=args.save_raw_feedback_output,
                write_top_level=write_top_level,
            )
            succeeded += 1

    feedback_save_json(output_file, output_items)
    feedback_save_json(failure_log_file, failures)

    status = "ok" if failed == 0 else "partial_failed"
    return {
        "dataset": dataset,
        "error_type": error_type,
        "input_file": str(input_file),
        "output_file": str(output_file),
        "failure_log_file": str(failure_log_file),
        "input_count": len(input_items),
        "processed": processed,
        "succeeded": succeeded,
        "failed": failed,
        "skipped_existing": skipped_existing,
        "status": status,
    }


def locate_feedback_input_file(input_root: Path, dataset: str, error_type: str) -> Optional[Path]:
    exact = input_root / error_type / f"{dataset}_sampled_200_{error_type}.json"
    if exact.exists():
        return exact
    candidates = sorted(input_root.glob(f"{error_type}/{dataset}_sampled_*_{error_type}.json"))
    if not candidates:
        return None
    return candidates[0]


def run_feedback_pipeline(args: argparse.Namespace) -> None:
    normalized_error_types = unique_preserve_order([normalize_error_type(x) for x in args.error_types])
    datasets = unique_preserve_order(list(args.datasets))

    system_prompts = build_feedback_system_prompts()

    llm = None
    tokenizer = None
    if not args.dry_run:
        print("🚀 Loading gpt-oss-120b with vLLM for feedback generation...")
        llm, tokenizer = init_llm_and_tokenizer(args)

    if args.feedback_input_file:
        if not args.feedback_output_file:
            raise ValueError("--feedback_output_file is required when --feedback_input_file is provided.")

        raw_error_type = args.feedback_error_type if args.feedback_error_type else (
            normalized_error_types[0] if normalized_error_types else None
        )
        if not raw_error_type:
            raise ValueError("--feedback_error_type is required for single-file feedback mode.")

        error_type = normalize_error_type(raw_error_type)
        system_prompt = system_prompts[error_type]

        input_file = Path(args.feedback_input_file)
        output_file = Path(args.feedback_output_file)
        failure_log_file = output_file.with_name(output_file.stem + "_feedback_failures.json")

        print(f"[run] single-file feedback error_type={error_type} file={input_file}")
        result = feedback_process_file(
            args=args,
            llm=llm,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            input_file=input_file,
            output_file=output_file,
            failure_log_file=failure_log_file,
            dataset="single",
            error_type=error_type,
        )
        print(
            f"[done] processed={result['processed']} succeeded={result['succeeded']} "
            f"failed={result['failed']} skipped_existing={result['skipped_existing']}"
        )
        return

    input_root = Path(args.input_root) / args.input_dir_name
    output_root = Path(args.input_root) / args.output_dir_name
    logs_root = output_root / "logs"
    output_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "args": {
            "datasets": datasets,
            "error_types": normalized_error_types,
            "model_path": args.model_path,
            "tensor_parallel_size": args.tensor_parallel_size,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "max_tokens": args.max_tokens,
            "feedback_max_tokens": args.feedback_max_tokens,
            "max_items_per_file": args.max_items_per_file,
            "overwrite": args.overwrite,
            "dry_run": args.dry_run,
            "save_raw_feedback_output": args.save_raw_feedback_output,
        },
        "files": [],
        "totals": {
            "files": 0,
            "input_items": 0,
            "processed": 0,
            "succeeded": 0,
            "failed": 0,
            "skipped_existing": 0,
            "missing_input_files": 0,
        },
    }

    for error_type in normalized_error_types:
        system_prompt = system_prompts.get(error_type)
        if system_prompt is None:
            raise ValueError(f"No system prompt available for error type '{error_type}'")

        for dataset in datasets:
            input_file = locate_feedback_input_file(input_root, dataset, error_type)
            if input_file is None:
                summary["totals"]["missing_input_files"] += 1
                summary["files"].append(
                    {
                        "dataset": dataset,
                        "error_type": error_type,
                        "status": "missing_input_file",
                    }
                )
                print(f"[skip] missing input: dataset={dataset} error_type={error_type}")
                continue

            output_file = output_root / error_type / input_file.name
            failure_log_file = logs_root / f"{dataset}_{error_type}_feedback_failures.json"

            print(f"[run] dataset={dataset} error_type={error_type} file={input_file.name}")
            file_result = feedback_process_file(
                args=args,
                llm=llm,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                input_file=input_file,
                output_file=output_file,
                failure_log_file=failure_log_file,
                dataset=dataset,
                error_type=error_type,
            )
            summary["files"].append(file_result)
            summary["totals"]["files"] += 1
            summary["totals"]["input_items"] += file_result["input_count"]
            summary["totals"]["processed"] += file_result["processed"]
            summary["totals"]["succeeded"] += file_result["succeeded"]
            summary["totals"]["failed"] += file_result["failed"]
            summary["totals"]["skipped_existing"] += file_result["skipped_existing"]

            print(
                f"[done] dataset={dataset} error_type={error_type} "
                f"processed={file_result['processed']} succeeded={file_result['succeeded']} "
                f"failed={file_result['failed']} skipped_existing={file_result['skipped_existing']}"
            )

    summary_path = output_root / "summary.json"
    feedback_save_json(summary_path, summary)
    print(f"🎉 Feedback completed. Summary saved: {summary_path}")


def run_injection_pipeline(args: argparse.Namespace):
    input_root = Path(args.input_root)
    out_dirs = ensure_dirs(input_root)

    normalized_error_types = [normalize_error_type(x) for x in args.error_types]
    normalized_error_types = unique_preserve_order(normalized_error_types)
    datasets = unique_preserve_order(list(args.datasets))

    prompts = {}
    for error_type in normalized_error_types:
        cfg = ERROR_CONFIGS[error_type]
        prompts[error_type] = load_prompt_from_file(cfg.prompt_file, cfg.prompt_var)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "args": {
            "datasets": datasets,
            "error_types": normalized_error_types,
            "model_path": args.model_path,
            "seed": args.seed,
            "max_items_per_dataset": args.max_items_per_dataset,
            "prepare_only": args.prepare_only,
        },
        "dataset_input_stats": {},
        "results": {},
    }

    records_by_dataset = {}
    sample_tag_by_dataset = {}
    for dataset in datasets:
        records, stats, sample_tag = load_dataset_records(
            input_root=input_root,
            dataset=dataset,
            max_items=args.max_items_per_dataset,
        )
        records_by_dataset[dataset] = records
        sample_tag_by_dataset[dataset] = sample_tag
        summary["dataset_input_stats"][dataset] = {
            **stats,
            "usable_records": len(records),
        }
        print(
            f"[prepare] {dataset}: ideal={stats['ideal_total']} sample={stats['sample_total']} "
            f"usable={len(records)}"
        )

    if args.prepare_only:
        summary_path = out_dirs["root"] / "summary.json"
        write_json(summary_path, summary, args.overwrite)
        print(f"✅ Prepare-only completed: {summary_path}")
        return

    print("🚀 Loading gpt-oss-120b with vLLM...")
    llm, tokenizer = init_llm_and_tokenizer(args)

    for dataset_idx, dataset in enumerate(datasets):
        dataset_records = records_by_dataset[dataset]
        summary["results"][dataset] = {}

        for error_idx, error_type in enumerate(normalized_error_types):
            cfg = ERROR_CONFIGS[error_type]
            rng = random.Random(args.seed + (dataset_idx * 1000) + error_idx)
            print(
                f"[run] dataset={dataset} error_type={error_type} items={len(dataset_records)}"
            )

            generated, failures, stats, generation_logs = run_generation_for_type(
                args=args,
                llm=llm,
                tokenizer=tokenizer,
                dataset_records=dataset_records,
                config=cfg,
                system_prompt=prompts[error_type],
                rng=rng,
            )

            type_dir = out_dirs["root"] / error_type
            output_path = (
                type_dir
                / f"{dataset}_sampled_{sample_tag_by_dataset[dataset]}_{error_type}.json"
            )
            failure_path = out_dirs["logs"] / f"{dataset}_{error_type}_failures.json"
            generation_log_path = out_dirs["logs"] / f"{dataset}_{error_type}_generation_logs.json"

            write_json(output_path, generated, args.overwrite)
            write_json(failure_path, failures, args.overwrite)
            if generation_logs:
                write_json(generation_log_path, generation_logs, args.overwrite)

            summary["results"][dataset][error_type] = {
                **stats,
                "output_path": str(output_path),
                "failure_path": str(failure_path),
                "generation_log_path": str(generation_log_path) if generation_logs else None,
            }

            print(
                f"[done] dataset={dataset} error_type={error_type} generated={stats['generated_records']} "
                f"failed={stats['failed_records']} skipped_no_candidate={stats['skipped_no_candidate']} "
                f"skipped_missing_answer={stats['skipped_missing_answer']}"
            )

    summary_path = out_dirs["root"] / "summary.json"
    write_json(summary_path, summary, args.overwrite)
    print(f"🎉 Completed. Summary: {summary_path}")


def main():
    args = parse_args()
    if args.task == "inject":
        run_injection_pipeline(args)
    else:
        run_feedback_pipeline(args)


if __name__ == "__main__":
    main()
