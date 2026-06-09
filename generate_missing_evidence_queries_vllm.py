#!/usr/bin/env python3
"""Generate missing-evidence diagnosis and search-query guidance with vLLM.

This script is intended for fourth_finetuning_data/only_correct.json.
Existing diagnosis/guidance values in the input are ignored; the output file
contains newly generated values for those two fields.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

try:
    from tqdm import tqdm
except ImportError:
    class _SimpleProgress:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def update(self, n: int = 1) -> None:
            return None

        def close(self) -> None:
            return None

    def tqdm(iterable=None, **kwargs):
        return _SimpleProgress(iterable, **kwargs)


DEFAULT_INPUT_PATH = "fourth_finetuning_data/only_correct.json"
DEFAULT_OUTPUT_PATH = "fourth_finetuning_data/only_correct_missing_evidence_queries.json"
DEFAULT_MODEL_PATH = "/workspace/hf_transformers/gpt-oss-120b"


SYSTEM_PROMPT = """You are an expert retrieval-gap diagnostician for multi-hop question answering.

You will receive:
1. Question
2. Retrieved Passages currently available to the reasoning model
3. Previous Steps
4. Current Step

The retrieved passages may be missing one passage that is needed to support the reasoning chain. The missing evidence may be implied by a previous step, the current step, or an explicit reference such as "removed passage" or "missing passage" inside a step.

Your task:
1. Infer the specific missing passage content/evidence needed for the reasoning chain.
2. Write a search query that would retrieve that missing passage.

Rules for "diagnosis":
- Describe the missing passage content, not the quality of the existing reasoning.
- Mention the key entity, relation, attribute, date, value, or comparison that the missing passage should contain.
- Do not say "the provided diagnosis" or "the provided guidance"; those fields are not part of the input.
- Do not say "removed passage"; call it missing evidence or a missing passage.
- Be concise: one or two sentences.

Rules for "guidance":
- Output a search query for the missing passage.
- Use important entity names and relation keywords.
- Do not write a full advice sentence.

Output only a valid JSON object with exactly these keys:
{"diagnosis": "...", "guidance": "..."}""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate new diagnosis/guidance fields for missing-evidence "
            "search-query training data using gpt-oss-120b with vLLM."
        )
    )
    parser.add_argument("--input_path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tensor_parallel_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_passage_chars",
        type=int,
        default=0,
        help=(
            "If > 0, truncate each retrieved passage to this many characters "
            "before prompting. Default 0 keeps full passages."
        ),
    )
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="Exclusive end index. Defaults to len(data).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many rows after start_index/end_index filtering.",
    )
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate rows even if the output file already marks them done.",
    )
    parser.add_argument(
        "--allow_in_place",
        action="store_true",
        help="Allow output_path to equal input_path. A temp file is still used for writes.",
    )
    parser.add_argument(
        "--keep_debug_fields",
        action="store_true",
        help="Keep raw model output and parse status fields in the output JSON.",
    )
    return parser.parse_args()


def preload_vllm_torch_workaround() -> None:
    """Match the local vLLM/Torch import-order workaround used in inference_vllm.py."""

    os.environ.setdefault("DAEYONG_VLLM_TORCH_PRELOAD", "1")
    try:
        import torch._logging  # noqa: F401
        import torch._numpy  # noqa: F401
        from torch._dynamo import config as _torch_dynamo_config  # noqa: F401
        from torch._guards import detect_fake_mode as _torch_detect_fake_mode  # noqa: F401
        from torch._logging import LazyString as _TorchLazyString  # noqa: F401
        from torch._subclasses.fake_tensor import (  # noqa: F401
            FakeTensor as _TorchFakeTensor,
            is_fake as _torch_is_fake,
            maybe_get_fake_mode as _torch_maybe_get_fake_mode,
        )
    except Exception:
        # The workaround is best-effort; older environments may not need it.
        return


def load_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}")
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Expected object at row {idx}, got {type(item).__name__}")
    return data


def atomic_write_json(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
        suffix=".tmp",
    ) as tmp:
        json.dump(data, tmp, ensure_ascii=False, indent=2)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def chunked(values: list[int], batch_size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def maybe_truncate(text: Any, max_chars: int) -> str:
    value = str(text or "")
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + " ... [truncated]"


def format_passages(passages: Any, max_passage_chars: int) -> str:
    if not isinstance(passages, list):
        passages = [] if passages is None else [passages]
    lines = []
    for idx, passage in enumerate(passages, start=1):
        lines.append(f"Passage {idx}: {maybe_truncate(passage, max_passage_chars)}")
    return "\n\n".join(lines) if lines else "No retrieved passages."


def format_steps(steps: Any) -> str:
    if not isinstance(steps, list):
        steps = [] if steps is None else [steps]
    if not steps:
        return "None."
    return "\n".join(str(step) for step in steps)


def build_user_prompt(item: dict[str, Any], max_passage_chars: int) -> str:
    return f"""Question:
{item.get("question", "")}

Retrieved Passages:
{format_passages(item.get("retrieved_passages", []), max_passage_chars)}

Previous Steps:
{format_steps(item.get("previous_steps", []))}

Current Step:
{item.get("current_step", "")}

Return the missing-evidence diagnosis and the search-query guidance as JSON.""".strip()


def build_prompt(tokenizer: Any, item: dict[str, Any], max_passage_chars: int) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(item, max_passage_chars)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def clean_generation_text(text: str) -> str:
    value = str(text or "").strip()
    if "assistantfinal" in value:
        value = value.split("assistantfinal")[-1].strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE).strip()
    value = re.sub(r"\s*```$", "", value).strip()
    return value


def extract_balanced_json_objects(text: str) -> list[str]:
    objects = []
    depth = 0
    start = None
    in_string = False
    escape = False
    quote_char = ""

    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote_char:
                in_string = False
            continue

        if ch in {"'", '"'}:
            in_string = True
            quote_char = ch
            continue
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start : idx + 1])
                start = None
    return objects


def parse_key_value_fallback(text: str) -> dict[str, str] | None:
    diagnosis_match = re.search(
        r"diagnosis\s*[:=-]\s*(.*?)(?:\n\s*guidance\s*[:=-]|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    guidance_match = re.search(
        r"guidance\s*[:=-]\s*(.*)$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not diagnosis_match or not guidance_match:
        return None
    return {
        "diagnosis": diagnosis_match.group(1).strip().strip('"').strip("'"),
        "guidance": guidance_match.group(1).strip().strip('"').strip("'"),
    }


def parse_model_output(raw_text: str) -> tuple[dict[str, str] | None, str | None]:
    text = clean_generation_text(raw_text)
    candidates = [text]
    candidates.extend(extract_balanced_json_objects(text))

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        parsed = None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(candidate)
                break
            except Exception:
                continue
        if not isinstance(parsed, dict):
            continue

        diagnosis = str(parsed.get("diagnosis", "")).strip()
        guidance = str(parsed.get("guidance", "")).strip()
        if diagnosis and guidance:
            return {"diagnosis": diagnosis, "guidance": guidance}, None

    fallback = parse_key_value_fallback(text)
    if fallback and fallback["diagnosis"] and fallback["guidance"]:
        return fallback, None

    return None, f"Could not parse diagnosis/guidance JSON: {text[:300]}"


def is_done(item: dict[str, Any], source_item: dict[str, Any]) -> bool:
    if item.get("_missing_evidence_generated") is True:
        return True
    if item.get("_missing_evidence_generated") is False:
        return False

    diagnosis = str(item.get("diagnosis", "")).strip()
    guidance = str(item.get("guidance", "")).strip()
    if not diagnosis or not guidance:
        return False

    source_diagnosis = str(source_item.get("diagnosis", "")).strip()
    source_guidance = str(source_item.get("guidance", "")).strip()
    return diagnosis != source_diagnosis or guidance != source_guidance


def strip_debug_fields(item: dict[str, Any]) -> None:
    for key in (
        "_missing_evidence_generated",
        "_missing_evidence_raw_output",
        "_missing_evidence_parse_error",
    ):
        item.pop(key, None)


def init_llm_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    preload_vllm_torch_workaround()
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
        seed=args.seed,
    )
    return llm, tokenizer


def make_sampling_params(args: argparse.Namespace) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )


def select_indices(args: argparse.Namespace, data_len: int) -> list[int]:
    start = max(0, args.start_index)
    end = data_len if args.end_index is None else min(args.end_index, data_len)
    if end < start:
        return []
    indices = list(range(start, end))
    if args.limit is not None:
        indices = indices[: max(0, args.limit)]
    return indices


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if input_path.resolve() == output_path.resolve() and not args.allow_in_place:
        raise ValueError(
            "output_path equals input_path. Use a distinct output path or pass "
            "--allow_in_place if you intentionally want to overwrite the input."
        )

    source_data = load_json_records(input_path)
    if output_path.exists():
        print(f"Resuming from existing output: {output_path}")
        output_data = load_json_records(output_path)
        if len(output_data) != len(source_data):
            raise ValueError(
                f"Output length {len(output_data)} does not match input length {len(source_data)}"
            )
    else:
        output_data = [dict(item) for item in source_data]

    target_indices = select_indices(args, len(source_data))
    if not args.force:
        target_indices = [
            idx
            for idx in target_indices
            if not is_done(output_data[idx], source_data[idx])
        ]

    print(f"Input rows: {len(source_data)}")
    print(f"Rows to generate: {len(target_indices)}")
    if not target_indices:
        print("Nothing to do.")
        return

    llm, tokenizer = init_llm_and_tokenizer(args)
    sampling_params = make_sampling_params(args)

    processed_since_save = 0
    failures = 0
    progress = tqdm(target_indices, desc="Generating missing-evidence queries")

    for batch_indices in chunked(target_indices, args.batch_size):
        prompts = [
            build_prompt(tokenizer, source_data[idx], args.max_passage_chars)
            for idx in batch_indices
        ]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        prompt_by_idx = dict(zip(batch_indices, prompts))

        retry_items: list[tuple[int, str]] = []
        for idx, output in zip(batch_indices, outputs):
            raw_text = output.outputs[0].text if output.outputs else ""
            parsed, parse_error = parse_model_output(raw_text)

            for attempt in range(args.max_retries):
                if parsed is not None:
                    break
                retry_prompt = prompt_by_idx[idx] + (
                    "\n\nReturn only a valid JSON object with keys diagnosis and guidance."
                )
                retry_output = llm.generate([retry_prompt], sampling_params, use_tqdm=False)[0]
                raw_text = retry_output.outputs[0].text if retry_output.outputs else ""
                parsed, parse_error = parse_model_output(raw_text)

            if parsed is None:
                failures += 1
                output_data[idx]["_missing_evidence_generated"] = False
                output_data[idx]["_missing_evidence_raw_output"] = clean_generation_text(raw_text)
                output_data[idx]["_missing_evidence_parse_error"] = parse_error
                retry_items.append((idx, parse_error or "parse failed"))
            else:
                output_data[idx]["diagnosis"] = parsed["diagnosis"]
                output_data[idx]["guidance"] = parsed["guidance"]
                output_data[idx]["_missing_evidence_generated"] = True
                output_data[idx]["_missing_evidence_raw_output"] = clean_generation_text(raw_text)
                output_data[idx].pop("_missing_evidence_parse_error", None)

            processed_since_save += 1
            progress.update(1)

        if retry_items:
            for idx, error in retry_items[:3]:
                print(f"[parse failure] row={idx}: {error}")

        if processed_since_save >= args.save_every:
            save_data = [dict(item) for item in output_data]
            if not args.keep_debug_fields:
                for item in save_data:
                    if item.get("_missing_evidence_generated") is True:
                        strip_debug_fields(item)
            atomic_write_json(output_path, save_data)
            processed_since_save = 0

    progress.close()

    save_data = [dict(item) for item in output_data]
    if not args.keep_debug_fields:
        for item in save_data:
            strip_debug_fields(item)
    atomic_write_json(output_path, save_data)
    print(f"Saved: {output_path}")
    if failures:
        print(f"Rows with parse failures: {failures}")


if __name__ == "__main__":
    main()
