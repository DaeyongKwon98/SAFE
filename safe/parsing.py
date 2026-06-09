from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict


FINAL_ANSWER_RE = re.compile(
    r"####ANSWER:\s*(.*?)(?:\s*\(Final Answer\))?(?:\n|$)",
    re.IGNORECASE,
)


def clean_generation(text: str, step_number: int) -> str:
    value = str(text).strip()
    if "assistantfinal" in value.lower():
        match = re.search(r"assistantfinal", value, flags=re.IGNORECASE)
        if match:
            value = value[match.end() :].strip()
    value = re.sub(r"<think>.*?</think>", "", value, flags=re.DOTALL).strip()
    value = value.splitlines()[0].strip() if value else ""
    expected = f"Step {step_number}:"
    if not value.startswith("Step "):
        value = f"{expected} {value}".strip()
    if "####ANSWER:" in value and "(Final Answer)" not in value:
        value += " (Final Answer)"
    if not value.endswith(("(Attribution)", "(Logical)", "(Final Answer)")):
        if "According to Passage" in value:
            value += " (Attribution)"
        elif value:
            value += " (Logical)"
    return value


def extract_final_answer(value: Any) -> str:
    if isinstance(value, list):
        values = reversed(value)
    else:
        values = [value]
    for item in values:
        match = FINAL_ANSWER_RE.search(str(item or ""))
        if match:
            return match.group(1).strip()
    return ""


def parse_json_object(text: str) -> Dict[str, Any]:
    value = str(text).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", value, flags=re.DOTALL)
    if fence:
        value = fence.group(1).strip()
    start = value.find("{")
    end = value.rfind("}")
    if start >= 0 and end > start:
        value = value[start : end + 1]
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Model output did not contain a valid JSON object")


def parse_evaluation(text: str) -> Dict[str, str]:
    parsed = parse_json_object(text)
    required = ("error_type", "diagnosis", "guidance")
    missing = [key for key in required if not str(parsed.get(key, "")).strip()]
    if missing:
        raise ValueError(f"Evaluation JSON missing fields: {missing}")
    return {key: str(parsed[key]).strip() for key in required}

