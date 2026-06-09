from __future__ import annotations

import ast
from typing import Any, Dict, Iterable, List


ERROR_TYPES = (
    "Correct (No Error)",
    "Contradictory",
    "Unsupported",
    "Logical Fallacy",
    "Information Miss",
    "Redundancy",
    "Overthinking",
    "Off-topic",
    "Inefficiency",
    "Premature Attribution",
    "Premature Conclusion",
    "Wrong Conclusion",
)


def _list_value(value: Any, field: str) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return [stripped]
        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip() for item in parsed if str(item).strip()]
        return [str(parsed).strip()]
    if value is None:
        return []
    raise ValueError(f"{field} must be a list or string")


def normalize_benchmark(record: Dict[str, Any]) -> Dict[str, Any]:
    missing = [
        key
        for key in ("id", "dataset", "question", "retrieved_passages")
        if record.get(key) in (None, "")
    ]
    if missing:
        raise ValueError(f"Benchmark record missing fields: {missing}")
    answers = _list_value(
        record.get("answers", record.get("answer_list", record.get("answer"))),
        "answers",
    )
    if not answers:
        raise ValueError("Benchmark record must contain at least one answer")
    return {
        "id": str(record["id"]),
        "dataset": str(record["dataset"]),
        "question": str(record["question"]).strip(),
        "retrieved_passages": _list_value(
            record["retrieved_passages"], "retrieved_passages"
        ),
        "answers": answers,
        **(
            {"ideal_steps": _list_value(record["ideal_steps"], "ideal_steps")}
            if record.get("ideal_steps") is not None
            else {}
        ),
    }


def normalize_evaluator(record: Dict[str, Any]) -> Dict[str, Any]:
    benchmark = normalize_benchmark(record)
    required = ("current_step", "error_type", "diagnosis", "guidance")
    missing = [key for key in required if record.get(key) in (None, "")]
    if missing:
        raise ValueError(f"Evaluator record missing fields: {missing}")
    error_type = str(record["error_type"]).strip()
    if error_type not in ERROR_TYPES:
        raise ValueError(f"Unsupported error_type: {error_type}")
    benchmark.update(
        {
            "previous_steps": _list_value(
                record.get("previous_steps", []), "previous_steps"
            ),
            "current_step": str(record["current_step"]).strip(),
            "error_type": error_type,
            "diagnosis": str(record["diagnosis"]).strip(),
            "guidance": str(record["guidance"]).strip(),
        }
    )
    return benchmark


def normalize_many(
    records: Iterable[Dict[str, Any]], evaluator: bool = False
) -> List[Dict[str, Any]]:
    normalizer = normalize_evaluator if evaluator else normalize_benchmark
    return [normalizer(record) for record in records]

