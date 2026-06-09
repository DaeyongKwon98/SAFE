from __future__ import annotations

import re
import string
from collections import Counter
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence

from .io import read_records, write_json
from .parsing import extract_final_answer


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value).lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, target: str) -> float:
    return float(prediction == target)


def token_f1(prediction: str, target: str) -> float:
    prediction_tokens = prediction.split()
    target_tokens = target.split()
    if not prediction_tokens and not target_tokens:
        return 1.0
    if not prediction_tokens or not target_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(target_tokens)
    shared = sum(common.values())
    if not shared:
        return 0.0
    precision = shared / len(prediction_tokens)
    recall = shared / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def score_record(record: Dict[str, Any]) -> Dict[str, Any]:
    prediction = normalize_text(
        record.get("final_answer") or extract_final_answer(record.get("response", []))
    )
    targets = [normalize_text(item) for item in record.get("answers", [])]
    if not targets:
        targets = [normalize_text(record.get("ground_truth", ""))]
    em = max((exact_match(prediction, target) for target in targets), default=0.0)
    f1 = max((token_f1(prediction, target) for target in targets), default=0.0)
    attempts = record.get("attempts", [])
    rejected = sum(item.get("result") == "rejected" for item in attempts)
    return {
        "id": record.get("id"),
        "dataset": record.get("dataset"),
        "mode": record.get("mode"),
        "prediction": prediction,
        "em": em,
        "f1": f1,
        "steps": len(record.get("response", [])),
        "retries": rejected,
        "status": record.get("status", ""),
    }


def summarize(scores: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not scores:
        return {"samples": 0, "em": 0.0, "f1": 0.0, "avg_steps": 0.0, "avg_retries": 0.0}
    return {
        "samples": len(scores),
        "em": mean(item["em"] for item in scores),
        "f1": mean(item["f1"] for item in scores),
        "avg_steps": mean(item["steps"] for item in scores),
        "avg_retries": mean(item["retries"] for item in scores),
    }


def evaluate_file(input_path: str, output_path: str) -> Dict[str, Any]:
    scores = [score_record(record) for record in read_records(input_path)]
    payload = {"summary": summarize(scores), "records": scores}
    write_json(output_path, payload)
    return payload

