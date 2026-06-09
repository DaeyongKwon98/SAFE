from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

from .backend import TextBackend, TransformersBackend
from .io import read_records, write_jsonl
from .parsing import parse_json_object
from .prompts import injection_messages
from .schema import ERROR_TYPES, normalize_evaluator, normalize_many


def correct_records(benchmarks: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for record in benchmarks:
        steps = record.get("ideal_steps", [])
        for index, step in enumerate(steps):
            is_final = index == len(steps) - 1
            guidance = (
                "Submit the concise final answer using ####ANSWER: <answer_value>."
                if is_final
                else "Continue with the next necessary atomic reasoning step."
            )
            output.append(
                {
                    **record,
                    "previous_steps": steps[:index],
                    "current_step": step,
                    "error_type": "Correct (No Error)",
                    "diagnosis": "The step is grounded and makes necessary progress.",
                    "guidance": guidance,
                }
            )
    return output


def inject_records(
    correct: Sequence[Dict[str, Any]],
    error_types: Sequence[str],
    backend: TextBackend,
    generation: Dict[str, Any],
) -> List[Dict[str, Any]]:
    invalid = [name for name in error_types if name not in ERROR_TYPES]
    if invalid:
        raise ValueError(f"Unsupported error types: {invalid}")
    targets = [
        (record, error_type)
        for record in correct
        for error_type in error_types
        if error_type != "Correct (No Error)"
    ]
    messages = [
        injection_messages(record, error_type)
        for record, error_type in targets
    ]
    results = backend.generate(messages, generation)
    output = []
    for (record, error_type), result in zip(targets, results):
        parsed = parse_json_object(result.text)
        candidate = {
            **record,
            "current_step": str(parsed.get("current_step", "")).strip(),
            "error_type": error_type,
            "diagnosis": str(parsed.get("diagnosis", "")).strip(),
            "guidance": str(parsed.get("guidance", "")).strip(),
        }
        output.append(normalize_evaluator(candidate))
    return output


def prepare_data(
    benchmark_path: str,
    output_path: str,
    config: Dict[str, Any],
    annotation_paths: Sequence[str] = (),
    error_types: Sequence[str] = (),
    limit: int = 0,
    backend: TextBackend | None = None,
) -> List[Dict[str, Any]]:
    benchmarks = normalize_many(read_records(benchmark_path))
    if limit > 0:
        benchmarks = benchmarks[:limit]

    prepared: List[Dict[str, Any]] = []
    for path in annotation_paths:
        prepared.extend(
            normalize_evaluator(record) for record in read_records(path)
        )

    correct = correct_records(benchmarks)
    prepared.extend(normalize_evaluator(record) for record in correct)

    if error_types:
        injector = backend or TransformersBackend.load(
            config["models"]["generator"],
            config,
        )
        prepared.extend(
            inject_records(
                correct,
                error_types,
                injector,
                config["generation"]["evaluator"],
            )
        )

    if not prepared:
        raise ValueError(
            "No evaluator records were produced. Supply ideal_steps or annotations."
        )
    write_jsonl(output_path, prepared)
    return prepared

