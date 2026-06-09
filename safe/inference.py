from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .backend import TextBackend, TransformersBackend
from .config import dump_config
from .io import read_records, write_json, write_jsonl
from .parsing import clean_generation, extract_final_answer, parse_evaluation
from .prompts import evaluator_messages, generator_messages
from .schema import normalize_many


@dataclass
class ReasoningState:
    record: Dict[str, Any]
    steps: List[str] = field(default_factory=list)
    feedback: List[Dict[str, str]] = field(default_factory=list)
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    pending_feedback: Dict[str, str] | None = None
    retries: int = 0
    finished: bool = False
    status: str = "running"


def _evaluate_with_repair(
    backend: TextBackend,
    messages: List[Dict[str, str]],
    generation: Dict[str, Any],
    parse_retries: int = 2,
) -> Dict[str, str]:
    current_messages = list(messages)
    last_error: Exception | None = None
    for attempt in range(parse_retries + 1):
        result = backend.generate([current_messages], generation)[0]
        try:
            return parse_evaluation(result.text)
        except ValueError as exc:
            last_error = exc
            current_messages = list(messages) + [
                {"role": "assistant", "content": result.text},
                {
                    "role": "user",
                    "content": (
                        "Return the same evaluation as valid JSON only, with non-empty "
                        "error_type, diagnosis, and guidance fields."
                    ),
                },
            ]
    raise ValueError(f"Evaluator JSON parsing failed: {last_error}")


def run_states(
    records: Iterable[Dict[str, Any]],
    mode: str,
    generator: TextBackend,
    evaluator: TextBackend | None,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if mode not in {"baseline", "self-feedback", "safe"}:
        raise ValueError(f"Unsupported inference mode: {mode}")
    if mode != "baseline" and evaluator is None:
        raise ValueError(f"{mode} mode requires an evaluator backend")

    states = [ReasoningState(record=record) for record in records]
    max_steps = int(config["inference"]["max_steps"])
    max_retries = int(config["inference"]["max_retries"])
    generator_config = config["generation"]["generator"]
    evaluator_config = config["generation"]["evaluator"]

    while any(not state.finished for state in states):
        active = [state for state in states if not state.finished]
        messages_batch = [
            generator_messages(
                question=state.record["question"],
                passages=state.record["retrieved_passages"],
                steps=state.steps,
                feedback=state.pending_feedback,
                mode=mode,
            )
            for state in active
        ]
        generated = generator.generate(messages_batch, generator_config)

        for state, result in zip(active, generated):
            step_number = len(state.steps) + 1
            step = clean_generation(result.text, step_number)
            attempt: Dict[str, Any] = {
                "step_number": step_number,
                "retry": state.retries,
                "generated_text": step,
                "prompt_tokens": result.prompt_tokens,
                "generated_tokens": result.generated_tokens,
            }

            if mode == "baseline":
                state.steps.append(step)
                state.attempts.append({**attempt, "result": "accepted"})
                if extract_final_answer(step):
                    state.finished = True
                    state.status = "completed"
                elif len(state.steps) >= max_steps:
                    state.finished = True
                    state.status = "max_steps"
                continue

            assert evaluator is not None
            evaluation = _evaluate_with_repair(
                evaluator,
                evaluator_messages(
                    question=state.record["question"],
                    passages=state.record["retrieved_passages"],
                    previous_steps=state.steps,
                    current_step=step,
                ),
                evaluator_config,
            )
            accepted = "correct" in evaluation["error_type"].lower()
            state.attempts.append(
                {
                    **attempt,
                    "evaluation": evaluation,
                    "result": "accepted" if accepted else "rejected",
                }
            )

            if accepted:
                state.steps.append(step)
                state.feedback.append(evaluation)
                state.pending_feedback = evaluation
                state.retries = 0
                if extract_final_answer(step):
                    state.finished = True
                    state.status = "completed"
            else:
                state.retries += 1
                state.pending_feedback = evaluation
                if state.retries >= max_retries:
                    state.steps.append(step)
                    state.feedback.append(evaluation)
                    state.retries = 0

            if not state.finished and len(state.steps) >= max_steps:
                state.finished = True
                state.status = "max_steps"

    output = []
    for state in states:
        output.append(
            {
                **state.record,
                "mode": mode,
                "response": state.steps,
                "feedback": state.feedback,
                "attempts": state.attempts,
                "final_answer": extract_final_answer(state.steps),
                "status": state.status,
            }
        )
    return output


def run_inference(
    input_path: str,
    output_path: str,
    mode: str,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    records = normalize_many(read_records(input_path))
    generator = TransformersBackend.load(
        config["models"]["generator"],
        config,
    )
    evaluator: TextBackend | None = None
    if mode == "self-feedback":
        evaluator = generator
    elif mode == "safe":
        evaluator = TransformersBackend.load(
            config["models"]["evaluator_base"],
            config,
            adapter_path=config["models"]["evaluator_adapter"],
        )
    results = run_states(records, mode, generator, evaluator, config)
    write_jsonl(output_path, results)
    output_dir = Path(output_path).parent
    dump_config(config, output_dir / "resolved_config.yaml")
    write_json(
        output_dir / "run_manifest.json",
        {
            "mode": mode,
            "input": str(input_path),
            "output": str(output_path),
            "samples": len(results),
        },
    )
    return results

