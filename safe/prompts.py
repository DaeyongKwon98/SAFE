from __future__ import annotations

from typing import Dict, Iterable, List

from .schema import ERROR_TYPES


ERROR_DEFINITIONS: Dict[str, str] = {
    "Correct (No Error)": "The step is grounded, logically sound, and makes necessary progress.",
    "Contradictory": "The step conflicts with an explicit statement in the retrieved passages.",
    "Unsupported": "The step states a fact that is absent from the retrieved passages.",
    "Logical Fallacy": "The evidence is valid, but the deduction or comparison is invalid.",
    "Information Miss": "The step says information is missing although it appears in the passages.",
    "Redundancy": "The step repeats an established fact or conclusion without progress.",
    "Overthinking": "The answer is already derivable, but the step continues unnecessary reasoning.",
    "Off-topic": "The step is unrelated to the information needed to answer the question.",
    "Inefficiency": "The step describes a plan instead of extracting evidence or reasoning.",
    "Premature Attribution": "The step uses an intermediate entity before its connection is established.",
    "Premature Conclusion": "The final answer is submitted before the reasoning supports it.",
    "Wrong Conclusion": "The submitted answer conflicts with the conclusion supported by prior steps.",
}

EVALUATOR_SYSTEM_PROMPT = """You are a precision evaluator for multi-hop question answering.
Use only the retrieved passages and previous reasoning steps. Classify the step using
the supplied error taxonomy. Return JSON only with the keys error_type, diagnosis,
and guidance. Guidance must describe one immediate next action. If the reasoning is
ready to finish, require this exact answer form: ####ANSWER: <answer_value>.

Error taxonomy:
{taxonomy}
""".strip().format(
    taxonomy="\n".join(
        f"- {name}: {ERROR_DEFINITIONS[name]}" for name in ERROR_TYPES
    )
)

GENERATOR_SYSTEM_PROMPT = """Generate exactly one next atomic reasoning step.
Use only the question, retrieved passages, previous steps, and evaluator feedback.

Valid forms:
- Step K: According to Passage N, <one explicit fact>. (Attribution)
- Step K: <one deduction from previous steps>. (Logical)
- Step K: ####ANSWER: <concise answer> (Final Answer)

Do not combine passage extraction and deduction. Follow evaluator guidance when
provided. The final answer step must contain no explanation.
""".strip()

BASELINE_SYSTEM_PROMPT = """Solve the question using only the retrieved passages.
Produce exactly one next atomic step on each call. Cite a single passage for an
attribution step, use only previous steps for a logical step, and finish with:
Step K: ####ANSWER: <concise answer> (Final Answer)
""".strip()

FORCE_ANSWER_SYSTEM_PROMPT = """Return only the final answer step:
Step K: ####ANSWER: <concise answer> (Final Answer)
Use the established reasoning and do not add explanation.
""".strip()


def format_passages(passages: Iterable[str]) -> str:
    return "\n".join(
        f"Passage {index}: {passage}"
        for index, passage in enumerate(passages, start=1)
    )


def format_steps(steps: Iterable[str]) -> str:
    values = [str(step).strip() for step in steps if str(step).strip()]
    return "\n".join(values) if values else "(No previous steps.)"


def generator_messages(
    question: str,
    passages: List[str],
    steps: List[str],
    feedback: Dict[str, str] | None,
    mode: str,
) -> List[Dict[str, str]]:
    step_number = len(steps) + 1
    feedback_text = "No evaluator feedback."
    if feedback:
        feedback_text = (
            f"Diagnosis: {feedback.get('diagnosis', '')}\n"
            f"Guidance: {feedback.get('guidance', '')}"
        )
    user = f"""Question:
{question}

Retrieved Passages:
{format_passages(passages)}

Previous Reasoning Steps:
{format_steps(steps)}

Feedback:
{feedback_text}

Generate Step {step_number}."""
    guidance = str((feedback or {}).get("guidance", "")).lower()
    force_answer = feedback is not None and any(
        marker in guidance
        for marker in ("####answer", "final answer", "answer the question", "no further steps")
    )
    if force_answer:
        system = FORCE_ANSWER_SYSTEM_PROMPT
        user += "\n\nReturn only the final answer step in the required format."
    else:
        system = BASELINE_SYSTEM_PROMPT if mode == "baseline" else GENERATOR_SYSTEM_PROMPT
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def evaluator_messages(
    question: str,
    passages: List[str],
    previous_steps: List[str],
    current_step: str,
) -> List[Dict[str, str]]:
    user = f"""Question:
{question}

Retrieved Passages:
{format_passages(passages)}

Previous Steps:
{format_steps(previous_steps)}

Step to evaluate:
{current_step}"""
    return [
        {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def injection_messages(
    record: Dict[str, object], error_type: str
) -> List[Dict[str, str]]:
    definition = ERROR_DEFINITIONS[error_type]
    user = f"""Question:
{record['question']}

Retrieved Passages:
{format_passages(record['retrieved_passages'])}

Previous Steps:
{format_steps(record.get('previous_steps', []))}

Correct Current Step:
{record['current_step']}

Create one replacement step exhibiting only this error: {error_type}.
Definition: {definition}

Return JSON only:
{{"current_step": "...", "diagnosis": "...", "guidance": "..."}}"""
    return [
        {
            "role": "system",
            "content": "You create controlled reasoning errors for evaluator training.",
        },
        {"role": "user", "content": user},
    ]

