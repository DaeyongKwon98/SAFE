# Migration from the experiment snapshot

The public repository replaces the flat experiment snapshot with a single
Transformers implementation.

| Historical entry points | Public command |
| --- | --- |
| `generate_*`, `inject_error_*`, training notebooks | `python -m safe prepare` |
| `finetune_validationset_fast.py` | `python -m safe train` |
| `inference_baseline_vllm.py` | `python -m safe infer --mode baseline` |
| `inference_self_feedback_vllm.py` | `python -m safe infer --mode self-feedback` |
| `inference_vllm.py` | `python -m safe infer --mode safe` |
| final-answer, judge, and EM/F1 scripts | `python -m safe evaluate` |
| Root shell pipelines | `python -m safe reproduce` |

Compatibility changes:

- vLLM, OpenAI, and Gemini backends are removed.
- Inputs and outputs use canonical JSONL schemas.
- Personal absolute paths are replaced by YAML and CLI values.
- Model IDs use Hugging Face identifiers or explicit local paths.
- Runs write predictions, a resolved config, and a manifest.
- Historical files remain available with
  `git switch --detach pre-public-refactor`.

