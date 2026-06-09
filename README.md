# SAFE

SAFE is a Transformers-based research toolkit for feedback-guided multi-hop
reasoning. It compares three modes through one interface:

- **Baseline** generates steps without evaluation.
- **Self-feedback** uses the generator to evaluate and revise its own steps.
- **SAFE** uses a trained LoRA evaluator for diagnosis and next-step guidance.

The public repository contains the main comparison pipeline. Historical
ablation, API, retrieval, and notebook code is preserved in the
`pre-public-refactor` Git tag.

Project page: <https://daeyongkwon98.github.io/SAFE/>

## Installation

The reference environment is Python 3.10, CUDA 12.8, and four NVIDIA RTX A6000
48 GB GPUs.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The default configuration uses 4-bit NF4 loading and `device_map="auto"`.
Model IDs, GPU memory limits, batches, and generation limits are configured in
[`configs/main.yaml`](configs/main.yaml).

## Data contracts

Benchmark JSONL records require `id`, `dataset`, `question`,
`retrieved_passages`, and `answers`. Optional `ideal_steps` create supervised
correct-step examples. Evaluator JSONL additionally requires `previous_steps`,
`current_step`, `error_type`, `diagnosis`, and `guidance`.

Small synthetic fixtures are provided in [`examples/`](examples/).

## Reproduction

Build canonical evaluator data:

```bash
python -m safe prepare \
  --config configs/main.yaml \
  --benchmark examples/benchmark.jsonl \
  --annotations examples/evaluator.jsonl \
  --output outputs/data/evaluator.jsonl
```

Generate controlled error examples with the configured Transformers model:

```bash
python -m safe prepare \
  --config configs/main.yaml \
  --benchmark examples/benchmark.jsonl \
  --errors Contradictory Unsupported "Logical Fallacy" \
  --output outputs/data/evaluator_with_errors.jsonl
```

Train the 4-bit LoRA evaluator:

```bash
accelerate launch -m safe train \
  --config configs/main.yaml \
  --dataset outputs/data/evaluator.jsonl
```

Run each reasoning mode:

```bash
python -m safe infer --config configs/main.yaml --mode baseline \
  --input examples/benchmark.jsonl --output outputs/baseline/predictions.jsonl

python -m safe infer --config configs/main.yaml --mode self-feedback \
  --input examples/benchmark.jsonl --output outputs/self-feedback/predictions.jsonl

python -m safe infer --config configs/main.yaml --mode safe \
  --input examples/benchmark.jsonl --output outputs/safe/predictions.jsonl
```

Compute Exact Match, token F1, accepted steps, and retry counts:

```bash
python -m safe evaluate \
  --input outputs/safe/predictions.jsonl \
  --output outputs/safe/metrics.json
```

Inspect the workflow without running models:

```bash
python -m safe reproduce --config configs/main.yaml --dry-run
```

Override YAML values from the command line:

```bash
python -m safe infer ... \
  --set generation.batch_size=1 \
  --set models.max_memory.0=18GiB
```

## Tests

```bash
python -m unittest discover -s tests -v
python scripts/check_docs.py
```

The unit tests use mock components and do not download checkpoints.

## Repository policy

Datasets, model weights, checkpoints, and generated outputs are not tracked.
Only synthetic fixtures are included. Follow the licenses and terms of the
original datasets and model checkpoints.

See [`MIGRATION.md`](MIGRATION.md) for the historical-script mapping.

## License

Code is released under the [MIT License](LICENSE).
