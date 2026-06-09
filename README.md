# SAFE: An LLM-as-Verifier Framework for Evidence-Grounded Multi-Hop Reasoning

This repository contains the official implementation of SAFE, an LLM-as-verifier framework that tackles "spurious correctness" in multi-hop QA by shifting evaluation from post-hoc answer judgment to stepwise reasoning verification.

SAFE decomposes reasoning into atomic, Knowledge Graph (KG)-grounded units to check intermediate steps during generation. By identifying invalid reasoning and providing immediate correction feedback before errors propagate, SAFE improves accuracy by 8.8 pp on average across three multi-hop QA benchmarks.

The repository provides three inference modes through one interface:

- **Baseline** generates steps without evaluation.
- **Self-feedback** uses the generator to evaluate and revise its own steps.
- **SAFE** uses a trained LoRA evaluator for diagnosis and next-step guidance.

## Installation

The reference environment is Python 3.10 and CUDA 12.8.

```bash
conda create -n safe python=3.10 -y
conda activate safe
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

## Benchmarks

The experiments use three public multi-hop question answering benchmarks. Download
them from their original project pages:

- [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop)
- [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa)
- [MuSiQue](https://github.com/stonybrooknlp/musique)

The repository does not redistribute the complete benchmark files. Convert the
downloaded data to the benchmark JSONL contract described above.

### Wrong-question splits

The [`data/`](data/) directory provides the question-only subsets whose
knowledge-graph evidence was marked insufficient. Each text file contains exactly one question per line.

| Dataset | Train | Validation |
| --- | ---: | ---: |
| 2WikiMultiHopQA | [`2wiki_train_wrong.txt`](data/2wiki_train_wrong.txt) (7,452) | [`2wiki_val_wrong.txt`](data/2wiki_val_wrong.txt) (901) |
| HotpotQA | [`hotpotqa_train_wrong.txt`](data/hotpotqa_train_wrong.txt) (2,376) | [`hotpotqa_val_wrong.txt`](data/hotpotqa_val_wrong.txt) (227) |
| MuSiQue | [`musique_train_wrong.txt`](data/musique_train_wrong.txt) (2,149) | [`musique_val_wrong.txt`](data/musique_val_wrong.txt) (342) |

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

## Citation

If you find our work useful, please cite our work!

```bibtex
@article{kwon2026safe,
  title={SAFE: Stepwise Atomic Feedback for Error correction in Multi-hop Reasoning},
  author={Kwon, Daeyong and Yoon, Soyoung and Hwang, Seung-won},
  journal={arXiv preprint arXiv:2604.01993},
  year={2026}
}
```

## License

Code is released under the [MIT License](LICENSE).
