# SAFE

Code snapshot for SAFE experiments.

## Contents

- Python scripts (`.py`)
- Jupyter notebooks (`.ipynb`)
- Shell scripts (`.sh`)

The original directory structure is preserved. Notebook outputs and execution
counts have been cleared before publication.

## Exclusions

Datasets, generated results, logs, images, model weights, environment
directories, and other large artifacts are intentionally omitted. The
following source directories are also excluded:

`audio`, `docker_build`, `conda_envs`, `chatbot`, `gaia`, `llm`, `MusTBench`,
`QASC`, `reproduce`, and `StrategyQA`.

Some scripts reference local datasets, models, or paths that are not included
in this repository.

## Credentials

Scripts that call the OpenAI API read credentials from the environment:

```bash
export OPENAI_API_KEY="..."
```
