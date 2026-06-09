from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def reproduction_commands(
    config_path: str,
    config: Dict[str, Any],
) -> List[List[str]]:
    paths = config.get("paths", {})
    required = ("benchmark", "evaluator_data", "inference_input", "output_root")
    missing = [key for key in required if not paths.get(key)]
    if missing:
        raise ValueError(f"Config paths section is missing: {missing}")
    root = Path(paths["output_root"])
    commands = [
        [
            sys.executable,
            "-m",
            "safe",
            "prepare",
            "--config",
            config_path,
            "--benchmark",
            paths["benchmark"],
            "--output",
            paths["evaluator_data"],
        ],
        [
            sys.executable,
            "-m",
            "safe",
            "train",
            "--config",
            config_path,
            "--dataset",
            paths["evaluator_data"],
        ],
    ]
    for mode in ("baseline", "self-feedback", "safe"):
        inference_output = root / mode / "predictions.jsonl"
        commands.append(
            [
                sys.executable,
                "-m",
                "safe",
                "infer",
                "--config",
                config_path,
                "--mode",
                mode,
                "--input",
                paths["inference_input"],
                "--output",
                str(inference_output),
            ]
        )
        commands.append(
            [
                sys.executable,
                "-m",
                "safe",
                "evaluate",
                "--input",
                str(inference_output),
                "--output",
                str(root / mode / "metrics.json"),
            ]
        )
    return commands


def run_reproduction(
    config_path: str,
    config: Dict[str, Any],
    dry_run: bool,
) -> List[str]:
    commands = reproduction_commands(config_path, config)
    rendered = [shlex.join(command) for command in commands]
    if not dry_run:
        for command in commands:
            subprocess.run(command, check=True)
    return rendered

