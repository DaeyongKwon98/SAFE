from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

from .config import apply_overrides, load_config
from .evaluation import evaluate_file
from .inference import run_inference
from .prepare import prepare_data
from .reproduce import run_reproduction
from .training import train_evaluator


def _config(args: argparse.Namespace) -> Dict[str, Any]:
    return apply_overrides(load_config(args.config), args.set or [])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m safe",
        description="SAFE research reproduction CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Build evaluator JSONL data.")
    prepare.add_argument("--config", default="configs/main.yaml")
    prepare.add_argument("--benchmark", required=True)
    prepare.add_argument("--annotations", nargs="*", default=[])
    prepare.add_argument("--errors", nargs="*", default=[])
    prepare.add_argument("--limit", type=int, default=0)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--set", action="append", default=[])

    train = subparsers.add_parser("train", help="Train the LoRA evaluator.")
    train.add_argument("--config", default="configs/main.yaml")
    train.add_argument("--dataset", required=True)
    train.add_argument("--output-dir")
    train.add_argument("--set", action="append", default=[])

    infer = subparsers.add_parser("infer", help="Run a reasoning mode.")
    infer.add_argument("--config", default="configs/main.yaml")
    infer.add_argument(
        "--mode",
        required=True,
        choices=("baseline", "self-feedback", "safe"),
    )
    infer.add_argument("--input", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--set", action="append", default=[])

    evaluate = subparsers.add_parser("evaluate", help="Compute EM/F1 and efficiency.")
    evaluate.add_argument("--input", required=True)
    evaluate.add_argument("--output", required=True)

    reproduce = subparsers.add_parser(
        "reproduce", help="Run or print the main reproduction workflow."
    )
    reproduce.add_argument("--config", default="configs/main.yaml")
    reproduce.add_argument("--dry-run", action="store_true")
    reproduce.add_argument("--set", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            records = prepare_data(
                args.benchmark,
                args.output,
                _config(args),
                annotation_paths=args.annotations,
                error_types=args.errors,
                limit=args.limit,
            )
            print(f"Prepared {len(records)} evaluator records: {args.output}")
        elif args.command == "train":
            train_evaluator(args.dataset, _config(args), args.output_dir)
        elif args.command == "infer":
            results = run_inference(
                args.input,
                args.output,
                args.mode,
                _config(args),
            )
            print(f"Wrote {len(results)} predictions: {args.output}")
        elif args.command == "evaluate":
            payload = evaluate_file(args.input, args.output)
            print(json.dumps(payload["summary"], indent=2))
        elif args.command == "reproduce":
            commands = run_reproduction(
                args.config,
                _config(args),
                dry_run=args.dry_run,
            )
            if args.dry_run:
                print("\n".join(commands))
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

