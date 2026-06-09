from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "models": {
        "generator": "",
        "evaluator_base": "",
        "evaluator_adapter": "",
        "load_in_4bit": True,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "enable_thinking": False,
        "max_memory": {},
    },
    "generation": {
        "batch_size": 4,
        "max_input_tokens": 8192,
        "generator": {
            "max_new_tokens": 256,
            "temperature": 0.0,
            "top_p": 1.0,
        },
        "evaluator": {
            "max_new_tokens": 384,
            "temperature": 0.0,
            "top_p": 1.0,
        },
    },
    "inference": {
        "max_steps": 10,
        "max_retries": 3,
    },
    "training": {
        "output_dir": "outputs/evaluator",
        "test_size": 0.1,
        "max_length": 8500,
        "num_train_epochs": 2.0,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 0.0001,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "logging_steps": 10,
        "eval_steps": 100,
        "save_steps": 100,
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
    },
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path | None) -> Dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is None:
        return config
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return deep_merge(config, loaded)


def set_nested(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = cursor.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot override non-mapping key: {dotted_key}")
        cursor = child
    cursor[parts[-1]] = value


def apply_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    result = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must use key=value syntax: {item}")
        key, raw_value = item.split("=", 1)
        set_nested(result, key.strip(), yaml.safe_load(raw_value))
    return result


def dump_config(config: Dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

