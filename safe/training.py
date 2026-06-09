from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from .io import read_records, write_json
from .prompts import EVALUATOR_SYSTEM_PROMPT, evaluator_messages
from .schema import normalize_evaluator


def build_training_examples(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    examples = []
    for raw_record in records:
        record = normalize_evaluator(raw_record)
        messages = evaluator_messages(
            record["question"],
            record["retrieved_passages"],
            record["previous_steps"],
            record["current_step"],
        )
        examples.append(
            {
                "prompt": messages,
                "completion": [
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "error_type": record["error_type"],
                                "diagnosis": record["diagnosis"],
                                "guidance": record["guidance"],
                            },
                            ensure_ascii=False,
                        ),
                    }
                ],
            }
        )
    return examples


def train_evaluator(
    dataset_path: str,
    config: Dict[str, Any],
    output_dir: str | None = None,
) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    model_id = config["models"]["evaluator_base"]
    if not model_id:
        raise ValueError("models.evaluator_base is required for training")
    training = config["training"]
    target_dir = output_dir or training["output_dir"]
    examples = build_training_examples(read_records(dataset_path))
    dataset = Dataset.from_list(examples)
    split = dataset.train_test_split(
        test_size=float(training["test_size"]),
        seed=int(config["seed"]),
    )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dtype = getattr(torch, str(config["models"].get("dtype", "bfloat16")))
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization,
        device_map={"": local_rank} if torch.cuda.is_available() else None,
        trust_remote_code=bool(config["models"].get("trust_remote_code", True)),
        use_cache=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=bool(config["models"].get("trust_remote_code", True)),
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_config = LoraConfig(
        r=int(training["lora_r"]),
        lora_alpha=int(training["lora_alpha"]),
        lora_dropout=float(training["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    trainer_config = SFTConfig(
        output_dir=target_dir,
        num_train_epochs=float(training["num_train_epochs"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        warmup_ratio=float(training["warmup_ratio"]),
        logging_steps=int(training["logging_steps"]),
        eval_steps=int(training["eval_steps"]),
        save_steps=int(training["save_steps"]),
        eval_strategy="steps",
        save_strategy="steps",
        bf16=True,
        gradient_checkpointing=True,
        max_length=int(training["max_length"]),
        packing=False,
        report_to="none",
        seed=int(config["seed"]),
        ddp_find_unused_parameters=False,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        peft_config=peft_config,
        args=trainer_config,
    )
    trainer.train()
    trainer.save_model(target_dir)
    tokenizer.save_pretrained(target_dir)
    if local_rank == 0:
        write_json(
            Path(target_dir) / "safe_training_manifest.json",
            {
                "model_id": model_id,
                "dataset": dataset_path,
                "samples": len(examples),
                "system_prompt": EVALUATOR_SYSTEM_PROMPT,
                "training": training,
            },
        )

