import argparse
import gc
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from prompts import evaluate_system_prompt_premature_attribution
from prompts_ablation import (
    evaluate_system_prompt_drop_contradictory,
    evaluate_system_prompt_drop_contradictory_information_miss_unsupported_premature_attribution,
    evaluate_system_prompt_drop_inefficiency,
    evaluate_system_prompt_drop_information_miss,
    evaluate_system_prompt_drop_logical_fallacy,
    evaluate_system_prompt_drop_off_topic,
    evaluate_system_prompt_drop_off_topic_inefficiency_redundancy_overthinking,
    evaluate_system_prompt_drop_overthinking,
    evaluate_system_prompt_drop_premature_attribution,
    evaluate_system_prompt_drop_redundancy,
    evaluate_system_prompt_drop_unsupported,
    evaluate_system_prompt_drop_wrong_conclusion,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

### 학습 명령어 예시 ###
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 finetune_validationset_fast_ablation.py
########################


LEGACY_PROMPT_KEY = "legacy_default"
DROP_EVALUATOR_PROMPT_MAP: Dict[str, str] = {
    "drop_wrong_conclusion": evaluate_system_prompt_drop_wrong_conclusion,
    "drop_overthinking": evaluate_system_prompt_drop_overthinking,
    "drop_off_topic": evaluate_system_prompt_drop_off_topic,
    "drop_redundancy": evaluate_system_prompt_drop_redundancy,
    "drop_inefficiency": evaluate_system_prompt_drop_inefficiency,
    "drop_contradictory": evaluate_system_prompt_drop_contradictory,
    "drop_unsupported": evaluate_system_prompt_drop_unsupported,
    "drop_information_miss": evaluate_system_prompt_drop_information_miss,
    "drop_premature_attribution": evaluate_system_prompt_drop_premature_attribution,
    "drop_logical_fallacy": evaluate_system_prompt_drop_logical_fallacy,
    "drop_contradictory_information_miss_unsupported_premature_attribution": (
        evaluate_system_prompt_drop_contradictory_information_miss_unsupported_premature_attribution
    ),
    "drop_off_topic_inefficiency_redundancy_overthinking": (
        evaluate_system_prompt_drop_off_topic_inefficiency_redundancy_overthinking
    ),
}

EXCLUDED_DROP_ERROR_TYPES = {"Correct (No Error)"}
FIXED_GROUP4_DROP_TARGETS: Dict[str, List[str]] = {
    "drop_contradictory_information_miss_unsupported_premature_attribution": [
        "Contradictory",
        "Information Miss",
        "Unsupported",
        "Premature Attribution",
    ],
    "drop_off_topic_inefficiency_redundancy_overthinking": [
        "Off-topic",
        "Inefficiency",
        "Redundancy",
        "Overthinking",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-8B 4bit LoRA ablation trainer (leave_one_out or fixed_group4)."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="/workspace/hf_transformers/Qwen3-8B",
        help="Base model path or model id.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/workspace/daeyong/fourth_finetuning_data/2wiki_added_ver3.jsonl",
        help="Training dataset path (.json or .jsonl).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./trained_models/qwen3-8b-2wiki_added_ver3_errortype_ablation",
        help=(
            "Root directory where drop_* run outputs are saved. "
            "For fixed_group4, default is auto-switched to "
            "./trained_models/qwen3-8b-2wiki_added_ver3_errortype_group4_ablation"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=8500)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument(
        "--disable_gradient_checkpointing",
        action="store_true",
        help="Disable gradient checkpointing. Default is enabled.",
    )
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--ablation_target_mode",
        type=str,
        choices=["leave_one_out", "fixed_group4"],
        default="leave_one_out",
        help=(
            "How to build ablation targets. "
            "leave_one_out=drop one error type per run; "
            "fixed_group4=run two fixed 4-error-group drops."
        ),
    )
    parser.add_argument(
        "--evaluator_prompt_mode",
        type=str,
        choices=["ablation", "legacy"],
        default="ablation",
        help=(
            "Prompt selection mode for evaluator training target. "
            "ablation=use drop-specific prompt from prompts_ablation.py; "
            "legacy=always use evaluate_system_prompt_premature_attribution."
        ),
    )
    parser.add_argument(
        "--missing_ablation_prompt_policy",
        type=str,
        choices=["error", "fallback_legacy"],
        default="error",
        help=(
            "When evaluator_prompt_mode=ablation and drop prompt mapping is missing: "
            "error=raise and stop, fallback_legacy=use legacy prompt."
        ),
    )
    return parser.parse_args()


def is_main_process(local_rank: int) -> bool:
    return local_rank == 0


def maybe_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify_error_type(error_type: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", error_type.lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    return slug if slug else "unknown_error_type"


def get_drop_key_from_error_type(error_type: str) -> str:
    return f"drop_{slugify_error_type(error_type)}"


def select_evaluator_system_prompt(
    *,
    drop_key: str,
    evaluator_prompt_mode: str,
    missing_ablation_prompt_policy: str,
) -> Tuple[str, str, str]:
    if evaluator_prompt_mode == "legacy":
        return (
            evaluate_system_prompt_premature_attribution,
            LEGACY_PROMPT_KEY,
            "prompts",
        )

    prompt = DROP_EVALUATOR_PROMPT_MAP.get(drop_key)
    if prompt is not None:
        return prompt, drop_key, "prompts_ablation"

    if missing_ablation_prompt_policy == "fallback_legacy":
        return (
            evaluate_system_prompt_premature_attribution,
            LEGACY_PROMPT_KEY,
            "prompts",
        )

    raise KeyError(
        "Missing ablation prompt mapping for "
        f"drop_key='{drop_key}'. "
        "Either add mapping in DROP_EVALUATOR_PROMPT_MAP or use "
        "--missing_ablation_prompt_policy fallback_legacy."
    )

def load_raw_items(dataset_path: str) -> List[Dict]:
    if dataset_path.endswith(".jsonl"):
        items: List[Dict] = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items

    if dataset_path.endswith(".json"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, list):
            return loaded
        if isinstance(loaded, dict) and isinstance(loaded.get("data"), list):
            return loaded["data"]
        if isinstance(loaded, dict):
            return [loaded]
        raise ValueError("Unsupported JSON structure. Expected list/dict/data-list format.")

    raise ValueError(f"Unsupported dataset extension: {dataset_path}")


def parse_training_records(
    dataset_path: str,
) -> Tuple[List[Dict[str, str]], Dict[str, int], Dict[str, int]]:
    raw_items = load_raw_items(dataset_path)
    error_counter: Counter = Counter()

    stats = {
        "raw_total": len(raw_items),
        "used_total": 0,
        "dropped_non_dict": 0,
        "dropped_missing_input_or_output": 0,
        "dropped_invalid_output_json": 0,
        "dropped_missing_error_type": 0,
    }

    records: List[Dict[str, str]] = []

    for item in raw_items:
        if not isinstance(item, dict):
            stats["dropped_non_dict"] += 1
            continue

        input_text = item.get("input")
        output_text = item.get("output")

        if input_text is None or output_text is None:
            stats["dropped_missing_input_or_output"] += 1
            continue

        if not isinstance(input_text, str):
            input_text = str(input_text)

        if isinstance(output_text, dict):
            output_str = json.dumps(output_text, ensure_ascii=False)
        elif isinstance(output_text, str):
            output_str = output_text
        else:
            output_str = str(output_text)

        error_type = item.get("error_type")

        if not error_type:
            if isinstance(output_text, dict):
                error_type = output_text.get("error_type")
            else:
                try:
                    parsed_output = json.loads(output_str)
                    if isinstance(parsed_output, dict):
                        error_type = parsed_output.get("error_type")
                except Exception:
                    stats["dropped_invalid_output_json"] += 1

        if not error_type or not str(error_type).strip():
            stats["dropped_missing_error_type"] += 1
            continue

        error_type = str(error_type).strip()
        error_counter[error_type] += 1

        records.append(
            {
                "input": input_text,
                "output": output_str,
                "error_type": error_type,
            }
        )
        stats["used_total"] += 1

    return records, dict(error_counter), stats


def build_chat_dataset(records: List[Dict[str, str]], system_prompt: str) -> Dataset:
    data = []
    for item in records:
        data.append(
            {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": item["input"]},
                ],
                "completion": [
                    {"role": "assistant", "content": item["output"]},
                ],
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
    return Dataset.from_list(data)


def load_model_and_tokenizer(model_id: str, local_rank: int):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    device_map = {"": local_rank}
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        use_cache=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def make_lora_config() -> LoraConfig:
    return LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=64,
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


def make_sft_config(args: argparse.Namespace, output_dir: str) -> SFTConfig:
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=(not args.disable_gradient_checkpointing),
        optim=args.optim,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=False,
        bf16=True,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=args.lr_scheduler_type,
        report_to=args.report_to,
        ddp_find_unused_parameters=False,
        max_length=args.max_length,
        packing=False,
        seed=args.seed,
        overwrite_output_dir=args.overwrite_output_dir,
    )


def save_json(path: str, payload: Dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        fallback_payload = {
            "serialization_error": str(e),
            "payload_repr": str(payload),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(fallback_payload, f, indent=2, ensure_ascii=False, default=str)


def has_trained_adapter(output_dir: str) -> bool:
    if not os.path.isdir(output_dir):
        return False

    adapter_config = os.path.join(output_dir, "adapter_config.json")
    adapter_safetensors = os.path.join(output_dir, "adapter_model.safetensors")
    adapter_bin = os.path.join(output_dir, "adapter_model.bin")

    return os.path.exists(adapter_config) and (
        os.path.exists(adapter_safetensors) or os.path.exists(adapter_bin)
    )



def build_drop_targets(
    *,
    discovered_error_types: List[str],
    ablation_target_mode: str,
) -> List[Dict[str, List[str]]]:
    if ablation_target_mode == "leave_one_out":
        return [
            {
                "drop_key": get_drop_key_from_error_type(error_type),
                "dropped_error_types": [error_type],
            }
            for error_type in discovered_error_types
            if error_type not in EXCLUDED_DROP_ERROR_TYPES
        ]

    if ablation_target_mode == "fixed_group4":
        return [
            {
                "drop_key": drop_key,
                "dropped_error_types": list(error_types),
            }
            for drop_key, error_types in FIXED_GROUP4_DROP_TARGETS.items()
        ]

    raise ValueError(f"Unsupported ablation_target_mode: {ablation_target_mode}")

def main() -> None:
    args = parse_args()

    default_leave_one_out_root = "./trained_models/qwen3-8b-2wiki_added_ver3_errortype_ablation"
    default_fixed_group4_root = "./trained_models/qwen3-8b-2wiki_added_ver3_errortype_group4_ablation"
    if args.ablation_target_mode == "fixed_group4" and args.output_root == default_leave_one_out_root:
        args.output_root = default_fixed_group4_root

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")
    torch.cuda.set_device(local_rank)

    if is_main_process(local_rank):
        os.makedirs(args.output_root, exist_ok=True)

    maybe_barrier()

    records, error_distribution, parse_stats = parse_training_records(args.dataset_path)
    discovered_error_types = sorted(error_distribution.keys())
    drop_targets = build_drop_targets(
        discovered_error_types=discovered_error_types,
        ablation_target_mode=args.ablation_target_mode,
    )

    if is_main_process(local_rank):
        print(f"📂 Dataset: {args.dataset_path}")
        print(f"✅ Parsed samples: {parse_stats['used_total']} / {parse_stats['raw_total']}")
        print(
            "ℹ️ Drop stats: "
            f"non_dict={parse_stats['dropped_non_dict']}, "
            f"missing_input_or_output={parse_stats['dropped_missing_input_or_output']}, "
            f"invalid_output_json={parse_stats['dropped_invalid_output_json']}, "
            f"missing_error_type={parse_stats['dropped_missing_error_type']}"
        )
        print(f"🔎 Discovered error types ({len(discovered_error_types)}): {discovered_error_types}")
        print(
            "🧠 Evaluator prompt config: "
            f"mode={args.evaluator_prompt_mode}, "
            f"missing_policy={args.missing_ablation_prompt_policy}, "
            f"ablation_target_mode={args.ablation_target_mode}"
        )

        if args.ablation_target_mode == "leave_one_out":
            print(
                f"🚫 Excluded from leave-one-out targets: {sorted(EXCLUDED_DROP_ERROR_TYPES)} | "
                f"actual runs={len(drop_targets)}"
            )
        else:
            print(f"🧪 Fixed group4 targets: {len(drop_targets)}")
            for target in drop_targets:
                print(f"  - {target['drop_key']}: {target['dropped_error_types']}")

        for error_type in discovered_error_types:
            print(f"  - {error_type}: {error_distribution[error_type]}")

    if not records:
        raise RuntimeError("No valid training records found after parsing/filtering.")

    if not discovered_error_types:
        raise RuntimeError("No error types found. Cannot run ablation.")

    if not drop_targets:
        raise RuntimeError(
            "No drop targets generated. "
            "Please check error_type distribution and ablation_target_mode."
        )

    run_summaries: List[Dict] = []
    lora_config = make_lora_config()

    for drop_target in drop_targets:
        drop_key = drop_target["drop_key"]
        dropped_error_types = list(drop_target["dropped_error_types"])
        dropped_error_types_set = set(dropped_error_types)
        primary_drop_type = dropped_error_types[0] if dropped_error_types else "unknown_error_type"
        primary_drop_slug = slugify_error_type(primary_drop_type)

        output_dir = os.path.join(args.output_root, drop_key)
        filtered_records = [
            r for r in records if r["error_type"] not in dropped_error_types_set
        ]

        selected_system_prompt: Optional[str] = None
        selected_prompt_key: Optional[str] = None
        selected_prompt_source: Optional[str] = None

        try:
            selected_system_prompt, selected_prompt_key, selected_prompt_source = select_evaluator_system_prompt(
                drop_key=drop_key,
                evaluator_prompt_mode=args.evaluator_prompt_mode,
                missing_ablation_prompt_policy=args.missing_ablation_prompt_policy,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to select evaluator system prompt for drop_key='{drop_key}': {e}"
            )

        if is_main_process(local_rank):
            print("\n" + "=" * 100)
            print(f"🚀 Ablation run start: drop_key='{drop_key}'")
            print(f"   Dropped error types: {dropped_error_types}")
            print(f"   Original samples: {len(records)}")
            print(f"   Filtered samples: {len(filtered_records)}")
            print(f"   Output dir: {output_dir}")
            print(
                "   Selected prompt: "
                f"key={selected_prompt_key}, source={selected_prompt_source}"
            )

        run_common_meta = {
            "drop_key": drop_key,
            "dropped_error_types": dropped_error_types,
            "dropped_error_type": primary_drop_type if len(dropped_error_types) == 1 else None,
            "dropped_error_type_slug": primary_drop_slug if len(dropped_error_types) == 1 else None,
            "evaluator_prompt_mode": args.evaluator_prompt_mode,
            "missing_ablation_prompt_policy": args.missing_ablation_prompt_policy,
            "selected_prompt_key": selected_prompt_key,
            "selected_prompt_source": selected_prompt_source,
            "output_dir": output_dir,
        }

        if len(filtered_records) == 0:
            if is_main_process(local_rank):
                run_summaries.append(
                    {
                        **run_common_meta,
                        "status": "skipped_empty_subset",
                        "original_samples": len(records),
                        "filtered_samples": 0,
                    }
                )
            continue

        if has_trained_adapter(output_dir) and (not args.overwrite_output_dir):
            if is_main_process(local_rank):
                print(
                    "   ⏭️ Existing trained adapter detected. Skipping this run: "
                    f"{output_dir}"
                )
                run_summaries.append(
                    {
                        **run_common_meta,
                        "status": "skipped_existing_trained_model",
                        "original_samples": len(records),
                        "filtered_samples": len(filtered_records),
                    }
                )
            continue

        if os.path.isdir(output_dir) and os.listdir(output_dir) and (not args.overwrite_output_dir):
            raise RuntimeError(
                f"Output directory already exists and is not empty: {output_dir}. "
                "Use --overwrite_output_dir to allow reuse."
            )

        maybe_barrier()

        chat_dataset = build_chat_dataset(filtered_records, system_prompt=selected_system_prompt)
        if len(chat_dataset) < 2:
            if is_main_process(local_rank):
                run_summaries.append(
                    {
                        **run_common_meta,
                        "status": "skipped_not_enough_samples_for_eval",
                        "original_samples": len(records),
                        "filtered_samples": len(filtered_records),
                    }
                )
            continue

        dataset_split = chat_dataset.train_test_split(test_size=0.1, seed=args.seed)
        train_dataset = dataset_split["train"]
        eval_dataset = dataset_split["test"]

        if len(eval_dataset) == 0:
            if is_main_process(local_rank):
                run_summaries.append(
                    {
                        **run_common_meta,
                        "status": "skipped_empty_eval_split",
                        "original_samples": len(records),
                        "filtered_samples": len(filtered_records),
                    }
                )
            continue

        if is_main_process(local_rank):
            print(f"   Train/Eval split: {len(train_dataset)} / {len(eval_dataset)} (90:10)")

        model, tokenizer = load_model_and_tokenizer(args.model_id, local_rank)
        sft_config = make_sft_config(args, output_dir)

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=lora_config,
            args=sft_config,
        )

        start_time = utc_now_iso()
        trainer.train()
        end_time = utc_now_iso()

        maybe_barrier()

        if is_main_process(local_rank):
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)

            hp_payload = {
                "meta_info": {
                    "model_id": args.model_id,
                    "dataset_path": args.dataset_path,
                    "output_root": args.output_root,
                    "output_dir": output_dir,
                    "drop_key": drop_key,
                    "dropped_error_types": dropped_error_types,
                    "dropped_error_type": primary_drop_type if len(dropped_error_types) == 1 else None,
                    "dropped_error_type_slug": primary_drop_slug if len(dropped_error_types) == 1 else None,
                    "evaluator_prompt_mode": args.evaluator_prompt_mode,
                    "missing_ablation_prompt_policy": args.missing_ablation_prompt_policy,
                    "selected_prompt_key": selected_prompt_key,
                    "selected_prompt_source": selected_prompt_source,
                    "raw_total": parse_stats["raw_total"],
                    "used_total": len(records),
                    "filtered_samples": len(filtered_records),
                    "train_samples": len(train_dataset),
                    "eval_samples": len(eval_dataset),
                    "run_started_at_utc": start_time,
                    "run_finished_at_utc": end_time,
                },
                "dataset_parse_stats": parse_stats,
                "original_error_distribution": error_distribution,
                "lora_config": lora_config.to_dict(),
                "training_args": sft_config.to_dict(),
            }
            save_json(os.path.join(output_dir, "hyperparameters.json"), hp_payload)
            print(f"✅ Saved run outputs to: {output_dir}")

            run_summaries.append(
                {
                    **run_common_meta,
                    "status": "completed",
                    "original_samples": len(records),
                    "filtered_samples": len(filtered_records),
                    "train_samples": len(train_dataset),
                    "eval_samples": len(eval_dataset),
                    "run_started_at_utc": start_time,
                    "run_finished_at_utc": end_time,
                }
            )

        del trainer
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        maybe_barrier()

    if is_main_process(local_rank):
        ablation_mode_name = (
            "leave_one_error_type_out"
            if args.ablation_target_mode == "leave_one_out"
            else "fixed_group4_error_type_out"
        )

        manifest = {
            "meta_info": {
                "created_at_utc": utc_now_iso(),
                "model_id": args.model_id,
                "dataset_path": args.dataset_path,
                "output_root": args.output_root,
                "ablation_mode": ablation_mode_name,
                "ablation_target_mode": args.ablation_target_mode,
                "train_mode": "train_with_eval_split_90_10",
                "evaluator_prompt_mode": args.evaluator_prompt_mode,
                "missing_ablation_prompt_policy": args.missing_ablation_prompt_policy,
            },
            "dataset_parse_stats": parse_stats,
            "discovered_error_types": discovered_error_types,
            "excluded_drop_error_types": sorted(EXCLUDED_DROP_ERROR_TYPES),
            "fixed_group4_drop_targets": FIXED_GROUP4_DROP_TARGETS,
            "drop_targets": drop_targets,
            "drop_error_types": [
                target["dropped_error_types"][0]
                for target in drop_targets
                if len(target["dropped_error_types"]) == 1
            ],
            "original_error_distribution": error_distribution,
            "run_count_planned": len(drop_targets),
            "run_count_executed": len(run_summaries),
            "runs": run_summaries,
        }
        manifest_path = os.path.join(args.output_root, "ablation_manifest.json")
        save_json(manifest_path, manifest)
        print("\n" + "=" * 100)
        print(f"🏁 All ablation runs finished. Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
