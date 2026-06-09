from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any, Dict, List, Protocol, Sequence


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    generated_tokens: int


class TextBackend(Protocol):
    def generate(
        self,
        messages: Sequence[Sequence[Dict[str, str]]],
        generation: Dict[str, Any],
    ) -> List[GenerationResult]:
        ...


def _normalize_max_memory(raw: Dict[Any, Any]) -> Dict[Any, str] | None:
    if not raw:
        return None
    normalized: Dict[Any, str] = {}
    for key, value in raw.items():
        normalized[int(key) if str(key).isdigit() else key] = str(value)
    return normalized


class TransformersBackend:
    """Batched text generation using a single Transformers model."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        batch_size: int = 4,
        max_input_tokens: int = 8192,
        enable_thinking: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_input_tokens = max_input_tokens
        self.enable_thinking = enable_thinking
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @classmethod
    def load(
        cls,
        model_id: str,
        config: Dict[str, Any],
        adapter_path: str | None = None,
    ) -> "TransformersBackend":
        if not model_id:
            raise ValueError("A model id or local model path is required")

        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        models = config["models"]
        dtype_name = str(models.get("dtype", "bfloat16"))
        dtype = getattr(torch, dtype_name)
        load_in_4bit = bool(models.get("load_in_4bit", True))
        model_kwargs: Dict[str, Any] = {
            "device_map": "auto",
            "dtype": dtype,
            "trust_remote_code": bool(models.get("trust_remote_code", True)),
            "max_memory": _normalize_max_memory(models.get("max_memory", {})),
        }
        if model_kwargs["max_memory"] is None:
            model_kwargs.pop("max_memory")
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=False,
            )

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=model_kwargs["trust_remote_code"],
        )
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        model.eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            batch_size=int(config["generation"]["batch_size"]),
            max_input_tokens=int(config["generation"]["max_input_tokens"]),
            enable_thinking=bool(models.get("enable_thinking", False)),
        )

    def _render(self, messages: Sequence[Dict[str, str]]) -> str:
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            return self.tokenizer.apply_chat_template(
                list(messages),
                enable_thinking=self.enable_thinking,
                **kwargs,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(list(messages), **kwargs)

    def _input_device(self) -> Any:
        try:
            return self.model.get_input_embeddings().weight.device
        except (AttributeError, TypeError):
            return self.model.device

    def generate(
        self,
        messages: Sequence[Sequence[Dict[str, str]]],
        generation: Dict[str, Any],
    ) -> List[GenerationResult]:
        import torch
        rendered = [self._render(item) for item in messages]
        order = sorted(range(len(rendered)), key=lambda index: len(rendered[index]))
        results: List[GenerationResult | None] = [None] * len(rendered)

        temperature = float(generation.get("temperature", 0.0))
        generation_config = copy.deepcopy(self.model.generation_config)
        generation_config.max_new_tokens = int(
            generation.get("max_new_tokens", 256)
        )
        generation_config.do_sample = temperature > 0
        generation_config.temperature = temperature if temperature > 0 else None
        generation_config.top_p = (
            float(generation.get("top_p", 1.0)) if temperature > 0 else None
        )
        generation_config.top_k = None
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        generation_config.eos_token_id = self.tokenizer.eos_token_id

        for start in range(0, len(order), self.batch_size):
            indices = order[start : start + self.batch_size]
            batch_text = [rendered[index] for index in indices]
            encoded = self.tokenizer(
                batch_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
            )
            encoded = {
                key: value.to(self._input_device())
                for key, value in encoded.items()
            }
            input_width = encoded["input_ids"].shape[1]
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **encoded,
                    generation_config=generation_config,
                    use_model_defaults=False,
                )
            new_ids = output_ids[:, input_width:]
            decoded = self.tokenizer.batch_decode(
                new_ids,
                skip_special_tokens=True,
            )
            prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
            generated_lengths = [
                int((row != self.tokenizer.pad_token_id).sum().item())
                for row in new_ids
            ]
            for index, text, prompt_len, generated_len in zip(
                indices, decoded, prompt_lengths, generated_lengths
            ):
                results[index] = GenerationResult(
                    text=text.strip(),
                    prompt_tokens=int(prompt_len),
                    generated_tokens=generated_len,
                )

        return [result for result in results if result is not None]

