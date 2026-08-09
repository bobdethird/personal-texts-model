"""Direct neutral-to-original Qwen LoRA supervised fine-tuning."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from imessage_mlx.stamp.neutralize import NEUTRAL_PAIR_FORMAT
from imessage_mlx.utils import (
    ensure_private_dir,
    ensure_private_file,
    read_jsonl,
    write_json,
)

REWRITE_SYSTEM_PROMPT = """\
Rewrite a neutral text-message draft in the phone owner's characteristic style.
Preserve meaning, facts, numbers, names, placeholders, and negation. Do not reply
to the message. Decide naturally whether the owner would send one bubble or a
short multi-bubble burst."""


@dataclass(frozen=True, slots=True)
class SFTTrainingConfig:
    train_path: str | Path
    validation_path: str | Path
    output_dir: str | Path
    model_name_or_path: str = "Qwen/Qwen3-4B"
    max_length: int = 1024
    epochs: float = 2.0
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 2
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str | Sequence[str] = "all-linear"
    dtype: str | None = "bfloat16"
    attn_implementation: str | None = "sdpa"
    resume: bool = True

    def __post_init__(self) -> None:
        if self.max_length < 8:
            raise ValueError("max_length must be at least 8")
        if self.epochs <= 0 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.lora_rank < 1 or self.lora_alpha < 1:
            raise ValueError("LoRA rank and alpha must be positive")


def _coerce_config(config: SFTTrainingConfig | Mapping[str, Any]) -> SFTTrainingConfig:
    if isinstance(config, SFTTrainingConfig):
        return config
    names = {field.name for field in fields(SFTTrainingConfig)}
    unknown = set(config) - names
    if unknown:
        raise ValueError(f"Unknown SFT configuration keys: {sorted(unknown)}")
    return SFTTrainingConfig(**dict(config))


def _bubbles(value: Any, *, field: str) -> list[str]:
    if isinstance(value, str):
        result = [value]
    elif isinstance(value, Sequence):
        result = [str(item) for item in value]
    else:
        raise ValueError(f"{field} must be a string or sequence")
    if not result or any(not bubble.strip() for bubble in result):
        raise ValueError(f"{field} must contain nonempty bubbles")
    return result


def format_rewrite_prompt(
    neutral_bubbles: str | Sequence[str],
    *,
    system_prompt: str = REWRITE_SYSTEM_PROMPT,
) -> str:
    """Create the stable standard-text prompt shared by SFT, CPO, and inference."""

    neutral = _bubbles(neutral_bubbles, field="neutral")
    payload = json.dumps(neutral, ensure_ascii=False)
    return (
        f"{system_prompt.strip()}\n\n"
        f"NEUTRAL_DRAFT:\n{payload}\n\n"
        'Return JSON only: {"original_bubbles": ["one or more styled bubbles"]}'
    )


def build_rewrite_messages(
    neutral_bubbles: str | Sequence[str],
    *,
    system_prompt: str = REWRITE_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """Return a Qwen-compatible chat prompt without loading a tokenizer."""

    neutral = _bubbles(neutral_bubbles, field="neutral")
    return [
        {"role": "system", "content": system_prompt.strip()},
        {
            "role": "user",
            "content": (
                f"NEUTRAL_DRAFT:\n{json.dumps(neutral, ensure_ascii=False)}\n\n"
                'Return JSON only: {"original_bubbles": ["one or more styled bubbles"]}'
            ),
        },
    ]


def render_qwen_prompt(
    tokenizer: Any,
    neutral_bubbles: str | Sequence[str],
    *,
    system_prompt: str = REWRITE_SYSTEM_PROMPT,
    add_generation_prompt: bool = True,
) -> str:
    """Render the prompt through the model tokenizer's native chat template."""

    if not hasattr(tokenizer, "apply_chat_template"):
        raise TypeError("Tokenizer must provide apply_chat_template")
    return str(
        tokenizer.apply_chat_template(
            build_rewrite_messages(neutral_bubbles, system_prompt=system_prompt),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    )


def parse_rewrite_output(
    output: str | Mapping[str, Any] | Sequence[str],
    *,
    expected_bubbles: int | None = None,
) -> list[str]:
    value: Any = output
    if isinstance(output, str):
        text = output.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1)
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError("Rewrite output must be valid JSON") from error
    if isinstance(value, Mapping):
        value = value.get("original_bubbles")
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError("Rewrite output must contain an original_bubbles array")
    bubbles = [str(item) for item in value]
    if expected_bubbles is not None and len(bubbles) != expected_bubbles:
        raise ValueError(f"Expected {expected_bubbles} bubbles, received {len(bubbles)}")
    if any(not bubble.strip() for bubble in bubbles):
        raise ValueError("Rewrite output contains an empty bubble")
    return bubbles


def build_sft_example(
    pair: Mapping[str, Any],
    *,
    system_prompt: str = REWRITE_SYSTEM_PROMPT,
) -> dict[str, Any]:
    """Convert one neutral pair to TRL conversational prompt-completion format."""

    if pair.get("format") != NEUTRAL_PAIR_FORMAT:
        raise ValueError(f"Unsupported neutral pair format {pair.get('format')!r}")
    neutral = _bubbles(pair.get("neutral"), field="neutral")
    original = _bubbles(pair.get("original"), field="original")
    return {
        "prompt": build_rewrite_messages(neutral, system_prompt=system_prompt),
        "completion": [
            {
                "role": "assistant",
                "content": json.dumps(
                    {"original_bubbles": original},
                    ensure_ascii=False,
                ),
            }
        ],
    }


def build_sft_examples(
    pairs: Sequence[Mapping[str, Any]],
    *,
    system_prompt: str = REWRITE_SYSTEM_PROMPT,
) -> list[dict[str, Any]]:
    return [build_sft_example(pair, system_prompt=system_prompt) for pair in pairs]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _privatize_tree(path: Path) -> None:
    for directory in [path, *(item for item in path.rglob("*") if item.is_dir())]:
        ensure_private_dir(directory)
    for file_path in (item for item in path.rglob("*") if item.is_file()):
        ensure_private_file(file_path)


def train_rewrite_sft(
    config: SFTTrainingConfig | Mapping[str, Any],
) -> dict[str, Any]:
    """Train a Qwen causal LM LoRA directly from neutral prompts to originals."""

    settings = _coerce_config(config)

    # Imported only for an actual training run.
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig, SFTTrainer

    train_pairs = list(read_jsonl(settings.train_path))
    validation_pairs = list(read_jsonl(settings.validation_path))
    if not train_pairs:
        raise ValueError("SFT training pairs must be nonempty")
    train_dataset = Dataset.from_list(build_sft_examples(train_pairs))
    validation_dataset = (
        Dataset.from_list(build_sft_examples(validation_pairs))
        if validation_pairs
        else None
    )
    tokenizer = AutoTokenizer.from_pretrained(settings.model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer must define an EOS or pad token")
        tokenizer.pad_token = tokenizer.eos_token

    output_dir = ensure_private_dir(settings.output_dir)
    has_validation = validation_dataset is not None
    model_init_kwargs = {
        key: value
        for key, value in {
            "dtype": settings.dtype,
            "attn_implementation": settings.attn_implementation,
        }.items()
        if value is not None
    }
    arguments = SFTConfig(
        output_dir=str(output_dir),
        max_length=settings.max_length,
        num_train_epochs=settings.epochs,
        per_device_train_batch_size=settings.batch_size,
        per_device_eval_batch_size=settings.batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        learning_rate=settings.learning_rate,
        warmup_ratio=settings.warmup_ratio,
        weight_decay=settings.weight_decay,
        logging_steps=settings.logging_steps,
        save_strategy="steps",
        save_steps=settings.save_steps,
        save_total_limit=settings.save_total_limit,
        eval_strategy="steps" if has_validation else "no",
        eval_steps=settings.save_steps if has_validation else None,
        report_to="none",
        seed=settings.seed,
        bf16=settings.bf16,
        gradient_checkpointing=settings.gradient_checkpointing,
        packing=False,
        model_init_kwargs=model_init_kwargs or None,
    )
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=settings.lora_rank,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        bias="none",
        target_modules=settings.lora_target_modules,
    )
    trainer = SFTTrainer(
        model=settings.model_name_or_path,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    resume_checkpoint = get_last_checkpoint(str(output_dir)) if settings.resume else None
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    final_dir = ensure_private_dir(output_dir / "final")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    summary = {
        "model_name_or_path": settings.model_name_or_path,
        "train_pairs": len(train_pairs),
        "validation_pairs": len(validation_pairs),
        "resumed_from": resume_checkpoint,
        "train_metrics": _jsonable(train_result.metrics),
        "final_dir": str(final_dir),
    }
    write_json(output_dir / "training-summary.json", summary)
    _privatize_tree(output_dir)
    return summary
