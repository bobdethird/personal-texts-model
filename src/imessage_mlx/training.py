from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from imessage_mlx.data.sft import SFT_FORMAT

SPECIAL_TOKENS = (
    "<|conversation|>",
    "<|user|>",
    "<|assistant|>",
    "<|turn_end|>",
)


def configure_tokenizer(tokenizer: Any) -> int:
    """Register the stable transcript delimiters and return the added-token count."""
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": list(SPECIAL_TOKENS)}
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer must provide either a pad token or an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    return int(added)


def render_example_parts(record: dict[str, Any]) -> tuple[str, str]:
    """Render history/prompt separately from the one supervised target."""
    if record.get("format") != SFT_FORMAT:
        raise ValueError(f"Unsupported SFT format {record.get('format')!r}")
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("SFT example must contain at least one message")
    target_index = int(record.get("target_index", -1))
    if target_index != len(messages) - 1:
        raise ValueError("The target must be the final message")
    if messages[target_index].get("role") != "assistant":
        raise ValueError("The target must have the assistant role")

    prompt = ["<|conversation|>\n"]
    for index, message in enumerate(messages):
        role = message.get("role")
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported conversation role {role!r}")
        marker = "<|user|>" if role == "user" else "<|assistant|>"
        if index == target_index:
            prompt.append(marker)
            break
        prompt.append(marker)
        participant = message.get("participant")
        if role == "user" and participant:
            prompt.append(f"[participant:{participant}]\n")
        prompt.append(str(message.get("content", "")))
        prompt.append("<|turn_end|>\n")

    target = f"{messages[target_index].get('content', '')}<|turn_end|>"
    return "".join(prompt), target


def encode_sft_record(
    record: dict[str, Any],
    tokenizer: Any,
    *,
    max_length: int,
) -> list[dict[str, Any]]:
    """Tokenize one target, masking all history and retaining every target token."""
    if max_length < 8:
        raise ValueError("max_length must be at least 8")

    prompt, target = render_example_parts(record)
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    target_ids = list(tokenizer.encode(target, add_special_tokens=False))
    if not prompt_ids:
        raise ValueError("Tokenizer produced an empty conversation prompt")
    if not target_ids:
        raise ValueError("Tokenizer produced an empty target")

    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    anchor = ([int(bos_token_id)] if bos_token_id is not None else []) + prompt_ids[:1]
    recent_prompt = prompt_ids[1:]
    maximum_segment = max_length - len(anchor) - 1
    if maximum_segment < 1:
        raise ValueError("max_length is too small for the transcript delimiters")
    segment_size = (
        len(target_ids)
        if len(target_ids) <= maximum_segment
        else min(maximum_segment, max(1, max_length // 2))
    )

    chunks: list[dict[str, Any]] = []
    for start in range(0, len(target_ids), segment_size):
        target_segment = target_ids[start : start + segment_size]
        context = recent_prompt + target_ids[:start]
        context_budget = max_length - len(anchor) - len(target_segment)
        recent_context = context[-context_budget:] if context_budget else []
        input_ids = anchor + recent_context + target_segment
        target_start = len(anchor) + len(recent_context)
        labels = [-100] * target_start + target_segment.copy()
        labels[0] = -100
        chunks.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }
        )
    return chunks


def _tokenize_dataset(dataset: Any, tokenizer: Any, max_length: int) -> Any:
    source_columns = dataset.column_names

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        output: dict[str, list[Any]] = defaultdict(list)
        row_count = len(batch["messages"])
        for row_index in range(row_count):
            record = {column: batch[column][row_index] for column in source_columns}
            for encoded in encode_sft_record(record, tokenizer, max_length=max_length):
                for key, value in encoded.items():
                    output[key].append(value)
        return dict(output)

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=source_columns,
        desc="Tokenizing loss-masked conversations",
    )


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    """Run LoRA SFT. Imports stay local so data preparation has light dependencies."""
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )
    from transformers.trainer_utils import get_last_checkpoint

    train_path = Path(config["train_path"])
    validation_path = Path(config["validation_path"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if not train_path.is_file() or train_path.stat().st_size == 0:
        raise ValueError(f"Training data is empty or missing: {train_path}")

    model_name = str(config["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    configure_tokenizer(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(config["lora_rank"]),
            lora_alpha=int(config["lora_alpha"]),
            lora_dropout=float(config["lora_dropout"]),
            bias="none",
            target_modules="all-linear",
        ),
    )
    model.print_trainable_parameters()

    train_dataset = load_dataset("json", data_files=str(train_path), split="train")
    train_dataset = _tokenize_dataset(
        train_dataset, tokenizer, int(config["max_length"])
    )
    validation_dataset = None
    if validation_path.is_file() and validation_path.stat().st_size:
        validation_dataset = load_dataset(
            "json", data_files=str(validation_path), split="train"
        )
        validation_dataset = _tokenize_dataset(
            validation_dataset, tokenizer, int(config["max_length"])
        )

    has_validation = validation_dataset is not None and len(validation_dataset) > 0
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=str(config["run_name"]),
        num_train_epochs=float(config["epochs"]),
        per_device_train_batch_size=int(config["batch_size"]),
        per_device_eval_batch_size=int(config["batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        learning_rate=float(config["learning_rate"]),
        warmup_ratio=float(config["warmup_ratio"]),
        weight_decay=float(config["weight_decay"]),
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_strategy="steps",
        logging_steps=int(config["logging_steps"]),
        save_strategy="steps",
        save_steps=int(config["save_steps"]),
        save_total_limit=int(config["save_total_limit"]),
        eval_strategy="steps" if has_validation else "no",
        eval_steps=int(config["save_steps"]) if has_validation else None,
        report_to="none",
        remove_unused_columns=False,
        seed=int(config["seed"]),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset if has_validation else None,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=None,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        ),
    )

    resume_checkpoint = None
    if bool(config.get("resume")):
        resume_checkpoint = get_last_checkpoint(str(output_dir))
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    model.config.use_cache = True
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    summary = {
        "model_name": model_name,
        "run_name": str(config["run_name"]),
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset) if has_validation else 0,
        "resumed_from": resume_checkpoint,
        "metrics": train_result.metrics,
        "final_dir": str(final_dir),
    }
    (output_dir / "training-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
