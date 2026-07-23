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


def _render_turn_text(message: dict[str, Any]) -> str:
    role = message.get("role")
    if role not in {"user", "assistant"}:
        raise ValueError(f"Unsupported conversation role {role!r}")
    marker = "<|user|>" if role == "user" else "<|assistant|>"
    parts = [marker]
    participant = message.get("participant")
    if role == "user" and participant:
        parts.append(f"[participant:{participant}]\n")
    parts.append(str(message.get("content", "")))
    parts.append("<|turn_end|>\n")
    return "".join(parts)


def render_session_turns(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Render each conversation turn with an explicit supervised flag."""
    if record.get("format") != SFT_FORMAT:
        raise ValueError(f"Unsupported SFT format {record.get('format')!r}")
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("SFT example must contain at least one message")

    supervised = set(int(index) for index in record.get("supervised_indexes", []))
    if not supervised:
        supervised = {
            index
            for index, message in enumerate(messages)
            if message.get("role") == "assistant"
        }
    if not supervised:
        raise ValueError("SFT example must supervise at least one assistant turn")

    turns: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported conversation role {role!r}")
        is_supervised = index in supervised
        if is_supervised and role != "assistant":
            raise ValueError("Only assistant turns may be supervised")
        turns.append(
            {
                "role": role,
                "text": _render_turn_text(message),
                "supervised": is_supervised,
            }
        )
    return turns


def _tokenize_turn(tokenizer: Any, turn: dict[str, Any]) -> dict[str, Any]:
    token_ids = list(tokenizer.encode(turn["text"], add_special_tokens=False))
    if not token_ids:
        raise ValueError("Tokenizer produced an empty conversation turn")
    return {
        "role": turn["role"],
        "token_ids": token_ids,
        "supervised": bool(turn["supervised"]),
    }


def _pack_window(
    prefix_ids: list[int],
    turns: list[dict[str, Any]],
) -> dict[str, Any]:
    input_ids = list(prefix_ids)
    labels = [-100] * len(prefix_ids)
    for turn in turns:
        start = len(input_ids)
        input_ids.extend(turn["token_ids"])
        if turn["supervised"]:
            labels.extend(turn["token_ids"])
        else:
            labels.extend([-100] * len(turn["token_ids"]))
        # Never train on the first token of the whole sequence.
        if start == 0:
            labels[0] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def _split_oversized_turn(
    turn: dict[str, Any],
    *,
    budget: int,
) -> list[dict[str, Any]]:
    """Split a single turn that cannot fit in one window.

    Supervised turns are sliced into full-budget segments; only the first segment
    carries the role marker, and every segment stays supervised so no target token
    is lost. Oversized masked turns are truncated instead.
    """
    token_ids = turn["token_ids"]
    if budget < 1:
        raise ValueError("max_length is too small for the transcript delimiters")
    if len(token_ids) <= budget:
        return [turn]

    if not turn["supervised"]:
        # Drop excess masked context rather than inventing partial user turns.
        return [{"role": turn["role"], "token_ids": token_ids[:budget], "supervised": False}]

    segments: list[dict[str, Any]] = []
    for start in range(0, len(token_ids), budget):
        segments.append(
            {
                "role": turn["role"],
                "token_ids": token_ids[start : start + budget],
                "supervised": True,
            }
        )
    return segments


def encode_sft_record(
    record: dict[str, Any],
    tokenizer: Any,
    *,
    max_length: int,
) -> list[dict[str, Any]]:
    """Tokenize a session, supervising every assistant turn exactly once.

    Sessions that exceed ``max_length`` are split into contiguous windows. Each
    assistant turn is fully contained in one window and contributes to loss there.
    When a window boundary falls mid-session, earlier turns may be repeated as
    masked context so later assistant replies still see recent history.
    """
    if max_length < 8:
        raise ValueError("max_length must be at least 8")

    rendered = render_session_turns(record)
    tokenized = [_tokenize_turn(tokenizer, turn) for turn in rendered]
    conversation_prefix = list(
        tokenizer.encode("<|conversation|>\n", add_special_tokens=False)
    )
    if not conversation_prefix:
        raise ValueError("Tokenizer produced an empty conversation marker")

    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    prefix = ([int(bos_token_id)] if bos_token_id is not None else []) + conversation_prefix
    if len(prefix) >= max_length:
        raise ValueError("max_length is too small for the transcript delimiters")

    windows: list[dict[str, Any]] = []
    pending = tokenized
    carry: list[dict[str, Any]] = []

    while pending:
        budget = max_length - len(prefix)
        selected: list[dict[str, Any]] = []
        used = 0
        rewrote_pending = False

        # Prefer recent masked carry-over so the next supervised turn keeps history.
        for turn in reversed(carry):
            if used + len(turn["token_ids"]) > budget:
                break
            selected.insert(0, {**turn, "supervised": False})
            used += len(turn["token_ids"])

        consumed = 0
        for turn in pending:
            turn_ids = turn["token_ids"]
            if used + len(turn_ids) <= budget:
                selected.append(turn)
                used += len(turn_ids)
                consumed += 1
                continue

            if not selected or (
                all(not item["supervised"] for item in selected) and turn["supervised"]
            ):
                # Either the window is empty, or it only has masked context that leaves
                # no room for the next supervised turn. Drop the masked filler and pack
                # the supervised turn on its own (splitting if needed).
                selected = []
                used = 0
                segments = _split_oversized_turn(turn, budget=budget)
                selected.append(segments[0])
                pending = segments[1:] + pending[consumed + 1 :]
                rewrote_pending = True
                break

            # Close the window before a turn we cannot fit whole.
            break

        if not selected:
            raise RuntimeError("Failed to pack any turns into a training window")

        window = _pack_window(prefix, selected)
        if len(window["input_ids"]) > max_length:
            raise RuntimeError("Packed window exceeded max_length")
        if all(label == -100 for label in window["labels"]):
            # Keep the masked turns as carry and advance past them.
            carry = [{**turn, "supervised": False} for turn in selected]
            if rewrote_pending:
                continue
            if consumed:
                pending = pending[consumed:]
                continue
            pending = pending[1:]
            continue

        windows.append(window)
        carry = [{**turn, "supervised": False} for turn in selected]
        if not rewrote_pending:
            pending = pending[consumed:]

    # Every supervised turn must appear in exactly one window's labels.
    supervised_token_total = sum(
        len(turn["token_ids"]) for turn in tokenized if turn["supervised"]
    )
    emitted_supervised = sum(
        sum(1 for label in window["labels"] if label != -100) for window in windows
    )
    if emitted_supervised != supervised_token_total:
        raise RuntimeError(
            "Window packing changed the supervised token count: "
            f"expected {supervised_token_total}, got {emitted_supervised}"
        )
    return windows


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
