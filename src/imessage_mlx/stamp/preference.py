"""CPO preference records and reference-free preference training."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from imessage_mlx.stamp.rewards import HopeFearSelection, RewardExponents
from imessage_mlx.stamp.sft import REWRITE_SYSTEM_PROMPT
from imessage_mlx.utils import (
    ensure_private_dir,
    ensure_private_file,
    read_jsonl,
    write_json,
    write_jsonl,
)

CPO_PAIR_FORMAT = "stamp-cpo-pair-v1"


@dataclass(frozen=True, slots=True)
class CPOTrainingConfig:
    train_path: str | Path
    validation_path: str | Path | None
    output_dir: str | Path
    model_name_or_path: str
    max_length: int = 1024
    epochs: float = 1.0
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 2
    seed: int = 42
    bf16: bool = True
    gradient_checkpointing: bool = True
    beta: float = 0.1
    cpo_alpha: float = 1.0
    loss_type: str = "sigmoid"
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
        if self.loss_type not in {"sigmoid", "hinge", "ipo", "simpo", "alphapo"}:
            raise ValueError("Unsupported CPO loss_type")
        if self.beta <= 0 or self.cpo_alpha < 0:
            raise ValueError("beta must be positive and cpo_alpha nonnegative")


def _coerce_config(config: CPOTrainingConfig | Mapping[str, Any]) -> CPOTrainingConfig:
    if isinstance(config, CPOTrainingConfig):
        return config
    names = {field.name for field in fields(CPOTrainingConfig)}
    unknown = set(config) - names
    if unknown:
        raise ValueError(f"Unknown CPO configuration keys: {sorted(unknown)}")
    return CPOTrainingConfig(**dict(config))


def _bubbles(value: str | Sequence[str], *, field: str) -> list[str]:
    if isinstance(value, str):
        bubbles = [value]
    elif isinstance(value, Sequence):
        bubbles = [str(item) for item in value]
    else:
        raise ValueError(f"{field} must be a string or sequence")
    if not bubbles or any(not item.strip() for item in bubbles):
        raise ValueError(f"{field} must contain nonempty bubbles")
    return bubbles


def format_rewrite_completion(bubbles: str | Sequence[str]) -> str:
    return json.dumps(
        {"original_bubbles": _bubbles(bubbles, field="completion")}, ensure_ascii=False
    )


def format_cpo_prompt(
    neutral_bubbles: str | Sequence[str],
    *,
    system_prompt: str = REWRITE_SYSTEM_PROMPT,
) -> str:
    """Render the Qwen chat prefix as a standard string for preference training."""
    neutral = json.dumps(_bubbles(neutral_bubbles, field="neutral"), ensure_ascii=False)
    user = (
        f"NEUTRAL_DRAFT:\n{neutral}\n\n"
        'Return JSON only: {"original_bubbles": ["one or more styled bubbles"]}'
    )
    return (
        f"<|im_start|>system\n{system_prompt.strip()}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def make_cpo_pair(
    *,
    pair_id: str,
    neutral_bubbles: str | Sequence[str],
    chosen_bubbles: str | Sequence[str],
    rejected_bubbles: str | Sequence[str],
    system_prompt: str = REWRITE_SYSTEM_PROMPT,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build TRL's standard string ``prompt/chosen/rejected`` record."""

    prompt = format_cpo_prompt(neutral_bubbles, system_prompt=system_prompt)
    chosen = format_rewrite_completion(chosen_bubbles)
    rejected = format_rewrite_completion(rejected_bubbles)
    if chosen == rejected:
        raise ValueError("CPO chosen and rejected completions must differ")
    record: dict[str, Any] = {
        "format": CPO_PAIR_FORMAT,
        "pair_id": str(pair_id),
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }
    if not record["pair_id"]:
        raise ValueError("CPO pair_id must be nonempty")
    if metadata:
        record["metadata"] = dict(metadata)
    return record


def make_cpo_pair_from_selection(
    *,
    pair_id: str,
    neutral_bubbles: str | Sequence[str],
    selection: HopeFearSelection,
    exponents: RewardExponents | None = None,
    system_prompt: str = REWRITE_SYSTEM_PROMPT,
) -> dict[str, Any]:
    """Serialize a deterministic hope/fear selection as one CPO pair."""

    return make_cpo_pair(
        pair_id=pair_id,
        neutral_bubbles=neutral_bubbles,
        chosen_bubbles=selection.hope.text,
        rejected_bubbles=selection.fear.text,
        system_prompt=system_prompt,
        metadata={
            "hope_candidate_id": selection.hope.candidate_id,
            "fear_candidate_id": selection.fear.candidate_id,
            "hope_score": selection.hope_score,
            "fear_score": selection.fear_score,
            "reward_exponents": (exponents or RewardExponents()).as_dict(),
        },
    )


def validate_cpo_pair(record: Mapping[str, Any]) -> None:
    if record.get("format") != CPO_PAIR_FORMAT:
        raise ValueError(f"Unsupported CPO pair format {record.get('format')!r}")
    if not str(record.get("pair_id", "")):
        raise ValueError("CPO pair_id must be nonempty")
    for field in ("prompt", "chosen", "rejected"):
        if not isinstance(record.get(field), str) or not str(record[field]).strip():
            raise ValueError(f"CPO {field} must be a nonempty string")
    if record["chosen"] == record["rejected"]:
        raise ValueError("CPO chosen and rejected completions must differ")


def write_cpo_pairs(
    output_path: str | Path,
    records: Sequence[Mapping[str, Any]],
) -> int:
    seen: set[str] = set()
    serialized: list[dict[str, Any]] = []
    for record in records:
        validate_cpo_pair(record)
        pair_id = str(record["pair_id"])
        if pair_id in seen:
            raise ValueError(f"Duplicate CPO pair_id {pair_id!r}")
        seen.add(pair_id)
        serialized.append(dict(record))
    return write_jsonl(output_path, serialized)


def cpo_pair_loss(
    chosen_log_probability: float,
    rejected_log_probability: float,
    *,
    chosen_nll: float,
    beta: float = 0.1,
    cpo_alpha: float = 1.0,
) -> float:
    """Return the reference-free CPO sigmoid loss for one preference pair."""
    if beta <= 0 or cpo_alpha < 0:
        raise ValueError("beta must be positive and cpo_alpha nonnegative")
    margin = beta * (chosen_log_probability - rejected_log_probability)
    preference = math.log1p(math.exp(-abs(margin))) + max(-margin, 0.0)
    return preference + cpo_alpha * chosen_nll


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


def train_cpo(
    config: CPOTrainingConfig | Mapping[str, Any],
) -> dict[str, Any]:
    """Train a PEFT LoRA with reference-free CPO and resumable checkpoints.

    Current TRL releases removed ``CPOTrainer`` while the last release that
    included it predates Qwen 3 support. The objective is small, so keeping it
    here avoids an incompatible Transformers downgrade:

    ``-logsigmoid(beta * (logp(chosen) - logp(rejected))) + alpha * NLL(chosen)``.
    """

    settings = _coerce_config(config)
    import torch
    import torch.nn.functional as functional
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from transformers.trainer_utils import get_last_checkpoint

    tokenizer = AutoTokenizer.from_pretrained(settings.model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer must define an EOS or pad token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    output_dir = ensure_private_dir(settings.output_dir)
    model_kwargs = {
        key: value
        for key, value in {
            "dtype": settings.dtype,
            "attn_implementation": settings.attn_implementation,
        }.items()
        if value is not None
    }
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_name_or_path,
        **model_kwargs,
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=settings.lora_rank,
            lora_alpha=settings.lora_alpha,
            lora_dropout=settings.lora_dropout,
            bias="none",
            target_modules=settings.lora_target_modules,
        ),
    )

    def encode_records(path: str | Path | None) -> Any | None:
        if path is None:
            return None
        encoded: list[dict[str, list[int]]] = []
        for record in read_jsonl(path):
            validate_cpo_pair(record)
            prompt_ids = list(tokenizer.encode(record["prompt"], add_special_tokens=False))
            if tokenizer.bos_token_id is not None:
                prompt_ids = [int(tokenizer.bos_token_id), *prompt_ids]
            stable_prompt_ids = tuple(prompt_ids)

            def encode_completion(
                value: str,
                base_prompt_ids: tuple[int, ...] = stable_prompt_ids,
            ) -> tuple[list[int], list[int]]:
                completion_ids = list(tokenizer.encode(value, add_special_tokens=False))
                im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
                if (
                    isinstance(im_end_id, int)
                    and im_end_id >= 0
                    and im_end_id != tokenizer.unk_token_id
                ):
                    completion_ids.append(im_end_id)
                if tokenizer.eos_token_id is not None:
                    completion_ids.append(int(tokenizer.eos_token_id))
                completion_ids = completion_ids[: settings.max_length - 1]
                prompt_budget = max(1, settings.max_length - len(completion_ids))
                kept_prompt = list(base_prompt_ids[-prompt_budget:])
                input_ids = [*kept_prompt, *completion_ids]
                labels = [-100] * len(kept_prompt) + completion_ids
                return input_ids, labels

            chosen_ids, chosen_labels = encode_completion(str(record["chosen"]))
            rejected_ids, rejected_labels = encode_completion(str(record["rejected"]))
            encoded.append(
                {
                    "chosen_input_ids": chosen_ids,
                    "chosen_labels": chosen_labels,
                    "rejected_input_ids": rejected_ids,
                    "rejected_labels": rejected_labels,
                }
            )
        return Dataset.from_list(encoded) if encoded else None

    train_dataset = encode_records(settings.train_path)
    if train_dataset is None:
        raise ValueError("CPO training pairs must be nonempty")
    validation_dataset = encode_records(settings.validation_path)
    has_validation = validation_dataset is not None

    class PreferenceCollator:
        def __call__(self, features: list[Mapping[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            maximum_length = max(
                len(feature[f"{side}_input_ids"])
                for feature in features
                for side in ("chosen", "rejected")
            )
            for side in ("chosen", "rejected"):
                sequences = [
                    torch.tensor(feature[f"{side}_input_ids"], dtype=torch.long)
                    for feature in features
                ]
                labels = [
                    torch.tensor(feature[f"{side}_labels"], dtype=torch.long)
                    for feature in features
                ]
                result[f"{side}_input_ids"] = torch.nn.utils.rnn.pad_sequence(
                    sequences,
                    batch_first=True,
                    padding_value=int(tokenizer.pad_token_id),
                )
                input_padding = maximum_length - result[f"{side}_input_ids"].shape[1]
                if input_padding:
                    result[f"{side}_input_ids"] = functional.pad(
                        result[f"{side}_input_ids"],
                        (0, input_padding),
                        value=int(tokenizer.pad_token_id),
                    )
                result[f"{side}_attention_mask"] = (
                    result[f"{side}_input_ids"] != int(tokenizer.pad_token_id)
                ).long()
                result[f"{side}_labels"] = torch.nn.utils.rnn.pad_sequence(
                    labels,
                    batch_first=True,
                    padding_value=-100,
                )
                label_padding = maximum_length - result[f"{side}_labels"].shape[1]
                if label_padding:
                    result[f"{side}_labels"] = functional.pad(
                        result[f"{side}_labels"],
                        (0, label_padding),
                        value=-100,
                    )
            return result

    class ReferenceFreeCPOTrainer(Trainer):
        def compute_loss(
            self,
            model: Any,
            inputs: Mapping[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            del num_items_in_batch
            chosen_size = inputs["chosen_input_ids"].shape[0]
            input_ids = torch.cat(
                (inputs["chosen_input_ids"], inputs["rejected_input_ids"]),
                dim=0,
            )
            attention_mask = torch.cat(
                (inputs["chosen_attention_mask"], inputs["rejected_attention_mask"]),
                dim=0,
            )
            labels = torch.cat(
                (inputs["chosen_labels"], inputs["rejected_labels"]),
                dim=0,
            )
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :].float()
            shifted_labels = labels[:, 1:]
            mask = shifted_labels != -100
            safe_labels = shifted_labels.masked_fill(~mask, 0)
            token_logps = torch.gather(
                functional.log_softmax(logits, dim=-1),
                dim=2,
                index=safe_labels.unsqueeze(2),
            ).squeeze(2)
            token_logps = token_logps * mask
            sequence_logps = token_logps.sum(dim=1)
            token_counts = mask.sum(dim=1).clamp_min(1)
            chosen_logps = sequence_logps[:chosen_size]
            rejected_logps = sequence_logps[chosen_size:]
            chosen_nll = -(chosen_logps / token_counts[:chosen_size])
            preference = -functional.logsigmoid(
                settings.beta * (chosen_logps - rejected_logps)
            )
            loss = (preference + settings.cpo_alpha * chosen_nll).mean()
            if return_outputs:
                return loss, {
                    "chosen_logps": chosen_logps.detach(),
                    "rejected_logps": rejected_logps.detach(),
                }
            return loss

    arguments = TrainingArguments(
        output_dir=str(output_dir),
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
        remove_unused_columns=False,
        label_names=["chosen_labels", "rejected_labels"],
    )
    trainer = ReferenceFreeCPOTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=PreferenceCollator(),
    )
    resume_checkpoint = get_last_checkpoint(str(output_dir)) if settings.resume else None
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    final_dir = ensure_private_dir(output_dir / "final")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    summary = {
        "trainer": "reference-free-cpo",
        "model_name_or_path": settings.model_name_or_path,
        "train_pairs": len(train_dataset),
        "validation_pairs": len(validation_dataset) if has_validation else 0,
        "resumed_from": resume_checkpoint,
        "loss_type": settings.loss_type,
        "train_metrics": _jsonable(train_result.metrics),
        "final_dir": str(final_dir),
    }
    write_json(output_dir / "training-summary.json", summary)
    _privatize_tree(output_dir)
    return summary
