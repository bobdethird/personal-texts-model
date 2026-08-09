"""Style classifier data preparation and training.

Heavy Hugging Face imports are local to :func:`train_style_classifier`, keeping
metric computation and record validation usable in lightweight environments.
"""

from __future__ import annotations

import math
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


@dataclass(frozen=True, slots=True)
class ClassifierTrainingConfig:
    train_path: str | Path
    validation_path: str | Path
    output_dir: str | Path
    test_path: str | Path | None = None
    model_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 512
    epochs: float = 3.0
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    logging_steps: int = 25
    seed: int = 42
    bf16: bool = False
    resume: bool = True

    def __post_init__(self) -> None:
        if self.max_length < 8:
            raise ValueError("max_length must be at least 8")
        if self.epochs <= 0 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")


def _coerce_config(
    config: ClassifierTrainingConfig | Mapping[str, Any],
) -> ClassifierTrainingConfig:
    if isinstance(config, ClassifierTrainingConfig):
        return config
    names = {field.name for field in fields(ClassifierTrainingConfig)}
    unknown = set(config) - names
    if unknown:
        raise ValueError(f"Unknown classifier configuration keys: {sorted(unknown)}")
    return ClassifierTrainingConfig(**dict(config))


def _join_bubbles(value: Any, *, field: str) -> str:
    if isinstance(value, str):
        bubbles = [value]
    elif isinstance(value, Sequence):
        bubbles = [str(item) for item in value]
    else:
        raise ValueError(f"{field} must be a string or sequence")
    text = "\n".join(bubble.strip() for bubble in bubbles).strip()
    if not text:
        raise ValueError(f"{field} must be nonempty")
    return text


def build_classifier_examples(
    pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Expand each pair into matched original-style and neutral examples."""

    examples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pair in pairs:
        if pair.get("format") != NEUTRAL_PAIR_FORMAT:
            raise ValueError(f"Unsupported neutral pair format {pair.get('format')!r}")
        pair_id = str(pair.get("pair_id", ""))
        if not pair_id or pair_id in seen:
            raise ValueError("Neutral pairs must have unique nonempty pair IDs")
        seen.add(pair_id)
        original = _join_bubbles(pair.get("original"), field="original")
        neutral = _join_bubbles(pair.get("neutral"), field="neutral")
        examples.extend(
            [
                {
                    "pair_id": pair_id,
                    "variant": "original",
                    "text": original,
                    "label": 1,
                },
                {
                    "pair_id": pair_id,
                    "variant": "neutral",
                    "text": neutral,
                    "label": 0,
                },
            ]
        )
    return examples


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        average = (start + 1 + end) / 2
        for position in range(start, end):
            ranks[ordered[position]] = average
        start = end
    return ranks


def binary_classification_metrics(
    labels: Sequence[int],
    style_probabilities: Sequence[float],
    *,
    threshold: float = 0.5,
) -> dict[str, float | None]:
    """Compute dependency-free heldout metrics for original-style probability."""

    if len(labels) != len(style_probabilities) or not labels:
        raise ValueError("labels and style_probabilities must have equal nonzero length")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    normalized_labels = [int(label) for label in labels]
    if any(label not in {0, 1} for label in normalized_labels):
        raise ValueError("labels must be binary")
    probabilities = [float(value) for value in style_probabilities]
    if any(not 0 <= value <= 1 for value in probabilities):
        raise ValueError("style probabilities must be in [0, 1]")

    predictions = [int(value >= threshold) for value in probabilities]
    true_positive = sum(
        label == prediction == 1
        for label, prediction in zip(normalized_labels, predictions, strict=True)
    )
    true_negative = sum(
        label == prediction == 0
        for label, prediction in zip(normalized_labels, predictions, strict=True)
    )
    false_positive = sum(
        label == 0 and prediction == 1
        for label, prediction in zip(normalized_labels, predictions, strict=True)
    )
    false_negative = sum(
        label == 1 and prediction == 0
        for label, prediction in zip(normalized_labels, predictions, strict=True)
    )
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    positives = sum(normalized_labels)
    negatives = len(normalized_labels) - positives
    roc_auc: float | None = None
    if positives and negatives:
        ranks = _average_ranks(probabilities)
        positive_rank_sum = sum(
            rank for rank, label in zip(ranks, normalized_labels, strict=True) if label
        )
        roc_auc = (
            positive_rank_sum - positives * (positives + 1) / 2
        ) / (positives * negatives)

    epsilon = 1e-12
    log_loss = -sum(
        label * math.log(min(1 - epsilon, max(epsilon, probability)))
        + (1 - label) * math.log(min(1 - epsilon, max(epsilon, 1 - probability)))
        for label, probability in zip(normalized_labels, probabilities, strict=True)
    ) / len(normalized_labels)
    brier = sum(
        (probability - label) ** 2
        for label, probability in zip(normalized_labels, probabilities, strict=True)
    ) / len(normalized_labels)
    return {
        "accuracy": (true_positive + true_negative) / len(normalized_labels),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "log_loss": log_loss,
        "brier": brier,
        "positive_rate": positives / len(normalized_labels),
    }


def _probabilities_from_logits(logits: Any) -> list[float]:
    values = logits[0] if isinstance(logits, tuple) else logits
    rows = values.tolist() if hasattr(values, "tolist") else values
    probabilities = []
    for row in rows:
        negative_logit, positive_logit = float(row[0]), float(row[1])
        difference = max(-60.0, min(60.0, positive_logit - negative_logit))
        probabilities.append(1.0 / (1.0 + math.exp(-difference)))
    return probabilities


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


def train_style_classifier(
    config: ClassifierTrainingConfig | Mapping[str, Any],
) -> dict[str, Any]:
    """Train an original-vs-neutral classifier and report heldout metrics."""

    settings = _coerce_config(config)

    # Lazy imports keep package import and pure unit tests lightweight.
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )
    from transformers.trainer_utils import get_last_checkpoint

    train_pairs = list(read_jsonl(settings.train_path))
    validation_pairs = list(read_jsonl(settings.validation_path))
    test_pairs = list(read_jsonl(settings.test_path)) if settings.test_path is not None else []
    if not train_pairs or not validation_pairs:
        raise ValueError("Classifier training and validation pair files must both be nonempty")
    train_examples = build_classifier_examples(train_pairs)
    validation_examples = build_classifier_examples(validation_pairs)
    test_examples = build_classifier_examples(test_pairs)

    tokenizer = AutoTokenizer.from_pretrained(settings.model_name, use_fast=True)

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=settings.max_length,
        )

    train_dataset = Dataset.from_list(train_examples).map(
        tokenize,
        batched=True,
        remove_columns=["text", "pair_id", "variant"],
    )
    validation_dataset = Dataset.from_list(validation_examples).map(
        tokenize,
        batched=True,
        remove_columns=["text", "pair_id", "variant"],
    )
    test_dataset = (
        Dataset.from_list(test_examples).map(
            tokenize,
            batched=True,
            remove_columns=["text", "pair_id", "variant"],
        )
        if test_examples
        else None
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        settings.model_name,
        num_labels=2,
        id2label={0: "neutral", 1: "original"},
        label2id={"neutral": 0, "original": 1},
    )
    output_dir = ensure_private_dir(settings.output_dir)
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=settings.epochs,
        per_device_train_batch_size=settings.batch_size,
        per_device_eval_batch_size=settings.batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
        warmup_ratio=settings.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=settings.logging_steps,
        report_to="none",
        seed=settings.seed,
        bf16=settings.bf16,
    )

    def compute_metrics(evaluation: Any) -> dict[str, float]:
        probabilities = _probabilities_from_logits(evaluation.predictions)
        labels = (
            evaluation.label_ids.tolist()
            if hasattr(evaluation.label_ids, "tolist")
            else evaluation.label_ids
        )
        return {
            key: float(value)
            for key, value in binary_classification_metrics(labels, probabilities).items()
            if value is not None
        }

    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    resume_checkpoint = get_last_checkpoint(str(output_dir)) if settings.resume else None
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    prediction = trainer.predict(validation_dataset)
    heldout = compute_metrics(prediction)
    test_metrics = (
        compute_metrics(trainer.predict(test_dataset)) if test_dataset is not None else None
    )

    final_dir = ensure_private_dir(output_dir / "final")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    summary = {
        "model_name": settings.model_name,
        "train_pairs": len(train_pairs),
        "validation_pairs": len(validation_pairs),
        "train_examples": len(train_examples),
        "heldout_examples": len(validation_examples),
        "test_examples": len(test_examples),
        "resumed_from": resume_checkpoint,
        "train_metrics": _jsonable(train_result.metrics),
        "heldout_metrics": heldout,
        "test_metrics": test_metrics,
        "final_dir": str(final_dir),
    }
    write_json(output_dir / "training-summary.json", summary)
    _privatize_tree(output_dir)
    return summary
