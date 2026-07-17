from __future__ import annotations

import argparse
import math
import os
import random
import re
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from imessage_mlx.config import load_yaml
from imessage_mlx.seq2seq_profile import format_seq2seq_source, resolve_seq2seq_profile
from imessage_mlx.utils import ensure_private_dir, read_jsonl, write_json, write_jsonl

REWRITE_INSTRUCTION = (
    "Rewrite the draft in the learned casual texting style. Preserve every fact, intent, "
    "question, negation, and degree of uncertainty. Return only the rewritten message.\n\nDraft:\n"
)
WORD_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
# Short drafts legitimately collapse into initialisms ("What do you mean?" -> "wdym"),
# so the candidate length floor only applies to sources long enough to drop content.
LENGTH_FLOOR_MIN_SOURCE_WORDS = 5


def normalized_words(text: str) -> list[str]:
    return WORD_PATTERN.findall(text.lower().replace("’", "'"))


def multiset_jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_counts = Counter(left)
    right_counts = Counter(right)
    union = sum((left_counts | right_counts).values())
    if not union:
        return 1.0
    return sum((left_counts & right_counts).values()) / union


def transformation_sampling_weights(
    rows: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any] | None,
) -> tuple[list[float], dict[str, Any]]:
    """Return deterministic sampling weights and an auditable distribution summary."""
    options = dict(settings or {})
    enabled = bool(options.get("enabled", False))
    if not enabled:
        return [1.0] * len(rows), {
            "enabled": False,
            "counts": {"standard": len(rows)},
            "expected_share": {"standard": 1.0 if rows else 0.0},
        }

    exact_weight = float(options.get("exact_copy_weight", 0.25))
    near_threshold = float(options.get("near_copy_threshold", 0.9))
    near_weight = float(options.get("near_copy_weight", 0.5))
    strong_threshold = float(options.get("strong_threshold", 0.7))
    strong_weight = float(options.get("strong_weight", 1.5))
    minimum_ratio = float(options.get("strong_min_length_ratio", 0.65))
    maximum_ratio = float(options.get("strong_max_length_ratio", 1.2))
    require_numbers = bool(options.get("require_source_numbers", True))
    deletion_ratio = float(options.get("severe_deletion_length_ratio", 0.5))
    deletion_similarity = float(options.get("severe_deletion_similarity", 0.5))
    deletion_weight = float(options.get("severe_deletion_weight", 1.0))
    if min(exact_weight, near_weight, strong_weight, deletion_weight) <= 0:
        raise ValueError("Sampling weights must be positive")
    if not 0 <= strong_threshold <= near_threshold <= 1:
        raise ValueError("Sampling thresholds must satisfy 0 <= strong <= near <= 1")
    if minimum_ratio <= 0 or maximum_ratio < minimum_ratio:
        raise ValueError("Strong transformation length-ratio bounds are invalid")

    weights: list[float] = []
    category_counts: Counter[str] = Counter()
    category_weight: Counter[str] = Counter()
    for row in rows:
        source = normalized_words(str(row["source"]))
        target = normalized_words(str(row["target"]))
        similarity = multiset_jaccard(source, target)
        length_ratio = len(target) / max(1, len(source))
        source_numbers = Counter(token for token in source if token.isdigit())
        target_numbers = Counter(token for token in target if token.isdigit())
        numbers_preserved = not (source_numbers - target_numbers)

        if source == target:
            category, weight = "exact_copy", exact_weight
        elif similarity >= near_threshold:
            category, weight = "near_copy", near_weight
        elif length_ratio < deletion_ratio and similarity < deletion_similarity:
            # Targets that silently drop most of the source teach length collapse.
            category, weight = "severe_deletion", deletion_weight
        elif (
            similarity < strong_threshold
            and minimum_ratio <= length_ratio <= maximum_ratio
            and (numbers_preserved or not require_numbers)
        ):
            category, weight = "quality_gated_strong", strong_weight
        else:
            category, weight = "standard", 1.0
        weights.append(weight)
        category_counts[category] += 1
        category_weight[category] += weight

    total_weight = sum(weights)
    return weights, {
        "enabled": True,
        "settings": options,
        "counts": dict(sorted(category_counts.items())),
        "expected_share": {
            category: round(weight / total_weight, 6)
            for category, weight in sorted(category_weight.items())
        },
    }


def decoding_options(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve inference guardrail settings; defaults keep legacy greedy decoding."""
    options = dict(config.get("decoding") or {})
    resolved = {
        "best_of": int(options.get("best_of", 1)),
        "temperature": float(options.get("temperature", 0.8)),
        "top_p": float(options.get("top_p", 0.9)),
        "min_new_tokens_ratio": float(options.get("min_new_tokens_ratio", 0.0)),
        "min_new_tokens_cap": int(options.get("min_new_tokens_cap", 24)),
        "length_ratio_min": float(options.get("length_ratio_min", 0.3)),
        "length_ratio_max": float(options.get("length_ratio_max", 1.6)),
        "copy_escape_jaccard": float(options.get("copy_escape_jaccard", 1.0)),
    }
    if resolved["best_of"] < 1:
        raise ValueError("decoding.best_of must be at least 1")
    if resolved["min_new_tokens_ratio"] < 0:
        raise ValueError("decoding.min_new_tokens_ratio cannot be negative")
    if not 0 < resolved["length_ratio_min"] <= resolved["length_ratio_max"]:
        raise ValueError("decoding length-ratio band is invalid")
    return resolved


def minimum_new_tokens(source_text: str, options: Mapping[str, Any]) -> int:
    ratio = float(options["min_new_tokens_ratio"])
    if ratio <= 0:
        return 0
    floor = round(ratio * len(normalized_words(source_text)))
    return max(1, min(int(options["min_new_tokens_cap"]), floor))


def select_rewrite_candidate(
    source_text: str,
    candidates: Sequence[str],
    options: Mapping[str, Any],
) -> str:
    """Trust the greedy candidate unless it fails checks or copies the source.

    Candidates that lose source numerals, flip negation polarity, or fall outside the
    word-length band are rejected. The greedy candidate wins whenever it passes checks
    and is not a near-copy; otherwise the escape picks the surviving alternative that
    deviates least while still clearing the copy threshold, so a forced escape never
    overshoots into a wild sample. The greedy candidate is the fallback when nothing
    survives.
    """
    from imessage_mlx.adapter_evaluation import NEGATIONS, normalized_numbers

    if not candidates:
        raise ValueError("Candidate selection requires at least one candidate")
    source_words = normalized_words(source_text)
    source_numbers = normalized_numbers(source_text)
    source_negated = any(word in NEGATIONS for word in source_words)
    copy_threshold = float(options["copy_escape_jaccard"])

    survivors: list[tuple[float, int, str]] = []
    for index, candidate in enumerate(candidates):
        words = normalized_words(candidate)
        if not words:
            continue
        if source_words:
            ratio = len(words) / len(source_words)
            if ratio > options["length_ratio_max"]:
                continue
            if (
                len(source_words) >= LENGTH_FLOOR_MIN_SOURCE_WORDS
                and ratio < options["length_ratio_min"]
            ):
                continue
        if source_numbers - normalized_numbers(candidate):
            continue
        if source_negated != any(word in NEGATIONS for word in words):
            continue
        survivors.append((multiset_jaccard(source_words, words), index, candidate))
    if not survivors:
        return candidates[0]
    greedy_survivor = next((entry for entry in survivors if entry[1] == 0), None)
    if greedy_survivor is not None and greedy_survivor[0] < copy_threshold:
        return greedy_survivor[2]
    transformed = [entry for entry in survivors if entry[0] < copy_threshold]
    if transformed:
        return max(transformed)[2]
    return greedy_survivor[2] if greedy_survivor is not None else survivors[0][2]


def _chmod_private_tree(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)
    root.chmod(0o700)


def _device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _qwen_model_reference(config: dict[str, Any]) -> str:
    from huggingface_hub import snapshot_download

    model_id = str(config["base_model"])
    revision = str(config["revision"])
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    snapshot = hub / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
    if snapshot.exists():
        return str(snapshot)
    return snapshot_download(model_id, revision=revision)


def train_seq2seq(
    config_path: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq

    config = load_yaml(config_path)
    profile = resolve_seq2seq_profile(config)
    seed = int(config.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    device = _device(torch)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    output = ensure_private_dir(output_dir)
    train_rows = list(read_jsonl(Path(data_dir) / "train.jsonl"))
    valid_rows = list(read_jsonl(Path(data_dir) / "valid.jsonl"))
    if not train_rows or not valid_rows:
        raise ValueError("Seq2seq adapter training requires non-empty train and valid JSONL")

    base_model = str(config["base_model"])
    revision = str(config.get("revision", "main"))
    tokenizer = AutoTokenizer.from_pretrained(base_model, revision=revision)
    max_length = int(config.get("max_length", 256))

    def within_token_limit(row: dict[str, Any]) -> bool:
        source_ids = tokenizer(
            format_seq2seq_source(str(row["source"]), profile),
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
        target_ids = tokenizer(
            str(row["target"]),
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
        return len(source_ids) <= max_length and len(target_ids) <= max_length

    original_train_count = len(train_rows)
    original_valid_count = len(valid_rows)
    train_rows = [row for row in train_rows if within_token_limit(row)]
    valid_rows = [row for row in valid_rows if within_token_limit(row)]
    if not train_rows or not valid_rows:
        raise ValueError("Seq2seq length filtering removed every train or validation example")
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model, revision=revision)
    full_finetune = bool(config.get("full_finetune", False))
    if not full_finetune:
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False,
                r=int(config.get("lora_rank", 8)),
                lora_alpha=int(config.get("lora_alpha", 16)),
                lora_dropout=float(config.get("lora_dropout", 0.1)),
                target_modules=list(profile.lora_target_modules),
            ),
        )
    model.to(device)

    class PairDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            row = self.rows[index]
            return tokenizer(
                format_seq2seq_source(str(row["source"]), profile),
                text_target=str(row["target"]),
                max_length=max_length,
                truncation=True,
            )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, return_tensors="pt")
    sampling_weights, sampling_report = transformation_sampling_weights(
        train_rows,
        config.get("sampling"),
    )
    sampling_enabled = bool(sampling_report["enabled"])
    sampler = (
        WeightedRandomSampler(
            sampling_weights,
            num_samples=len(train_rows),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        if sampling_enabled
        else None
    )
    train_loader = DataLoader(
        PairDataset(train_rows),
        batch_size=int(config.get("batch_size", 2)),
        shuffle=not sampling_enabled,
        sampler=sampler,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collator,
        num_workers=0,
    )
    valid_loader = DataLoader(
        PairDataset(valid_rows),
        batch_size=int(config.get("batch_size", 2)),
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_parameters = sum(parameter.numel() for parameter in trainable)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config.get("learning_rate", 2e-4)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    accumulation = int(config.get("gradient_accumulation_steps", 8))
    epochs = int(config.get("epochs", 3))
    total_updates = max(1, math.ceil(len(train_loader) / accumulation) * epochs)
    warmup_updates = int(total_updates * float(config.get("warmup_fraction", 0.05)))

    def learning_rate(update: int) -> float:
        if update < warmup_updates:
            return max(1, update + 1) / max(1, warmup_updates)
        progress = (update - warmup_updates) / max(1, total_updates - warmup_updates)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate)
    patience = int(config.get("early_stopping_patience", 2))
    best_loss = math.inf
    stale_epochs = 0
    updates = 0
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    total_batches = len(train_loader) * epochs
    completed_batches = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_index, batch in enumerate(train_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss / accumulation
            loss.backward()
            train_loss += float(loss.item()) * accumulation
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
            completed_batches += 1
            if completed_batches % 10 == 0 or batch_index + 1 == len(train_loader):
                elapsed = max(time.perf_counter() - started, 0.001)
                batches_per_second = completed_batches / elapsed
                remaining = max(total_batches - completed_batches, 0)
                eta_minutes = remaining / batches_per_second / 60
                fraction = completed_batches / total_batches
                width = 30
                filled = min(width, int(width * fraction))
                bar = "#" * filled + "-" * (width - filled)
                print(
                    f"\r[{bar}] epoch {epoch + 1}/{epochs} "
                    f"batch {batch_index + 1}/{len(train_loader)} "
                    f"loss {train_loss / (batch_index + 1):.4f} "
                    f"{batches_per_second:.2f} batch/s ETA {eta_minutes:.1f}m",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )

        print("\nvalidating...", file=sys.stderr, flush=True)
        model.eval()
        validation_loss = 0.0
        validation_batches = 0
        with torch.no_grad():
            for batch in valid_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                validation_loss += float(model(**batch).loss.item())
                validation_batches += 1
        validation_loss /= max(1, validation_batches)
        history.append(
            {
                "epoch": epoch + 1,
                "training_loss": train_loss / max(1, len(train_loader)),
                "validation_loss": validation_loss,
            }
        )
        print(
            f"epoch {epoch + 1}: train_loss={history[-1]['training_loss']:.4f} "
            f"validation_loss={validation_loss:.4f}",
            file=sys.stderr,
            flush=True,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            stale_epochs = 0
            model.save_pretrained(output / "adapter", safe_serialization=True)
            tokenizer.save_pretrained(output / "tokenizer")
            _chmod_private_tree(output)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    elapsed = time.perf_counter() - started
    if device.type == "cuda":
        peak_memory = int(torch.cuda.max_memory_allocated())
    elif device.type == "mps":
        peak_memory = int(torch.mps.driver_allocated_memory())
    else:
        peak_memory = 0
    report = {
        "schema_version": 1,
        "architecture": profile.architecture,
        "base_model": base_model,
        "base_revision": revision,
        "base_model_license": profile.base_model_license,
        "source_prefix": profile.source_prefix,
        "full_finetune": full_finetune,
        "lora_target_modules": None if full_finetune else list(profile.lora_target_modules),
        "lora_rank": None if full_finetune else int(config.get("lora_rank", 8)),
        "lora_alpha": None if full_finetune else int(config.get("lora_alpha", 16)),
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_parameter_fraction": trainable_parameters / total_parameters,
        "sampling": sampling_report,
        "train_examples": len(train_rows),
        "validation_examples": len(valid_rows),
        "skipped_oversized_train_examples": original_train_count - len(train_rows),
        "skipped_oversized_validation_examples": original_valid_count - len(valid_rows),
        "epochs_completed": len(history),
        "optimizer_updates": updates,
        "best_validation_loss": best_loss,
        "elapsed_seconds": elapsed,
        "examples_per_second": len(train_rows) * len(history) / max(elapsed, 1e-9),
        "peak_memory_bytes": peak_memory,
        "device": str(device),
        "history": history,
        "private_local_artifact": True,
    }
    write_json(output / "training-report.json", report)
    _chmod_private_tree(output)
    return report


def _generate_rewrite_candidates(
    model,
    tokenizer,
    encoded: Mapping[str, Any],
    source_text: str,
    max_new_tokens: int,
    options: Mapping[str, Any],
) -> list[str]:
    floor = minimum_new_tokens(source_text, options)
    length_options = {"min_new_tokens": floor} if floor else {}
    greedy = model.generate(
        **encoded,
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        **length_options,
    )
    candidates = [tokenizer.decode(greedy[0], skip_special_tokens=True).strip()]
    sampled_count = int(options["best_of"]) - 1
    if sampled_count > 0:
        sampled = model.generate(
            **encoded,
            do_sample=True,
            temperature=float(options["temperature"]),
            top_p=float(options["top_p"]),
            num_return_sequences=sampled_count,
            max_new_tokens=max_new_tokens,
            **length_options,
        )
        candidates.extend(
            tokenizer.decode(sequence, skip_special_tokens=True).strip() for sequence in sampled
        )
    return candidates


def predict_seq2seq(
    config_path: str | Path,
    data_path: str | Path,
    adapter_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    config = load_yaml(config_path)
    profile = resolve_seq2seq_profile(config)
    options = decoding_options(config)
    torch.manual_seed(int(config.get("seed", 42)))
    device = _device(torch)
    base_model = str(config["base_model"])
    revision = str(config.get("revision", "main"))
    tokenizer_path = Path(adapter_dir) / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if bool(config.get("full_finetune", False)):
        model = AutoModelForSeq2SeqLM.from_pretrained(Path(adapter_dir) / "adapter")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model, revision=revision)
        model = PeftModel.from_pretrained(model, Path(adapter_dir) / "adapter")
    model.to(device)
    model.eval()
    rows = list(read_jsonl(data_path))
    started = time.perf_counter()

    def predictions():
        with torch.no_grad():
            for index, row in enumerate(rows, start=1):
                encoded = tokenizer(
                    format_seq2seq_source(str(row["source"]), profile),
                    return_tensors="pt",
                    max_length=int(config.get("max_length", 256)),
                    truncation=True,
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                candidates = _generate_rewrite_candidates(
                    model,
                    tokenizer,
                    encoded,
                    str(row["source"]),
                    int(config.get("max_new_tokens", 64)),
                    options,
                )
                selected = select_rewrite_candidate(str(row["source"]), candidates, options)
                metadata = {
                    key: str(row[key])
                    for key in ("target_id", "variant_kind", "split", "source_pair_id")
                    if key in row
                }
                if index % 10 == 0 or index == len(rows):
                    elapsed = max(time.perf_counter() - started, 0.001)
                    rate = index / elapsed
                    eta_minutes = (len(rows) - index) / rate / 60
                    fraction = index / len(rows) if rows else 1.0
                    width = 30
                    filled = min(width, int(width * fraction))
                    bar = "#" * filled + "-" * (width - filled)
                    print(
                        f"\r[{bar}] {index}/{len(rows)} ({fraction:6.2%}) "
                        f"{rate:.2f} examples/s ETA {eta_minutes:.1f}m",
                        end="",
                        file=sys.stderr,
                        flush=True,
                    )
                yield {
                    "pair_id": str(row["pair_id"]),
                    "neutral_text": str(row["source"]),
                    "target_text": str(row["target"]),
                    "generated_text": selected,
                    **metadata,
                }

    count = write_jsonl(output_path, predictions())
    print(file=sys.stderr)
    elapsed = time.perf_counter() - started
    return {
        "architecture": profile.architecture,
        "predictions": count,
        "elapsed_seconds": elapsed,
        "examples_per_second": count / max(elapsed, 1e-9),
        "output": str(Path(output_path)),
    }


def _translategemma_prompt(tokenizer, draft: str, config: Mapping[str, Any]) -> str:
    """Render TranslateGemma's mandatory structured translation template (en -> en)."""
    content = [
        {
            "type": "text",
            "source_lang_code": str(config.get("source_lang_code", "en")),
            "target_lang_code": str(config.get("target_lang_code", "en")),
            "text": draft,
        }
    ]
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    # generate() re-encodes with a BOS token, so drop the template's own.
    return prompt.removeprefix("<bos>")


def _mlx_generation_prompt(
    tokenizer,
    draft: str,
    raw_prompt: str,
    config: Mapping[str, Any],
) -> str:
    if str(config.get("chat_format", "")) == "translategemma":
        return _translategemma_prompt(tokenizer, draft, config)
    return raw_prompt


def predict_qwen(
    config_path: str | Path,
    data_path: str | Path,
    adapter_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    from mlx_lm import generate, load

    config = load_yaml(config_path)
    model, tokenizer = load(_qwen_model_reference(config), adapter_path=str(adapter_dir))
    rows = list(read_jsonl(data_path))
    started = time.perf_counter()

    def predictions():
        for index, row in enumerate(rows, start=1):
            draft = str(row["prompt"]).rsplit("Draft:\n", maxsplit=1)[-1]
            output = generate(
                model,
                tokenizer,
                prompt=_mlx_generation_prompt(tokenizer, draft, str(row["prompt"]), config),
                max_tokens=int(config.get("max_new_tokens", 64)),
                verbose=False,
            )
            metadata = {
                key: str(row[key])
                for key in ("target_id", "variant_kind", "split", "source_pair_id")
                if key in row
            }
            if index % 10 == 0 or index == len(rows):
                elapsed = max(time.perf_counter() - started, 0.001)
                rate = index / elapsed
                eta_minutes = (len(rows) - index) / rate / 60
                fraction = index / len(rows) if rows else 1.0
                width = 30
                filled = min(width, int(width * fraction))
                bar = "#" * filled + "-" * (width - filled)
                print(
                    f"\r[{bar}] {index}/{len(rows)} ({fraction:6.2%}) "
                    f"{rate:.2f} examples/s ETA {eta_minutes:.1f}m",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
            yield {
                "pair_id": str(row["pair_id"]),
                "neutral_text": draft,
                "target_text": str(row["completion"]),
                "generated_text": output.split("<end_of_turn>")[0].strip(),
                **metadata,
            }

    count = write_jsonl(output_path, predictions())
    print(file=sys.stderr)
    elapsed = time.perf_counter() - started
    return {
        "architecture": "qwen",
        "predictions": count,
        "elapsed_seconds": elapsed,
        "examples_per_second": count / max(elapsed, 1e-9),
        "output": str(Path(output_path)),
    }


def rewrite_seq2seq(
    config_path: str | Path,
    adapter_dir: str | Path,
    draft: str,
) -> str:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    config = load_yaml(config_path)
    profile = resolve_seq2seq_profile(config)
    options = decoding_options(config)
    torch.manual_seed(int(config.get("seed", 42)))
    device = _device(torch)
    run = Path(adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(run / "tokenizer")
    if bool(config.get("full_finetune", False)):
        model = AutoModelForSeq2SeqLM.from_pretrained(run / "adapter")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            str(config["base_model"]),
            revision=str(config.get("revision", "main")),
        )
        model = PeftModel.from_pretrained(model, run / "adapter")
    model.to(device)
    model.eval()
    encoded = tokenizer(
        format_seq2seq_source(draft, profile),
        return_tensors="pt",
        max_length=int(config.get("max_length", 256)),
        truncation=True,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        candidates = _generate_rewrite_candidates(
            model,
            tokenizer,
            encoded,
            draft,
            int(config.get("max_new_tokens", 64)),
            options,
        )
    return select_rewrite_candidate(draft, candidates, options)


def train_dpo_seq2seq(
    config_path: str | Path,
    data_dir: str | Path,
    adapter_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Direct preference optimization for the seq2seq LoRA adapter.

    The frozen SFT adapter provides reference log-probabilities (precomputed, since the
    policy only drifts after optimization starts), then the LoRA weights are trained to
    widen the chosen-vs-rejected margin relative to that reference. Implemented locally
    because current TRL releases dropped encoder-decoder DPO support.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    config = load_yaml(config_path)
    profile = resolve_seq2seq_profile(config)
    options = dict(config.get("dpo") or {})
    beta = float(options.get("beta", 0.1))
    learning_rate = float(options.get("learning_rate", 5e-6))
    epochs = int(options.get("epochs", 1))
    batch_size = int(options.get("batch_size", 2))
    accumulation = int(options.get("gradient_accumulation_steps", 8))
    max_length = int(config.get("max_length", 256))
    seed = int(config.get("seed", 42))
    if beta <= 0 or learning_rate <= 0 or epochs <= 0 or batch_size <= 0 or accumulation <= 0:
        raise ValueError("DPO settings must be positive")
    torch.manual_seed(seed)
    random.seed(seed)
    device = _device(torch)

    train_rows = list(read_jsonl(Path(data_dir) / "train.jsonl"))
    valid_rows = list(read_jsonl(Path(data_dir) / "valid.jsonl"))
    if not train_rows or not valid_rows:
        raise ValueError("DPO requires non-empty preference train and valid JSONL")

    output = ensure_private_dir(output_dir)
    run = Path(adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(run / "tokenizer")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(config["base_model"]),
        revision=str(config.get("revision", "main")),
    )
    model = PeftModel.from_pretrained(model, run / "adapter", is_trainable=True)
    model.to(device)

    pad_id = int(tokenizer.pad_token_id)

    def encode_batch(rows: Sequence[Mapping[str, Any]]):
        sources, labels = [], []
        for row in rows:
            for key in ("chosen", "rejected"):
                sources.append(format_seq2seq_source(str(row["draft"]), profile))
                labels.append(str(row[key]))
        encoded = tokenizer(
            sources,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        targets = tokenizer(
            text_target=labels,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )["input_ids"]
        targets[targets == pad_id] = -100
        return {key: value.to(device) for key, value in encoded.items()}, targets.to(device)

    def sequence_log_probs(rows: Sequence[Mapping[str, Any]]):
        encoded, targets = encode_batch(rows)
        logits = model(**encoded, labels=targets).logits
        log_probs = torch.log_softmax(logits, dim=-1)
        mask = targets != -100
        gathered = torch.gather(
            log_probs,
            dim=-1,
            index=targets.clamp(min=0).unsqueeze(-1),
        ).squeeze(-1)
        totals = (gathered * mask).sum(dim=-1)
        return totals[0::2], totals[1::2]

    def reference_margins(rows: Sequence[Mapping[str, Any]]):
        margins = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                chosen, rejected = sequence_log_probs(rows[start : start + batch_size])
                margins.extend((chosen - rejected).tolist())
        return margins

    started = time.perf_counter()
    print("precomputing reference margins...", file=sys.stderr, flush=True)
    train_reference = reference_margins(train_rows)
    valid_reference = reference_margins(valid_rows)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)

    def validation_stats() -> tuple[float, float]:
        model.eval()
        margins = []
        with torch.no_grad():
            for start in range(0, len(valid_rows), batch_size):
                chosen, rejected = sequence_log_probs(valid_rows[start : start + batch_size])
                for offset, margin in enumerate((chosen - rejected).tolist()):
                    margins.append(margin - valid_reference[start + offset])
        wins = sum(margin > 0 for margin in margins)
        return sum(margins) / len(margins), wins / len(margins)

    history: list[dict[str, Any]] = []
    best_margin = -math.inf
    order = list(range(len(train_rows)))
    total_batches = math.ceil(len(order) / batch_size) * epochs
    completed_batches = 0
    for epoch in range(epochs):
        model.train()
        random.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        batch_starts = range(0, len(order), batch_size)
        for batch_index, start in enumerate(batch_starts):
            indices = order[start : start + batch_size]
            rows = [train_rows[index] for index in indices]
            chosen, rejected = sequence_log_probs(rows)
            reference = torch.tensor(
                [train_reference[index] for index in indices],
                device=chosen.device,
            )
            loss = -torch.nn.functional.logsigmoid(
                beta * ((chosen - rejected) - reference)
            ).mean()
            (loss / accumulation).backward()
            epoch_loss += float(loss.item())
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(batch_starts):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            completed_batches += 1
            if completed_batches % 10 == 0 or batch_index + 1 == len(batch_starts):
                elapsed = max(time.perf_counter() - started, 0.001)
                rate = completed_batches / elapsed
                eta_minutes = (total_batches - completed_batches) / rate / 60
                fraction = completed_batches / total_batches
                width = 30
                filled = min(width, int(width * fraction))
                bar = "#" * filled + "-" * (width - filled)
                print(
                    f"\r[{bar}] dpo epoch {epoch + 1}/{epochs} "
                    f"batch {batch_index + 1}/{len(batch_starts)} "
                    f"loss {epoch_loss / (batch_index + 1):.4f} "
                    f"{rate:.2f} batch/s ETA {eta_minutes:.1f}m",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
        mean_margin, accuracy = validation_stats()
        history.append(
            {
                "epoch": epoch + 1,
                "training_loss": epoch_loss / max(1, len(batch_starts)),
                "validation_margin": mean_margin,
                "validation_preference_accuracy": accuracy,
            }
        )
        print(
            f"\ndpo epoch {epoch + 1}: margin={mean_margin:.4f} accuracy={accuracy:.4f}",
            file=sys.stderr,
            flush=True,
        )
        if mean_margin > best_margin:
            best_margin = mean_margin
            model.save_pretrained(output / "adapter", safe_serialization=True)
            tokenizer.save_pretrained(output / "tokenizer")
            _chmod_private_tree(output)

    report = {
        "schema_version": 1,
        "task": "seq2seq_dpo",
        "architecture": profile.architecture,
        "base_model": str(config["base_model"]),
        "base_revision": str(config.get("revision", "main")),
        "sft_adapter": str(run),
        "beta": beta,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "preference_pairs": {"train": len(train_rows), "valid": len(valid_rows)},
        "best_validation_margin": best_margin,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "private_local_artifact": True,
    }
    write_json(output / "training-report.json", report)
    _chmod_private_tree(output)
    return report


train_bart = train_seq2seq
predict_bart = predict_seq2seq
rewrite_bart = rewrite_seq2seq


def rewrite_qwen(
    config_path: str | Path,
    adapter_dir: str | Path,
    draft: str,
) -> str:
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    config = load_yaml(config_path)
    model, tokenizer = load(
        _qwen_model_reference(config),
        adapter_path=str(Path(adapter_dir) / "adapter"),
    )
    prompt = _mlx_generation_prompt(tokenizer, draft, f"{REWRITE_INSTRUCTION}{draft}", config)
    output = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=int(config.get("max_new_tokens", 64)),
        sampler=make_sampler(temp=0.0),
        verbose=False,
    )
    return output.split("<end_of_turn>")[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "train-bart",
        "train-seq2seq",
        "train-dpo-bart",
        "train-dpo-seq2seq",
        "predict-bart",
        "predict-seq2seq",
        "predict-qwen",
        "rewrite-bart",
        "rewrite-seq2seq",
        "rewrite-qwen",
    ):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", required=True)
        if command.startswith("rewrite-"):
            subparser.add_argument("--adapter", required=True)
            subparser.add_argument("--draft", required=True)
            continue
        subparser.add_argument("--output", required=True)
        subparser.add_argument("--data", required=True)
        if command.startswith("predict-") or command.startswith("train-dpo-"):
            subparser.add_argument("--adapter", required=True)
    arguments = parser.parse_args()
    if arguments.command in {"rewrite-bart", "rewrite-seq2seq"}:
        print(rewrite_seq2seq(arguments.config, arguments.adapter, arguments.draft))
        return
    if arguments.command == "rewrite-qwen":
        print(rewrite_qwen(arguments.config, arguments.adapter, arguments.draft))
        return
    if arguments.command in {"train-bart", "train-seq2seq"}:
        report = train_seq2seq(arguments.config, arguments.data, arguments.output)
    elif arguments.command in {"train-dpo-bart", "train-dpo-seq2seq"}:
        report = train_dpo_seq2seq(
            arguments.config,
            arguments.data,
            arguments.adapter,
            arguments.output,
        )
    elif arguments.command in {"predict-bart", "predict-seq2seq"}:
        report = predict_seq2seq(
            arguments.config,
            arguments.data,
            arguments.adapter,
            arguments.output,
        )
    else:
        report = predict_qwen(arguments.config, arguments.data, arguments.adapter, arguments.output)
    print(report)


if __name__ == "__main__":
    main()
