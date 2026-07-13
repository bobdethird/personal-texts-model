from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from imessage_mlx.data.rewrite import format_rewrite_prompt
from imessage_mlx.tokenizer.train import load_tokenizer
from imessage_mlx.utils import ensure_private_dir, read_jsonl, write_json


def encode_split(
    jsonl_path: str | Path,
    tokenizer_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    tokenizer = load_tokenizer(tokenizer_path)
    token_ids: list[int] = []
    session_count = 0
    for record in read_jsonl(jsonl_path):
        encoded = tokenizer.encode(str(record["text"]), add_special_tokens=False).ids
        token_ids.extend(encoded)
        session_count += 1
    array = np.asarray(token_ids, dtype=np.uint32)
    destination = Path(output_path)
    ensure_private_dir(destination.parent)
    np.save(destination, array, allow_pickle=False)
    destination.chmod(0o600)
    return {
        "sessions": session_count,
        "tokens": int(array.size),
        "dtype": str(array.dtype),
        "sha256_token_sum": int(array.astype(np.uint64).sum() % (2**63 - 1)),
    }


def encode_all_splits(
    splits_dir: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output = ensure_private_dir(output_dir)
    report = {
        name: encode_split(
            Path(splits_dir) / f"{name}.jsonl",
            tokenizer_path,
            output / f"{name}.npy",
        )
        for name in ("train", "validation", "test")
    }
    write_json(output / "token-counts.json", report)
    return report


def encode_rewrite_split(
    jsonl_path: str | Path,
    tokenizer_path: str | Path,
    output_path: str | Path,
    mask_output_path: str | Path,
    *,
    context_length: int,
    skip_oversized: bool = False,
) -> dict[str, Any]:
    tokenizer = load_tokenizer(tokenizer_path)
    pad_id = tokenizer.token_to_id("<|pad|>")
    if pad_id is None:
        raise ValueError("Tokenizer is missing required <|pad|> token")
    for token in (
        "<|bos|>",
        "<|eos|>",
        "<|rewrite|>",
        "<|draft|>",
        "<|me|>",
        "<|turn_end|>",
    ):
        if tokenizer.token_to_id(token) is None:
            raise ValueError(f"Tokenizer is missing required {token} token")

    block_size = context_length + 1
    token_blocks: list[list[int]] = []
    mask_blocks: list[list[int]] = []
    current_tokens: list[int] = []
    current_mask: list[int] = []
    pair_count = 0
    skipped_oversized_pairs = 0
    source_tokens = 0
    supervised_tokens = 0

    def flush_block() -> None:
        if not current_tokens:
            return
        padding = block_size - len(current_tokens)
        token_blocks.append([*current_tokens, *([pad_id] * padding)])
        mask_blocks.append([*current_mask, *([0] * padding)])
        current_tokens.clear()
        current_mask.clear()

    for record in read_jsonl(jsonl_path):
        pair_id = str(record.get("pair_id", f"record-{pair_count}"))
        neutral_text = record.get("neutral_text")
        styled_text = record.get("styled_text")
        if not isinstance(neutral_text, str) or not isinstance(styled_text, str):
            raise ValueError(f"Rewrite pair {pair_id!r} is missing validated text fields")

        prompt_ids = tokenizer.encode(
            format_rewrite_prompt(neutral_text), add_special_tokens=False
        ).ids
        target_ids = tokenizer.encode(f"{styled_text}<|turn_end|>", add_special_tokens=False).ids
        suffix_ids = tokenizer.encode("\n<|eos|>", add_special_tokens=False).ids
        pair_tokens = [*prompt_ids, *target_ids, *suffix_ids]
        pair_mask = [*([0] * len(prompt_ids)), *([1] * len(target_ids)), *([0] * len(suffix_ids))]
        if len(pair_tokens) > block_size:
            if skip_oversized:
                skipped_oversized_pairs += 1
                continue
            raise ValueError(
                f"Rewrite pair {pair_id!r} requires {len(pair_tokens)} tokens, "
                f"exceeding block size {block_size}"
            )
        if current_tokens and len(current_tokens) + len(pair_tokens) > block_size:
            flush_block()
        current_tokens.extend(pair_tokens)
        current_mask.extend(pair_mask)
        pair_count += 1
        source_tokens += len(pair_tokens)
        supervised_tokens += len(target_ids)
    flush_block()

    if not token_blocks:
        raise ValueError("Rewrite split contains no encodable pairs")
    token_array = np.asarray(token_blocks, dtype=np.uint32).reshape(-1)
    mask_array = np.asarray(mask_blocks, dtype=np.uint8).reshape(-1)
    destination = Path(output_path)
    mask_destination = Path(mask_output_path)
    ensure_private_dir(destination.parent)
    ensure_private_dir(mask_destination.parent)
    np.save(destination, token_array, allow_pickle=False)
    np.save(mask_destination, mask_array, allow_pickle=False)
    destination.chmod(0o600)
    mask_destination.chmod(0o600)
    return {
        "pairs": pair_count,
        "skipped_oversized_pairs": skipped_oversized_pairs,
        "windows": len(token_blocks),
        "tokens": source_tokens,
        "encoded_tokens": int(token_array.size),
        "supervised_tokens": supervised_tokens,
        "padding_tokens": int(token_array.size - source_tokens),
        "context_length": context_length,
    }


def encode_all_rewrite_splits(
    splits_dir: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    *,
    context_length: int,
) -> dict[str, Any]:
    output = ensure_private_dir(output_dir)
    report = {
        name: encode_rewrite_split(
            Path(splits_dir) / f"{name}.jsonl",
            tokenizer_path,
            output / f"{name}.npy",
            output / f"{name}-loss-mask.npy",
            context_length=context_length,
            skip_oversized=True,
        )
        for name in ("train", "validation", "test")
    }
    write_json(output / "token-counts.json", report)
    return report


def load_tokens(path: str | Path, *, mmap: bool = True) -> np.ndarray:
    return np.load(Path(path), mmap_mode="r" if mmap else None, allow_pickle=False)


def window_count(tokens: np.ndarray, context_length: int) -> int:
    return int(tokens.size // (context_length + 1))


def batch_indices(
    count: int,
    batch_size: int,
    *,
    seed: int,
    epoch: int,
    start_batch: int = 0,
) -> Iterator[np.ndarray]:
    if count <= 0:
        return
    generator = np.random.default_rng(seed + epoch)
    permutation = generator.permutation(count)
    for batch_number, start in enumerate(range(0, count, batch_size)):
        if batch_number < start_batch:
            continue
        yield permutation[start : start + batch_size]


def materialize_batch(
    tokens: np.ndarray, indices: np.ndarray, context_length: int
) -> tuple[np.ndarray, np.ndarray]:
    size = context_length + 1
    windows = np.stack(
        [np.asarray(tokens[int(index) * size : (int(index) + 1) * size]) for index in indices]
    ).astype(np.int32, copy=False)
    return windows[:, :-1], windows[:, 1:]


def materialize_masked_batch(
    tokens: np.ndarray,
    loss_mask: np.ndarray,
    indices: np.ndarray,
    context_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if tokens.size != loss_mask.size:
        raise ValueError("Token and loss-mask arrays must have identical lengths")
    inputs, targets = materialize_batch(tokens, indices, context_length)
    size = context_length + 1
    mask_windows = np.stack(
        [np.asarray(loss_mask[int(index) * size : (int(index) + 1) * size]) for index in indices]
    ).astype(np.float32, copy=False)
    return inputs, targets, mask_windows[:, 1:]


def estimate_parameter_count(config: dict[str, Any], vocab_size: int) -> int:
    hidden = int(config["hidden_size"])
    layers = int(config["num_layers"])
    intermediate = int(config["intermediate_size"])
    tied = bool(config.get("tie_embeddings", True))
    embeddings = vocab_size * hidden
    attention = layers * (4 * hidden * hidden)
    feed_forward = layers * (3 * hidden * intermediate)
    norms = layers * (2 * hidden) + hidden
    output = 0 if tied else vocab_size * hidden
    return embeddings + attention + feed_forward + norms + output


def select_model(
    train_tokens: int,
    vocab_size: int,
    one_million_config: dict[str, Any],
    seven_million_config: dict[str, Any],
) -> dict[str, Any]:
    candidates = []
    for config in (one_million_config, seven_million_config):
        parameters = estimate_parameter_count(config, vocab_size)
        candidates.append(
            {
                "name": config["name"],
                "parameters": parameters,
                "tokens_per_parameter": train_tokens / parameters if parameters else math.inf,
                "eligible": train_tokens / parameters >= 10 if parameters else False,
            }
        )
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    selected = max(eligible, key=lambda value: value["parameters"], default=candidates[0])
    return {
        "train_tokens": train_tokens,
        "vocab_size": vocab_size,
        "minimum_tokens_per_parameter": 10,
        "candidates": candidates,
        "selected": selected["name"],
        "memorization_prone_experiment": not bool(eligible),
        "enough_tokens_to_train": train_tokens >= 1_000_000,
    }


def select_rewrite_model(
    train_tokens: int,
    train_pairs: int,
    vocab_size: int,
    candidate_configs: list[dict[str, Any]],
    *,
    minimum_train_tokens: int = 100_000,
    minimum_train_pairs: int = 10_000,
    minimum_tokens_per_parameter: float = 1.5,
) -> dict[str, Any]:
    if not candidate_configs:
        raise ValueError("At least one rewrite model candidate is required")
    token_minimum_met = train_tokens >= minimum_train_tokens
    pair_minimum_met = train_pairs >= minimum_train_pairs
    candidates = []
    for config in candidate_configs:
        parameters = estimate_parameter_count(config, vocab_size)
        ratio = train_tokens / parameters if parameters else math.inf
        ratio_minimum_met = ratio >= minimum_tokens_per_parameter
        rejection_reasons = []
        if not token_minimum_met:
            rejection_reasons.append("insufficient_supervised_target_tokens")
        if not pair_minimum_met:
            rejection_reasons.append("insufficient_training_pairs")
        if not ratio_minimum_met:
            rejection_reasons.append("insufficient_tokens_per_parameter")
        candidates.append(
            {
                "name": config["name"],
                "parameters": parameters,
                "tokens_per_parameter": ratio,
                "supervised_target_tokens_per_parameter": ratio,
                "token_minimum_met": token_minimum_met,
                "pair_minimum_met": pair_minimum_met,
                "ratio_minimum_met": ratio_minimum_met,
                "eligible": not rejection_reasons,
                "rejection_reasons": rejection_reasons,
            }
        )
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    selected = max(eligible, key=lambda value: value["parameters"], default=None)
    return {
        "schema_version": 1,
        "task": "rewrite",
        "selection_scope": "training_split",
        "train_tokens": train_tokens,
        "train_supervised_target_tokens": train_tokens,
        "train_pairs": train_pairs,
        "vocab_size": vocab_size,
        "minimum_train_tokens": minimum_train_tokens,
        "minimum_train_pairs": minimum_train_pairs,
        "minimum_tokens_per_parameter": minimum_tokens_per_parameter,
        "candidates": candidates,
        "selected": selected["name"] if selected else None,
        "eligible_to_train": bool(selected),
        "experimental_from_scratch": True,
    }
