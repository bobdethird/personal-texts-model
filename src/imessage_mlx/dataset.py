from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

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
