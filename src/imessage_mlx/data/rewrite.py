from __future__ import annotations

import re
from collections.abc import Iterator
from itertools import chain
from pathlib import Path
from typing import Any

from imessage_mlx.data.normalize import normalize_text
from imessage_mlx.data.split import split_sessions
from imessage_mlx.utils import read_jsonl, sha256_text, write_json, write_jsonl

STRUCTURAL_TOKEN_RE = re.compile(r"<\|[^|\n]+?\|>")


def format_rewrite_prompt(neutral_text: str) -> str:
    return "\n".join(
        (
            "<|bos|><|rewrite|>",
            f"<|draft|>{neutral_text}<|turn_end|>",
            "<|me|>",
        )
    )


def format_rewrite_example(neutral_text: str, styled_text: str) -> str:
    return f"{format_rewrite_prompt(neutral_text)}{styled_text}<|turn_end|>\n<|eos|>"


def _validated_text(value: Any, field: str, pair_id: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Rewrite pair {pair_id!r} field {field!r} must be a string")
    normalized = normalize_text(value)
    if not normalized:
        raise ValueError(f"Rewrite pair {pair_id!r} field {field!r} cannot be empty")
    if STRUCTURAL_TOKEN_RE.search(normalized):
        raise ValueError(
            f"Rewrite pair {pair_id!r} field {field!r} contains a reserved structural token"
        )
    return normalized


def validate_rewrite_pair(value: dict[str, Any]) -> dict[str, Any]:
    pair_id = value.get("pair_id")
    if not isinstance(pair_id, str) or not pair_id.strip():
        raise ValueError("Rewrite pair field 'pair_id' must be a non-empty string")
    pair_id = pair_id.strip()

    timestamp_ns = value.get("timestamp_ns")
    if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int) or timestamp_ns < 0:
        raise ValueError(
            f"Rewrite pair {pair_id!r} field 'timestamp_ns' must be a nonnegative integer"
        )

    neutral_text = _validated_text(value.get("neutral_text"), "neutral_text", pair_id)
    styled_text = _validated_text(value.get("styled_text"), "styled_text", pair_id)
    return {
        "pair_id": pair_id,
        "timestamp_ns": timestamp_ns,
        "neutral_text": neutral_text,
        "styled_text": styled_text,
    }


def build_rewrite_pairs(
    pairs_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    input_count = 0
    output_count = 0
    seen_ids: set[str] = set()
    source = iter(read_jsonl(pairs_path))
    try:
        first_pair = next(source)
    except StopIteration as error:
        raise ValueError("Rewrite pair input contains no records") from error

    def records() -> Iterator[dict[str, Any]]:
        nonlocal input_count, output_count
        for raw_pair in chain((first_pair,), source):
            input_count += 1
            pair = validate_rewrite_pair(raw_pair)
            pair_id = str(pair["pair_id"])
            if pair_id in seen_ids:
                raise ValueError(f"Duplicate rewrite pair_id {pair_id!r}")
            seen_ids.add(pair_id)
            timestamp_ns = int(pair["timestamp_ns"])
            neutral_text = str(pair["neutral_text"])
            styled_text = str(pair["styled_text"])
            text = format_rewrite_example(neutral_text, styled_text)
            output_count += 1
            yield {
                **pair,
                "session_id": sha256_text(pair_id)[:24],
                "start_ns": timestamp_ns,
                "end_ns": timestamp_ns,
                "turn_count": 2,
                "is_group": False,
                "task": "rewrite",
                "text": text,
            }

    write_jsonl(output_path, records())
    report = {
        "input_pairs": input_count,
        "output_pairs": output_count,
        "task": "rewrite",
    }
    write_json(report_path, report)
    return report


def prepare_rewrite_dataset(
    pairs_path: str | Path,
    processed_path: str | Path,
    splits_dir: str | Path,
    preparation_report_path: str | Path,
    split_report_path: str | Path,
    *,
    train_fraction: float = 0.90,
    validation_fraction: float = 0.05,
    test_fraction: float = 0.05,
    guard_days: int = 7,
) -> dict[str, Any]:
    preparation = build_rewrite_pairs(pairs_path, processed_path, preparation_report_path)
    split = split_sessions(
        processed_path,
        splits_dir,
        split_report_path,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        guard_days=guard_days,
    )
    return {"preparation": preparation, "split": split}
