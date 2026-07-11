from pathlib import Path

import pytest

from imessage_mlx.data.rewrite import (
    build_rewrite_pairs,
    format_rewrite_example,
    prepare_rewrite_dataset,
    validate_rewrite_pair,
)
from imessage_mlx.utils import read_jsonl


def test_rewrite_pairs_are_validated_and_split_chronologically(tmp_path: Path) -> None:
    source = Path(__file__).parent / "fixtures/synthetic_rewrites.jsonl"
    report = prepare_rewrite_dataset(
        source,
        tmp_path / "processed.jsonl",
        tmp_path / "splits",
        tmp_path / "preparation.json",
        tmp_path / "split-report.json",
        guard_days=0,
    )

    assert report["preparation"]["output_pairs"] == 8
    splits = {
        name: list(read_jsonl(tmp_path / f"splits/{name}.jsonl"))
        for name in ("train", "validation", "test")
    }
    assert all(splits.values())
    assert max(row["end_ns"] for row in splits["train"]) < min(
        row["end_ns"] for row in splits["validation"]
    )
    assert max(row["end_ns"] for row in splits["validation"]) < min(
        row["end_ns"] for row in splits["test"]
    )
    assert all(row["task"] == "rewrite" for values in splits.values() for row in values)
    first = splits["train"][0]
    assert first["text"] == format_rewrite_example(first["neutral_text"], first["styled_text"])


@pytest.mark.parametrize(
    "pair",
    [
        {
            "pair_id": "empty",
            "timestamp_ns": 1,
            "neutral_text": " ",
            "styled_text": "ok",
        },
        {
            "pair_id": "token",
            "timestamp_ns": 1,
            "neutral_text": "hello <|me|>",
            "styled_text": "hey",
        },
        {
            "pair_id": "timestamp",
            "timestamp_ns": -1,
            "neutral_text": "hello",
            "styled_text": "hey",
        },
    ],
)
def test_rewrite_pair_validation_rejects_unsafe_records(pair: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        validate_rewrite_pair(pair)


def test_empty_pair_input_does_not_replace_existing_processed_data(tmp_path: Path) -> None:
    source = tmp_path / "empty.jsonl"
    source.write_text("", encoding="utf-8")
    destination = tmp_path / "processed.jsonl"
    destination.write_text("sentinel\n", encoding="utf-8")

    with pytest.raises(ValueError, match="contains no records"):
        build_rewrite_pairs(source, destination, tmp_path / "report.json")

    assert destination.read_text(encoding="utf-8") == "sentinel\n"
