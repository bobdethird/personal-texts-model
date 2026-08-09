from pathlib import Path

import pytest

from imessage_mlx.data.pairs import (
    PAIR_FORMAT,
    build_context_query,
    derive_pairs,
    prepare_pair_dataset,
)
from imessage_mlx.data.sft import SFT_FORMAT
from imessage_mlx.utils import read_jsonl, write_jsonl


def _session(session_id: str) -> dict[str, object]:
    return {
        "format": SFT_FORMAT,
        "example_id": session_id,
        "session_id": session_id,
        "supervised_indexes": [1, 2, 4],
        "messages": [
            {
                "role": "user",
                "content": "free later?",
                "participant": "person-a",
            },
            {"role": "assistant", "content": "after 7"},
            {"role": "assistant", "content": "want food?"},
            {"role": "user", "content": "yes"},
            {"role": "assistant", "content": "cool"},
        ],
    }


def test_derives_consecutive_replies_with_full_prefix_context() -> None:
    pairs, skipped = derive_pairs([_session("session-a")], split="train")

    assert skipped == 0
    assert [pair["format"] for pair in pairs] == [PAIR_FORMAT] * 3
    assert [pair["query"] for pair in pairs] == ["free later?", "free later?", "yes"]
    assert [pair["reply"] for pair in pairs] == ["after 7", "want food?", "cool"]
    assert pairs[1]["messages"][-1] == {"role": "assistant", "content": "after 7"}
    assert pairs[0]["messages"][0]["participant"] == "person-a"
    assert all(pair["session_id"] == "session-a" for pair in pairs)
    assert all(pair["split"] == "train" for pair in pairs)
    # The retrieval query carries the recent multi-turn context, not just the last message.
    assert pairs[0]["context_query"] == "Them: free later?"
    assert pairs[2]["context_query"] == (
        "Them: free later?\nMe: after 7\nMe: want food?\nThem: yes"
    )
    # Structured context turns back the retrieval demonstrations (metadata stripped).
    assert pairs[2]["context_messages"] == [
        {"role": "user", "content": "free later?"},
        {"role": "assistant", "content": "after 7"},
        {"role": "assistant", "content": "want food?"},
        {"role": "user", "content": "yes"},
    ]


def test_context_query_keeps_recent_turns_from_both_sides() -> None:
    messages = [
        {"role": "assistant", "content": "notion good for medical stuff"},
        {"role": "assistant", "content": "like the dentist"},
        {"role": "user", "content": "wait wdym"},
        {"role": "assistant", "content": "like my retainer got loose"},
    ]

    query = build_context_query(messages)

    assert query == (
        "Me: notion good for medical stuff\n"
        "Me: like the dentist\n"
        "Them: wait wdym\n"
        "Me: like my retainer got loose"
    )


def test_context_query_truncates_to_recent_turns() -> None:
    messages = [{"role": "user", "content": f"turn {index}"} for index in range(10)]

    query = build_context_query(messages, max_turns=3)

    assert query == "Them: turn 7\nThem: turn 8\nThem: turn 9"


def test_skips_targets_without_a_nonempty_incoming_message() -> None:
    record = {
        "format": SFT_FORMAT,
        "session_id": "session-a",
        "supervised_indexes": [0, 2],
        "messages": [
            {"role": "assistant", "content": "started it"},
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "still no query"},
        ],
    }

    pairs, skipped = derive_pairs([record], split="train")

    assert pairs == []
    assert skipped == 2


def test_prepare_pairs_inherits_whole_session_split(tmp_path: Path) -> None:
    train_sessions = tmp_path / "sft" / "train.jsonl"
    validation_sessions = tmp_path / "sft" / "validation.jsonl"
    write_jsonl(train_sessions, [_session("train-session")])
    write_jsonl(validation_sessions, [_session("validation-session")])

    report = prepare_pair_dataset(
        train_sessions,
        validation_sessions,
        tmp_path / "pairs" / "train.jsonl",
        tmp_path / "pairs" / "validation.jsonl",
        tmp_path / "pairs" / "report.json",
    )
    train_pairs = list(read_jsonl(tmp_path / "pairs" / "train.jsonl"))
    validation_pairs = list(read_jsonl(tmp_path / "pairs" / "validation.jsonl"))

    assert report["train_pairs"] == 3
    assert report["validation_pairs"] == 3
    assert {pair["session_id"] for pair in train_pairs} == {"train-session"}
    assert {pair["split"] for pair in train_pairs} == {"train"}
    assert {pair["session_id"] for pair in validation_pairs} == {"validation-session"}
    assert {pair["split"] for pair in validation_pairs} == {"validation"}


def test_prepare_pairs_rejects_session_split_leakage(tmp_path: Path) -> None:
    train_sessions = tmp_path / "train.jsonl"
    validation_sessions = tmp_path / "validation.jsonl"
    write_jsonl(train_sessions, [_session("overlap")])
    write_jsonl(validation_sessions, [_session("overlap")])

    with pytest.raises(ValueError, match="overlapping sessions"):
        prepare_pair_dataset(
            train_sessions,
            validation_sessions,
            tmp_path / "pairs" / "train.jsonl",
            tmp_path / "pairs" / "validation.jsonl",
            tmp_path / "pairs" / "report.json",
        )
