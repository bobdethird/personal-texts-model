from pathlib import Path

from imessage_mlx.data.style_targets import build_style_targets, split_style_targets
from imessage_mlx.utils import read_jsonl, sha256_text, write_json, write_jsonl


def _message(
    message_id: str,
    timestamp_ns: int,
    role: str,
    text: str,
    *,
    chat_id: str = "chat",
    reply_to_message_id: str | None = None,
    thread_root_message_id: str | None = None,
) -> dict[str, object]:
    return {
        "message_id": message_id,
        "chat_id": chat_id,
        "timestamp_ns": timestamp_ns,
        "sender_role": role,
        "participant_id": "me" if role == "me" else "other",
        "text": text,
        "reply_to_message_id": reply_to_message_id,
        "thread_root_message_id": thread_root_message_id,
        "thread_originator_part": None,
        "has_attachment": False,
        "is_group": False,
    }


def test_style_targets_merge_outgoing_bursts_and_include_prior_context(tmp_path: Path) -> None:
    messages = tmp_path / "messages.jsonl"
    minute = 60 * 1_000_000_000
    write_jsonl(
        messages,
        [
            _message("incoming", minute, "other", "Can you come later?"),
            _message("part-1", 2 * minute, "me", "yeah i can"),
            _message("part-2", 3 * minute, "me", "after dinner"),
            _message("incoming-2", 4 * minute, "other", "Great"),
            _message("reaction", 5 * minute, "me", "lol"),
            _message("incoming-3", 6 * minute, "other", "Look at this"),
            _message("attachment", 7 * minute, "me", "look\n<|attachment|>"),
        ],
    )

    report = build_style_targets(
        messages,
        tmp_path / "targets.jsonl",
        tmp_path / "report.json",
        merge_gap_minutes=2,
    )
    targets = list(read_jsonl(tmp_path / "targets.jsonl"))

    assert len(targets) == 1
    assert targets[0]["styled_text"] == "yeah i can\nafter dinner"
    assert targets[0]["merged_message_count"] == 2
    assert targets[0]["context"] == [{"role": "other", "text": "Can you come later?"}]
    assert report["counts"]["excluded_low_content"] == 1
    assert report["counts"]["excluded_attachment_or_structural"] == 1
    assert report["pre_split_text_deduplication"] is False


def test_style_targets_keep_adjacent_bubbles_as_separate_meanings_by_default(
    tmp_path: Path,
) -> None:
    messages = tmp_path / "messages.jsonl"
    minute = 60 * 1_000_000_000
    write_jsonl(
        messages,
        [
            _message("incoming", minute, "other", "What should we do tonight?"),
            _message("reply", 2 * minute, "me", "we could get dinner"),
            _message("new-topic", 3 * minute, "me", "also i finished the report"),
        ],
    )

    report = build_style_targets(
        messages,
        tmp_path / "targets.jsonl",
        tmp_path / "report.json",
    )
    targets = list(read_jsonl(tmp_path / "targets.jsonl"))

    assert [row["styled_text"] for row in targets] == [
        "we could get dinner",
        "also i finished the report",
    ]
    assert all(row["merged_message_count"] == 1 for row in targets)
    assert targets[1]["context"][-1] == {
        "role": "me",
        "text": "we could get dinner",
    }
    assert report["target_unit"] == "single_message_bubble"


def test_style_targets_prioritize_exact_reply_target_outside_recent_context(
    tmp_path: Path,
) -> None:
    messages = tmp_path / "messages.jsonl"
    minute = 60 * 1_000_000_000
    write_jsonl(
        messages,
        [
            _message("specific-question", minute, "other", "Can you bring the blue folder?"),
            _message("other-1", 2 * minute, "other", "Separate topic one"),
            _message("other-2", 3 * minute, "other", "Separate topic two"),
            _message(
                "direct-reply",
                4 * minute,
                "me",
                "yeah ill bring it",
                reply_to_message_id="specific-question",
                thread_root_message_id="specific-question",
            ),
        ],
    )

    report = build_style_targets(
        messages,
        tmp_path / "targets.jsonl",
        tmp_path / "report.json",
        context_turns=1,
    )
    target = list(read_jsonl(tmp_path / "targets.jsonl"))[0]

    assert target["context"][0] == {
        "role": "other",
        "text": "Can you bring the blue folder?",
        "relation": "reply_to",
    }
    assert target["context"][-1]["text"] == "Separate topic two"
    assert target["reply_to_message_id"] == "specific-question"
    assert report["counts"]["reply_linked_targets"] == 1
    assert report["counts"]["reply_context_resolved_outside_recent_window"] == 1


def test_style_targets_add_temporal_cross_chat_evidence_and_approved_glossary(
    tmp_path: Path,
) -> None:
    messages = tmp_path / "messages.jsonl"
    glossary = tmp_path / "glossary.json"
    minute = 60 * 1_000_000_000
    write_jsonl(
        messages,
        [
            _message(
                "definition",
                minute,
                "me",
                "Cercor is my AI agent project that calls tools",
                chat_id="project-chat",
            ),
            _message(
                "target",
                3 * minute,
                "me",
                "and my Cercor uses tool calls",
                chat_id="friend-chat",
            ),
            _message(
                "future",
                5 * minute,
                "me",
                "Cercor now uses browser tools",
                chat_id="other-chat",
            ),
        ],
    )
    write_json(
        glossary,
        {
            "schema_version": 1,
            "task": "private_entity_glossary",
            "approval_required": True,
            "entries": [
                {
                    "entry_id": sha256_text("entity-glossary:1:cercor")[:24],
                    "term": "Cercor",
                    "definition": "The author's AI-agent project that invokes tools.",
                    "approved": True,
                    "valid_from_timestamp_ns": minute,
                    "evidence_message_ids": ["definition"],
                }
            ],
        },
    )

    build_style_targets(
        messages,
        tmp_path / "targets.jsonl",
        tmp_path / "report.json",
        context_turns=0,
        glossary_path=glossary,
    )
    target = next(
        row
        for row in read_jsonl(tmp_path / "targets.jsonl")
        if row["styled_text"] == "and my Cercor uses tool calls"
    )

    evidence = target["context_bundle"]["retrieved_evidence"]
    assert [row["message_id"] for row in evidence] == ["definition"]
    assert evidence[0]["chat_id"] == "project-chat"
    assert all(row["message_id"] != "future" for row in evidence)
    assert target["context_bundle"]["glossary_entries"][0]["term"] == "Cercor"


def test_style_target_split_preserves_training_frequency_and_removes_cross_split_duplicates(
    tmp_path: Path,
) -> None:
    targets = []
    texts = [
        "same phrase",
        "same phrase",
        "training only",
        "another training target",
        "last training target",
        "validation unique",
        "Same phrase!",
        "validation second",
        "test unique",
        "same phrase",
    ]
    for index, text in enumerate(texts, start=1):
        targets.append(
            {
                "pair_id": f"pair-{index}",
                "timestamp_ns": index,
                "styled_text": text,
                "context": [],
                "is_group": False,
                "merged_message_count": 1,
            }
        )
    source = tmp_path / "targets.jsonl"
    write_jsonl(source, targets)

    report = split_style_targets(
        source,
        tmp_path / "splits",
        tmp_path / "split-report.json",
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
        guard_days=0,
    )

    train = list(read_jsonl(tmp_path / "splits/train.jsonl"))
    valid = list(read_jsonl(tmp_path / "splits/valid.jsonl"))
    test = list(read_jsonl(tmp_path / "splits/test.jsonl"))
    assert [row["styled_text"] for row in train].count("same phrase") == 2
    assert all(row["styled_text"] != "Same phrase!" for row in valid)
    assert all(row["styled_text"] != "same phrase" for row in test)
    assert report["removed_cross_split_duplicates"] == {
        "train": 0,
        "valid": 1,
        "test": 1,
    }
