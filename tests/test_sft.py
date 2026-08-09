from pathlib import Path

from imessage_mlx.data.sft import SFT_FORMAT, prepare_sft_dataset
from imessage_mlx.utils import read_jsonl, write_jsonl


def _message(
    message_id: str,
    *,
    timestamp_ns: int,
    sender_role: str,
    text: str,
    participant_id: str = "person-a",
    chat_id: str = "chat-a",
    is_group: bool = False,
) -> dict[str, object]:
    return {
        "message_id": message_id,
        "chat_id": chat_id,
        "timestamp_ns": timestamp_ns,
        "sender_role": sender_role,
        "participant_id": participant_id,
        "text": text,
        "is_group": is_group,
    }


def test_creates_one_session_example_supervising_every_outgoing_message(
    tmp_path: Path,
) -> None:
    messages = [
        _message("incoming", timestamp_ns=1, sender_role="other", text="Question"),
        _message("mine-1", timestamp_ns=2, sender_role="me", text="First reply"),
        _message("mine-2", timestamp_ns=3, sender_role="me", text="One more thing"),
        _message("incoming-2", timestamp_ns=4, sender_role="other", text="Okay"),
        _message("mine-3", timestamp_ns=5, sender_role="me", text="Last reply"),
    ]
    source = tmp_path / "messages.jsonl"
    train = tmp_path / "sft" / "train.jsonl"
    validation = tmp_path / "sft" / "validation.jsonl"
    report_path = tmp_path / "sft" / "report.json"
    write_jsonl(source, messages)

    report = prepare_sft_dataset(
        source,
        train,
        validation,
        report_path,
        validation_fraction=0,
    )
    records = list(read_jsonl(train))

    assert report["supervised_outgoing_messages"] == 3
    assert report["train_examples"] == 1
    assert list(read_jsonl(validation)) == []
    assert len(records) == 1
    record = records[0]
    assert record["format"] == SFT_FORMAT
    assert record["supervised_indexes"] == [1, 2, 4]
    assert record["messages"] == [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "First reply"},
        {"role": "assistant", "content": "One more thing"},
        {"role": "user", "content": "Okay"},
        {"role": "assistant", "content": "Last reply"},
    ]


def test_skips_sessions_without_outgoing_and_keeps_group_speaker_metadata(
    tmp_path: Path,
) -> None:
    messages = [
        _message(
            "other-only",
            timestamp_ns=1,
            sender_role="other",
            text="Nobody home",
            chat_id="chat-b",
        ),
        _message(
            "other-a",
            timestamp_ns=10,
            sender_role="other",
            text="From A",
            participant_id="person-a",
            is_group=True,
        ),
        _message(
            "other-b",
            timestamp_ns=11,
            sender_role="other",
            text="From B",
            participant_id="person-b",
            is_group=True,
        ),
        _message(
            "mine",
            timestamp_ns=12,
            sender_role="me",
            text="My answer",
            participant_id="me",
            is_group=True,
        ),
    ]
    source = tmp_path / "messages.jsonl"
    write_jsonl(source, messages)

    report = prepare_sft_dataset(
        source,
        tmp_path / "train.jsonl",
        tmp_path / "validation.jsonl",
        tmp_path / "report.json",
        validation_fraction=0,
    )
    records = list(read_jsonl(tmp_path / "train.jsonl"))

    assert report["skipped_sessions_without_outgoing"] == 1
    assert report["train_examples"] == 1
    assert records[0]["messages"] == [
        {
            "role": "user",
            "content": "From A",
            "participant": "person-a",
        },
        {
            "role": "user",
            "content": "From B",
            "participant": "person-b",
        },
        {"role": "assistant", "content": "My answer"},
    ]
    assert records[0]["supervised_indexes"] == [2]


def test_strips_placeholders_and_drops_empty_messages(tmp_path: Path) -> None:
    messages = [
        _message(
            "with-attachment",
            timestamp_ns=1,
            sender_role="other",
            text="look at this\n<|attachment|>",
        ),
        _message(
            "attachment-only",
            timestamp_ns=2,
            sender_role="me",
            text="<|attachment|>",
        ),
        _message("real-reply", timestamp_ns=3, sender_role="me", text="nice pic"),
        _message(
            "long-participant",
            timestamp_ns=4,
            sender_role="other",
            text="who took it? <|url|> here",
            participant_id="abcdef0123456789deadbeef",
            is_group=True,
        ),
        _message("final", timestamp_ns=5, sender_role="me", text="me lol"),
    ]
    source = tmp_path / "messages.jsonl"
    write_jsonl(source, messages)

    report = prepare_sft_dataset(
        source,
        tmp_path / "train.jsonl",
        tmp_path / "validation.jsonl",
        tmp_path / "report.json",
        validation_fraction=0,
    )
    records = list(read_jsonl(tmp_path / "train.jsonl"))

    assert report["dropped_empty_messages"] == 1
    assert len(records) == 1
    contents = [message["content"] for message in records[0]["messages"]]
    assert contents == ["look at this", "nice pic", "who took it? here", "me lol"]
    assert all("<|" not in content for content in contents)
    assert records[0]["messages"][2]["participant"] == "abcdef01"
    assert records[0]["supervised_indexes"] == [1, 3]
