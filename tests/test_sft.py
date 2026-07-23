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


def test_creates_one_final_assistant_target_for_every_outgoing_message(
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

    assert report["target_outgoing_messages"] == 3
    assert report["train_examples"] == 3
    assert list(read_jsonl(validation)) == []
    assert [record["target_message_id"] for record in records] == [
        "mine-1",
        "mine-2",
        "mine-3",
    ]
    assert all(record["format"] == SFT_FORMAT for record in records)
    assert all(record["target_index"] == len(record["messages"]) - 1 for record in records)
    assert all(record["messages"][-1]["role"] == "assistant" for record in records)
    assert records[1]["messages"] == [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "First reply"},
        {"role": "assistant", "content": "One more thing"},
    ]


def test_limits_history_and_preserves_group_speaker_metadata(tmp_path: Path) -> None:
    messages = [
        _message(
            "other-a",
            timestamp_ns=1,
            sender_role="other",
            text="From A",
            participant_id="person-a",
            is_group=True,
        ),
        _message(
            "other-b",
            timestamp_ns=2,
            sender_role="other",
            text="From B",
            participant_id="person-b",
            is_group=True,
        ),
        _message(
            "mine",
            timestamp_ns=3,
            sender_role="me",
            text="My answer",
            participant_id="me",
            is_group=True,
        ),
    ]
    source = tmp_path / "messages.jsonl"
    write_jsonl(source, messages)

    prepare_sft_dataset(
        source,
        tmp_path / "train.jsonl",
        tmp_path / "validation.jsonl",
        tmp_path / "report.json",
        validation_fraction=0,
        max_history_messages=1,
    )
    record = next(read_jsonl(tmp_path / "train.jsonl"))

    assert record["messages"] == [
        {
            "role": "user",
            "content": "From B",
            "participant": "person-b",
        },
        {"role": "assistant", "content": "My answer"},
    ]
