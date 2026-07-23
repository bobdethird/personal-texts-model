import json
from pathlib import Path

from imessage_mlx.data.sessions import build_sessions
from imessage_mlx.utils import read_jsonl, write_jsonl


def _message(
    message_id: str,
    *,
    chat_id: str = "chat-a",
    timestamp_ns: int,
    sender_role: str,
    participant_id: str,
    text: str,
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


def test_builds_role_conditioned_sessions(tmp_path: Path) -> None:
    minute = 60 * 1_000_000_000
    messages = [
        _message(
            "m1",
            timestamp_ns=0,
            sender_role="other",
            participant_id="person-a",
            text="First",
        ),
        _message(
            "m2",
            timestamp_ns=minute,
            sender_role="other",
            participant_id="person-a",
            text="Second",
        ),
        _message(
            "m3",
            timestamp_ns=3 * minute,
            sender_role="me",
            participant_id="me",
            text="Reply",
        ),
        _message(
            "m4",
            timestamp_ns=400 * minute,
            sender_role="other",
            participant_id="person-a",
            text="Later",
        ),
        _message(
            "m5",
            chat_id="chat-b",
            timestamp_ns=0,
            sender_role="other",
            participant_id="person-b",
            text="Only turn",
            is_group=True,
        ),
    ]
    source = tmp_path / "messages.jsonl"
    output = tmp_path / "sessions.jsonl"
    report_path = tmp_path / "report.json"
    write_jsonl(source, messages)

    report = build_sessions(source, output, report_path)
    sessions = list(read_jsonl(output))

    assert report == {
        "input_messages": 5,
        "chat_count": 2,
        "session_count": 1,
        "discarded_single_turn_sessions": 2,
        "session_gap_minutes": 360,
        "merge_gap_minutes": 2,
    }
    assert len(sessions) == 1
    assert sessions[0]["turn_count"] == 2
    assert sessions[0]["text"] == (
        "<|bos|><|conversation|>\n"
        "<|other|>First\nSecond<|turn_end|>\n"
        "<|me|>Reply<|turn_end|>\n"
        "<|eos|>"
    )
    assert "participant_id" not in json.dumps(sessions)
    assert json.loads(report_path.read_text()) == report
