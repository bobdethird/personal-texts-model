from __future__ import annotations

from pathlib import Path
from typing import Any

from imessage_mlx.data.sessions import iter_message_sessions
from imessage_mlx.utils import read_jsonl, sha256_text, write_json, write_jsonl

SFT_FORMAT = "imessage-session-v2"


def _session_id(chat_id: str, messages: list[dict[str, Any]]) -> str:
    message_ids = "\0".join(str(message["message_id"]) for message in messages)
    return sha256_text(
        f"{chat_id}\0{messages[0]['timestamp_ns']}\0{messages[-1]['timestamp_ns']}"
        f"\0{message_ids}"
    )[:24]


def _validation_session(session_id: str, validation_fraction: float) -> bool:
    if validation_fraction <= 0:
        return False
    bucket = int(sha256_text(f"validation\0{session_id}")[:16], 16) / 16**16
    return bucket < validation_fraction


def _conversation_message(message: dict[str, Any]) -> dict[str, str]:
    sender_role = str(message["sender_role"])
    if sender_role == "me":
        return {"role": "assistant", "content": str(message["text"])}
    if sender_role != "other":
        raise ValueError(f"Unexpected sender_role {sender_role!r}")

    result = {"role": "user", "content": str(message["text"])}
    if bool(message.get("is_group")):
        # participant_id is already pseudonymized during extraction. Keeping it as
        # metadata lets the renderer distinguish speakers without teaching the model
        # to emit participant labels in its own response.
        result["participant"] = str(message["participant_id"])
    return result


def prepare_sft_dataset(
    messages_path: str | Path,
    train_path: str | Path,
    validation_path: str | Path,
    report_path: str | Path,
    *,
    session_gap_minutes: int = 360,
    validation_fraction: float = 0.05,
) -> dict[str, Any]:
    """Create one multi-turn SFT example per conversation session.

    Every outgoing (`me`) message in the session is an assistant turn and will be
    supervised during training. Incoming messages stay in the transcript as masked
    context. Sessions with no outgoing messages are skipped. Token windows longer
    than the model context are handled later by the training encoder.
    """
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")

    source_messages = list(read_jsonl(messages_path))
    train_records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    session_count = 0
    skipped_without_outgoing = 0
    supervised_outgoing = 0

    for chat_id, session in iter_message_sessions(
        source_messages, session_gap_minutes=session_gap_minutes
    ):
        session_count += 1
        conversation = [_conversation_message(message) for message in session]
        assistant_indexes = [
            index for index, message in enumerate(conversation) if message["role"] == "assistant"
        ]
        if not assistant_indexes:
            skipped_without_outgoing += 1
            continue

        session_id = _session_id(chat_id, session)
        supervised_outgoing += len(assistant_indexes)
        record = {
            "format": SFT_FORMAT,
            "example_id": session_id,
            "session_id": session_id,
            "messages": conversation,
            "supervised_indexes": assistant_indexes,
        }
        if _validation_session(session_id, validation_fraction):
            validation_records.append(record)
        else:
            train_records.append(record)

    write_jsonl(train_path, train_records)
    write_jsonl(validation_path, validation_records)
    report = {
        "format": SFT_FORMAT,
        "input_messages": len(source_messages),
        "session_count": session_count,
        "skipped_sessions_without_outgoing": skipped_without_outgoing,
        "supervised_outgoing_messages": supervised_outgoing,
        "train_examples": len(train_records),
        "validation_examples": len(validation_records),
        "validation_fraction": validation_fraction,
        "session_gap_minutes": session_gap_minutes,
        "loss_policy": "every assistant turn in the session",
    }
    write_json(report_path, report)
    return report
