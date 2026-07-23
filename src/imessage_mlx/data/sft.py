from __future__ import annotations

from pathlib import Path
from typing import Any

from imessage_mlx.data.sessions import iter_message_sessions
from imessage_mlx.utils import read_jsonl, sha256_text, write_json, write_jsonl

SFT_FORMAT = "imessage-next-message-v1"


def _session_id(chat_id: str, messages: list[dict[str, Any]]) -> str:
    message_ids = "\0".join(str(message["message_id"]) for message in messages)
    return sha256_text(
        f"{chat_id}\0{messages[0]['timestamp_ns']}\0{messages[-1]['timestamp_ns']}\0{message_ids}"
    )[:24]


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
    max_history_messages: int | None = None,
) -> dict[str, Any]:
    """Create one next-message SFT example for every outgoing message.

    The final message in every example is the sole loss target. Earlier outgoing
    messages remain useful context but are masked by the training tokenizer, and
    incoming messages are never selected as targets.
    """
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if max_history_messages is not None and max_history_messages < 0:
        raise ValueError("max_history_messages cannot be negative")

    source_messages = list(read_jsonl(messages_path))
    session_examples: list[tuple[str, list[dict[str, Any]]]] = []
    session_count = 0
    target_count = 0

    for chat_id, session in iter_message_sessions(
        source_messages, session_gap_minutes=session_gap_minutes
    ):
        session_count += 1
        session_id = _session_id(chat_id, session)
        records: list[dict[str, Any]] = []

        for target_position, message in enumerate(session):
            if message["sender_role"] != "me":
                continue

            history_start = 0
            if max_history_messages is not None:
                history_start = max(0, target_position - max_history_messages)
            selected = session[history_start : target_position + 1]
            conversation = [_conversation_message(item) for item in selected]
            target_index = len(conversation) - 1
            target_count += 1
            records.append(
                {
                    "format": SFT_FORMAT,
                    "example_id": sha256_text(
                        f"{session_id}\0{message['message_id']}\0{target_position}"
                    )[:24],
                    "session_id": session_id,
                    "target_message_id": str(message["message_id"]),
                    "target_index": target_index,
                    "messages": conversation,
                }
            )
        if records:
            session_examples.append((session_id, records))

    validation_session_ids: set[str] = set()
    if validation_fraction > 0 and len(session_examples) > 1:
        validation_session_count = min(
            len(session_examples) - 1,
            max(1, round(len(session_examples) * validation_fraction)),
        )
        ranked_sessions = sorted(
            session_examples,
            key=lambda item: sha256_text(f"validation\0{item[0]}"),
        )
        validation_session_ids = {
            session_id for session_id, _records in ranked_sessions[:validation_session_count]
        }

    train_records = [
        record
        for session_id, records in session_examples
        if session_id not in validation_session_ids
        for record in records
    ]
    validation_records = [
        record
        for session_id, records in session_examples
        if session_id in validation_session_ids
        for record in records
    ]

    write_jsonl(train_path, train_records)
    write_jsonl(validation_path, validation_records)
    report = {
        "format": SFT_FORMAT,
        "input_messages": len(source_messages),
        "session_count": session_count,
        "target_outgoing_messages": target_count,
        "train_examples": len(train_records),
        "validation_examples": len(validation_records),
        "validation_sessions": len(validation_session_ids),
        "validation_fraction": validation_fraction,
        "session_gap_minutes": session_gap_minutes,
        "max_history_messages": max_history_messages,
        "loss_policy": "final assistant message only",
    }
    write_json(report_path, report)
    return report
