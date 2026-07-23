from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from imessage_mlx.utils import read_jsonl, sha256_text, write_json, write_jsonl


def _serialize_turns(turns: list[dict[str, Any]]) -> str:
    lines = ["<|bos|><|conversation|>"]
    for turn in turns:
        role = "<|me|>" if turn["sender_role"] == "me" else "<|other|>"
        lines.append(f"{role}{turn['text']}<|turn_end|>")
    lines.append("<|eos|>")
    return "\n".join(lines)


def iter_message_sessions(
    messages: Iterable[dict[str, Any]],
    *,
    session_gap_minutes: int = 360,
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """Yield chronological, unmerged message sessions grouped by chat."""
    if session_gap_minutes < 1:
        raise ValueError("session_gap_minutes must be at least 1")

    chats: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for message in messages:
        chats[str(message["chat_id"])].append(message)

    session_gap_ns = session_gap_minutes * 60 * 1_000_000_000
    for chat_id in sorted(chats):
        ordered = sorted(
            chats[chat_id], key=lambda row: (int(row["timestamp_ns"]), str(row["message_id"]))
        )
        current: list[dict[str, Any]] = []
        for message in ordered:
            if (
                current
                and int(message["timestamp_ns"]) - int(current[-1]["timestamp_ns"]) > session_gap_ns
            ):
                yield chat_id, current
                current = []
            current.append(message)
        if current:
            yield chat_id, current


def build_sessions(
    messages_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    *,
    session_gap_minutes: int = 360,
    merge_gap_minutes: int = 2,
) -> dict[str, Any]:
    messages = list(read_jsonl(messages_path))
    merge_gap_ns = merge_gap_minutes * 60 * 1_000_000_000
    discarded_short = 0
    output_count = 0

    def session_records() -> Iterator[dict[str, Any]]:
        nonlocal discarded_short, output_count
        for chat_id, bucket in iter_message_sessions(
            messages, session_gap_minutes=session_gap_minutes
        ):
            turns: list[dict[str, Any]] = []
            for message in bucket:
                if (
                    turns
                    and turns[-1]["participant_id"] == message["participant_id"]
                    and int(message["timestamp_ns"]) - int(turns[-1]["timestamp_ns"])
                    <= merge_gap_ns
                ):
                    turns[-1]["text"] += "\n" + message["text"]
                    turns[-1]["timestamp_ns"] = message["timestamp_ns"]
                else:
                    turns.append(dict(message))
            if len(turns) < 2:
                discarded_short += 1
                continue
            text = _serialize_turns(turns)
            output_count += 1
            yield {
                "session_id": sha256_text(
                    f"{chat_id}\0{bucket[0]['timestamp_ns']}\0{bucket[-1]['timestamp_ns']}\0{text}"
                )[:24],
                "start_ns": int(bucket[0]["timestamp_ns"]),
                "end_ns": int(bucket[-1]["timestamp_ns"]),
                "turn_count": len(turns),
                "is_group": any(bool(message["is_group"]) for message in bucket),
                "text": text,
            }

    write_jsonl(output_path, session_records())
    report = {
        "input_messages": len(messages),
        "chat_count": len({str(message["chat_id"]) for message in messages}),
        "session_count": output_count,
        "discarded_single_turn_sessions": discarded_short,
        "session_gap_minutes": session_gap_minutes,
        "merge_gap_minutes": merge_gap_minutes,
    }
    write_json(report_path, report)
    return report
