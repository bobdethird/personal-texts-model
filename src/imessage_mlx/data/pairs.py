from __future__ import annotations

from pathlib import Path
from typing import Any

from imessage_mlx.data.sft import SFT_FORMAT
from imessage_mlx.utils import read_jsonl, sha256_text, write_json, write_jsonl

PAIR_FORMAT = "imessage-pair-v1"

# Short incoming messages ("wait wdym", "lol", "when") are ambiguous on their own
# and retrieve unrelated conversations. Embedding the last few turns of context
# instead disambiguates them and stabilises the retrieved demonstrations.
CONTEXT_QUERY_TURNS = 6
CONTEXT_QUERY_MAX_CHARS = 600


def _role_label(role: str) -> str:
    return "Me" if role == "assistant" else "Them"


def select_context_messages(
    messages: list[dict[str, Any]],
    *,
    max_turns: int = CONTEXT_QUERY_TURNS,
) -> list[dict[str, str]]:
    """Return the most recent non-empty turns as plain ``{role, content}`` dicts.

    Participant tags and other metadata are dropped so the turns can be reused as
    self-contained retrieval demonstrations without leaking identifiers.
    """
    if max_turns < 1:
        raise ValueError("max_turns must be at least 1")

    turns: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", ""))
        content = str(message.get("content", "")).strip()
        if not content or role not in {"user", "assistant"}:
            continue
        turns.append({"role": role, "content": content})
    return turns[-max_turns:]


def build_context_query(
    messages: list[dict[str, Any]],
    *,
    max_turns: int = CONTEXT_QUERY_TURNS,
    max_chars: int = CONTEXT_QUERY_MAX_CHARS,
) -> str:
    """Render the recent conversation as a role-labelled retrieval query.

    ``messages`` is the prefix leading up to (but excluding) the reply. We keep the
    most recent ``max_turns`` non-empty turns from both sides so that a short final
    message is grounded by what was said before it.
    """
    recent = select_context_messages(messages, max_turns=max_turns)
    if not recent:
        return ""
    text = "\n".join(f"{_role_label(turn['role'])}: {turn['content']}" for turn in recent)
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text


def _previous_user_index(messages: list[dict[str, Any]], target_index: int) -> int | None:
    for index in range(target_index - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    return None


def derive_pairs(
    records: list[dict[str, Any]],
    *,
    split: str,
) -> tuple[list[dict[str, Any]], int]:
    """Derive next-message pairs while preserving each target's full session context."""
    if split not in {"train", "validation"}:
        raise ValueError("split must be 'train' or 'validation'")

    pairs: list[dict[str, Any]] = []
    skipped_without_incoming = 0
    for record in records:
        if record.get("format") != SFT_FORMAT:
            raise ValueError(f"Unsupported session format {record.get('format')!r}")
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("Session record must contain messages")

        session_id = str(record.get("session_id") or record.get("example_id") or "")
        if not session_id:
            raise ValueError("Session record must contain session_id or example_id")

        for raw_index in record.get("supervised_indexes", []):
            target_index = int(raw_index)
            if not 0 <= target_index < len(messages):
                raise ValueError(f"Supervised index {target_index} is out of range")
            target = messages[target_index]
            if target.get("role") != "assistant":
                raise ValueError("Only assistant messages may be pair targets")

            user_index = _previous_user_index(messages, target_index)
            if user_index is None:
                skipped_without_incoming += 1
                continue

            query = str(messages[user_index].get("content", "")).strip()
            reply = str(target.get("content", "")).strip()
            if not query or not reply:
                skipped_without_incoming += 1
                continue

            prefix = messages[:target_index]
            context_messages = select_context_messages(prefix)
            context_query = build_context_query(prefix) or query
            pair_id = sha256_text(f"{session_id}\0{target_index}\0{reply}")[:24]
            pairs.append(
                {
                    "format": PAIR_FORMAT,
                    "pair_id": pair_id,
                    "session_id": session_id,
                    "split": split,
                    "query": query,
                    "context_query": context_query,
                    "context_messages": context_messages,
                    "reply": reply,
                    "messages": prefix,
                    "target_index": target_index,
                    "query_index": user_index,
                }
            )
    return pairs, skipped_without_incoming


def prepare_pair_dataset(
    train_sessions_path: str | Path,
    validation_sessions_path: str | Path,
    train_path: str | Path,
    validation_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    """Create pair files from an existing leakage-safe session split."""
    train_sessions = list(read_jsonl(train_sessions_path))
    validation_sessions = list(read_jsonl(validation_sessions_path))
    train_session_ids = {
        str(record.get("session_id") or record.get("example_id") or "") for record in train_sessions
    }
    validation_session_ids = {
        str(record.get("session_id") or record.get("example_id") or "")
        for record in validation_sessions
    }
    overlap = (train_session_ids & validation_session_ids) - {""}
    if overlap:
        raise ValueError("Train and validation files contain overlapping sessions")

    train_pairs, train_skipped = derive_pairs(train_sessions, split="train")
    validation_pairs, validation_skipped = derive_pairs(
        validation_sessions,
        split="validation",
    )

    write_jsonl(train_path, train_pairs)
    write_jsonl(validation_path, validation_pairs)
    report = {
        "format": PAIR_FORMAT,
        "source_format": SFT_FORMAT,
        "train_sessions": len(train_sessions),
        "validation_sessions": len(validation_sessions),
        "train_pairs": len(train_pairs),
        "validation_pairs": len(validation_pairs),
        "skipped_targets_without_incoming": train_skipped + validation_skipped,
        "split_policy": "inherited from whole-session SFT split",
    }
    write_json(report_path, report)
    return report
