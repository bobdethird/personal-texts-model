from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from imessage_mlx.data.adapters import normalized_fingerprint
from imessage_mlx.data.context_retrieval import (
    CONTEXT_BUNDLE_SCHEMA_VERSION,
    RETRIEVER_VERSION,
    TemporalMessageIndex,
    matched_glossary_entries,
)
from imessage_mlx.data.entity_glossary import glossary_fingerprint, load_glossary
from imessage_mlx.data.rewrite import STRUCTURAL_TOKEN_RE
from imessage_mlx.utils import (
    ensure_private_dir,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)

WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)
REACTION_ONLY_RE = re.compile(
    r"^(?:"
    r"(?:ha)+|(?:he)+|lol+|lmao+|lmfao+|"
    r"ok+|okay+|k+|yes+|yeah+|yep+|no+|nope+|"
    r"thanks?|thx|ty|sure|bet|word|"
    r"[\W_]+"
    r")$",
    re.IGNORECASE,
)


def _is_low_content(text: str) -> bool:
    value = text.strip()
    words = WORD_RE.findall(value)
    return (
        len(value) < 4
        or not words
        or (len(words) == 1 and REACTION_ONLY_RE.fullmatch(value) is not None)
    )


def _merged_turns(
    messages: list[dict[str, Any]],
    *,
    session_gap_ns: int,
    merge_gap_ns: int,
) -> Iterator[list[dict[str, Any]]]:
    current_session: list[dict[str, Any]] = []
    for message in sorted(
        messages,
        key=lambda row: (int(row["timestamp_ns"]), str(row["message_id"])),
    ):
        if (
            current_session
            and int(message["timestamp_ns"]) - int(current_session[-1]["timestamp_ns"])
            > session_gap_ns
        ):
            yield current_session
            current_session = []
        if (
            current_session
            and merge_gap_ns > 0
            and current_session[-1]["sender_role"] == message["sender_role"]
            and current_session[-1]["participant_id"] == message["participant_id"]
            and int(message["timestamp_ns"]) - int(current_session[-1]["timestamp_ns"])
            <= merge_gap_ns
        ):
            current_session[-1]["text"] += "\n" + str(message["text"])
            current_session[-1]["timestamp_ns"] = int(message["timestamp_ns"])
            current_session[-1]["message_ids"].append(str(message["message_id"]))
            for field in ("reply_to_message_id", "thread_root_message_id"):
                linked_id = message.get(field)
                if linked_id and linked_id not in current_session[-1][f"{field}s"]:
                    current_session[-1][f"{field}s"].append(str(linked_id))
            current_session[-1]["has_attachment"] = bool(
                current_session[-1]["has_attachment"] or message.get("has_attachment")
            )
            current_session[-1]["is_group"] = bool(
                current_session[-1]["is_group"] or message.get("is_group")
            )
            continue
        current_session.append(
            {
                "sender_role": str(message["sender_role"]),
                "participant_id": str(message["participant_id"]),
                "text": str(message["text"]),
                "timestamp_ns": int(message["timestamp_ns"]),
                "message_ids": [str(message["message_id"])],
                "reply_to_message_ids": (
                    [str(message["reply_to_message_id"])]
                    if message.get("reply_to_message_id")
                    else []
                ),
                "thread_root_message_ids": (
                    [str(message["thread_root_message_id"])]
                    if message.get("thread_root_message_id")
                    else []
                ),
                "thread_originator_part": message.get("thread_originator_part"),
                "has_attachment": bool(message.get("has_attachment")),
                "is_group": bool(message.get("is_group")),
            }
        )
    if current_session:
        yield current_session


def build_style_targets(
    messages_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    *,
    session_gap_minutes: int = 360,
    merge_gap_minutes: int = 0,
    context_turns: int = 4,
    max_target_characters: int = 512,
    max_context_characters: int = 2_000,
    retrieval_results: int = 4,
    max_retrieval_characters: int = 1_200,
    glossary_path: str | Path | None = None,
) -> dict[str, Any]:
    if session_gap_minutes <= 0 or merge_gap_minutes < 0:
        raise ValueError("Session gap must be positive and merge gap must be nonnegative")
    if (
        context_turns < 0
        or max_target_characters <= 0
        or max_context_characters <= 0
        or retrieval_results < 0
        or max_retrieval_characters < 0
    ):
        raise ValueError("Context and length limits must be nonnegative and nonzero")

    chats: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_messages = list(read_jsonl(messages_path))
    for message in all_messages:
        required = (
            "message_id",
            "chat_id",
            "timestamp_ns",
            "sender_role",
            "participant_id",
            "text",
        )
        if any(key not in message for key in required):
            raise ValueError("Extracted message is missing required style-target fields")
        chats[str(message["chat_id"])].append(message)

    retrieval_index = TemporalMessageIndex(all_messages)
    glossary_entries = load_glossary(glossary_path)
    corpus_fingerprint = sha256_file(messages_path)
    approved_glossary_fingerprint = glossary_fingerprint(glossary_entries)
    session_gap_ns = session_gap_minutes * 60 * 1_000_000_000
    merge_gap_ns = merge_gap_minutes * 60 * 1_000_000_000
    counts: Counter[str] = Counter()

    def records() -> Iterator[dict[str, Any]]:
        for chat_id in sorted(chats):
            message_by_id = {str(message["message_id"]): message for message in chats[chat_id]}
            for session in _merged_turns(
                chats[chat_id],
                session_gap_ns=session_gap_ns,
                merge_gap_ns=merge_gap_ns,
            ):
                for index, turn in enumerate(session):
                    if turn["sender_role"] != "me":
                        continue
                    counts["outgoing_turns"] += 1
                    target = str(turn["text"]).strip()
                    if turn["has_attachment"] or STRUCTURAL_TOKEN_RE.search(target):
                        counts["excluded_attachment_or_structural"] += 1
                        continue
                    if len(target) > max_target_characters:
                        counts["excluded_too_long"] += 1
                        continue
                    if _is_low_content(target):
                        counts["excluded_low_content"] += 1
                        continue

                    reply_to_ids = [str(value) for value in turn["reply_to_message_ids"]]
                    thread_root_ids = [str(value) for value in turn["thread_root_message_ids"]]
                    linked_context_id = (
                        reply_to_ids[0]
                        if reply_to_ids
                        else thread_root_ids[0]
                        if thread_root_ids
                        else None
                    )
                    linked_relation = "reply_to" if reply_to_ids else "thread_root"
                    context_items: list[tuple[dict[str, Any], set[str]]] = []
                    context_characters = 0
                    for previous in reversed(session[max(0, index - context_turns) : index]):
                        text = str(previous["text"]).strip()
                        if not text or STRUCTURAL_TOKEN_RE.search(text):
                            continue
                        if context_characters + len(text) > max_context_characters:
                            break
                        entry = {
                            "role": str(previous["sender_role"]),
                            "participant_id": str(previous["participant_id"]),
                            "text": text,
                            "timestamp_ns": int(previous["timestamp_ns"]),
                            "message_ids": [str(value) for value in previous["message_ids"]],
                        }
                        previous_ids = {str(value) for value in previous["message_ids"]}
                        if linked_context_id in previous_ids:
                            entry["relation"] = linked_relation
                        context_items.append((entry, previous_ids))
                        context_characters += len(text)
                    context_items.reverse()
                    linked_in_recent_context = any(
                        linked_context_id in message_ids for _entry, message_ids in context_items
                    )
                    if linked_context_id and not linked_in_recent_context:
                        linked_message = message_by_id.get(linked_context_id)
                        if linked_message is not None:
                            linked_text = str(linked_message["text"]).strip()
                            if (
                                linked_text
                                and not STRUCTURAL_TOKEN_RE.search(linked_text)
                                and len(linked_text) <= max_context_characters
                            ):
                                while (
                                    context_items
                                    and context_characters + len(linked_text)
                                    > max_context_characters
                                ):
                                    removed, _message_ids = context_items.pop(0)
                                    context_characters -= len(removed["text"])
                                context_items.insert(
                                    0,
                                    (
                                        {
                                            "role": str(linked_message["sender_role"]),
                                            "participant_id": str(linked_message["participant_id"]),
                                            "text": linked_text,
                                            "relation": linked_relation,
                                            "timestamp_ns": int(linked_message["timestamp_ns"]),
                                            "message_ids": [linked_context_id],
                                        },
                                        {linked_context_id},
                                    ),
                                )
                                counts["reply_context_resolved_outside_recent_window"] += 1
                            else:
                                counts["reply_context_unusable"] += 1
                        else:
                            counts["reply_context_unresolved"] += 1
                    if linked_context_id:
                        counts["reply_linked_targets"] += 1
                    chronological_context = [entry for entry, _message_ids in context_items]
                    excluded_ids = {str(value) for value in turn["message_ids"]} | {
                        message_id
                        for _entry, message_ids in context_items
                        for message_id in message_ids
                    }
                    resolution_queries = retrieval_index.plan_queries(target)
                    retrieved = retrieval_index.search(
                        target,
                        before_timestamp_ns=int(turn["timestamp_ns"]),
                        exclude_message_ids=excluded_ids,
                        limit=retrieval_results,
                    )
                    selected_retrieved: list[dict[str, Any]] = []
                    retrieval_characters = 0
                    for evidence in retrieved:
                        text = str(evidence["text"])
                        if retrieval_characters + len(text) > max_retrieval_characters:
                            continue
                        selected_retrieved.append(evidence)
                        retrieval_characters += len(text)
                    glossary = matched_glossary_entries(
                        target,
                        glossary_entries,
                        target_timestamp_ns=int(turn["timestamp_ns"]),
                    )
                    context = [
                        {key: entry[key] for key in ("role", "text", "relation") if key in entry}
                        for entry in chronological_context + selected_retrieved
                    ]
                    if selected_retrieved:
                        counts["targets_with_historical_retrieval"] += 1
                        counts["historical_evidence_rows"] += len(selected_retrieved)
                    if glossary:
                        counts["targets_with_approved_glossary"] += 1
                    if context:
                        counts["targets_with_context"] += 1
                    else:
                        counts["targets_without_context"] += 1

                    message_key = "\0".join(str(value) for value in turn["message_ids"])
                    counts["output_targets"] += 1
                    yield {
                        "pair_id": sha256_text(f"style-target\0{chat_id}\0{message_key}")[:24],
                        "timestamp_ns": int(turn["timestamp_ns"]),
                        "styled_text": target,
                        "context": context,
                        "context_bundle": {
                            "schema_version": CONTEXT_BUNDLE_SCHEMA_VERSION,
                            "exact_links": [
                                entry
                                for entry in chronological_context
                                if entry.get("relation") in {"reply_to", "thread_root"}
                            ],
                            "recent_turns": [
                                entry
                                for entry in chronological_context
                                if entry.get("relation") not in {"reply_to", "thread_root"}
                            ],
                            "retrieved_evidence": selected_retrieved,
                            "glossary_entries": glossary,
                            "resolution_queries": resolution_queries,
                            "retriever": {
                                "kind": "bm25",
                                "version": RETRIEVER_VERSION,
                                "scope": "all_prior_chats_local",
                                "corpus_sha256": corpus_fingerprint,
                                "glossary_sha256": approved_glossary_fingerprint,
                                "maximum_results": retrieval_results,
                                "maximum_characters": max_retrieval_characters,
                                "future_messages_allowed": False,
                            },
                        },
                        "chat_id": chat_id,
                        "target_message_ids": [str(value) for value in turn["message_ids"]],
                        "participant_id": str(turn["participant_id"]),
                        "reply_to_message_id": reply_to_ids[0] if len(reply_to_ids) == 1 else None,
                        "thread_root_message_id": (
                            thread_root_ids[0] if len(thread_root_ids) == 1 else None
                        ),
                        "thread_originator_part": turn["thread_originator_part"],
                        "is_group": bool(turn["is_group"]),
                        "merged_message_count": len(turn["message_ids"]),
                    }

    written = write_jsonl(output_path, records())
    report = {
        "schema_version": 2,
        "task": "style_target_extraction",
        "input_messages": sum(len(values) for values in chats.values()),
        "chat_count": len(chats),
        "written_targets": written,
        "counts": dict(sorted(counts.items())),
        "session_gap_minutes": session_gap_minutes,
        "merge_gap_minutes": merge_gap_minutes,
        "target_unit": (
            "single_message_bubble"
            if merge_gap_minutes == 0
            else "experimental_time_merged_bubbles"
        ),
        "context_turns": context_turns,
        "maximum_target_characters": max_target_characters,
        "maximum_context_characters": max_context_characters,
        "retrieval": {
            "enabled": retrieval_results > 0,
            "scope": "all_prior_chats_local",
            "version": RETRIEVER_VERSION,
            "maximum_results": retrieval_results,
            "maximum_characters": max_retrieval_characters,
            "corpus_sha256": corpus_fingerprint,
            "approved_glossary_entries": len(glossary_entries),
            "glossary_sha256": approved_glossary_fingerprint,
            "future_messages_allowed": False,
        },
        "pre_split_text_deduplication": False,
        "message_text_persisted_in_report": False,
    }
    write_json(report_path, report)
    return report


def split_style_targets(
    targets_path: str | Path,
    output_dir: str | Path,
    report_path: str | Path,
    *,
    train_fraction: float = 0.90,
    validation_fraction: float = 0.05,
    test_fraction: float = 0.05,
    guard_days: int = 7,
) -> dict[str, Any]:
    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to one")
    records = sorted(
        read_jsonl(targets_path),
        key=lambda row: (int(row["timestamp_ns"]), str(row["pair_id"])),
    )
    if len(records) < 3:
        raise ValueError("At least three style targets are required")

    count = len(records)
    train_end_index = max(1, min(count - 2, int(count * train_fraction)))
    validation_end_index = max(
        train_end_index + 1,
        min(count - 1, int(count * (train_fraction + validation_fraction))),
    )
    train_boundary = int(records[train_end_index]["timestamp_ns"])
    validation_boundary = int(records[validation_end_index]["timestamp_ns"])
    guard_ns = guard_days * 24 * 60 * 60 * 1_000_000_000

    provisional: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "valid": [],
        "test": [],
    }
    guarded = 0
    for record in records:
        timestamp = int(record["timestamp_ns"])
        if guard_ns and (
            abs(timestamp - train_boundary) < guard_ns
            or abs(timestamp - validation_boundary) < guard_ns
        ):
            guarded += 1
            continue
        if timestamp < train_boundary:
            provisional["train"].append(record)
        elif timestamp < validation_boundary:
            provisional["valid"].append(record)
        else:
            provisional["test"].append(record)

    accepted: dict[str, list[dict[str, Any]]] = {name: [] for name in provisional}
    seen_fingerprints: set[str] = set()
    removed_cross_split_duplicates = {"train": 0, "valid": 0, "test": 0}
    for name in ("train", "valid", "test"):
        for record in provisional[name]:
            fingerprint = normalized_fingerprint(str(record["styled_text"]))
            if name != "train" and fingerprint in seen_fingerprints:
                removed_cross_split_duplicates[name] += 1
                continue
            accepted[name].append(record)
            seen_fingerprints.add(fingerprint)

    if any(not values for values in accepted.values()):
        raise ValueError("Style-target filtering left an empty split")

    output = ensure_private_dir(output_dir)
    for name, values in accepted.items():
        write_jsonl(output / f"{name}.jsonl", values)
    report = {
        "schema_version": 2,
        "task": "style_target_split",
        "input_targets": len(records),
        "guard_days": guard_days,
        "removed_by_guard_band": guarded,
        "removed_cross_split_duplicates": removed_cross_split_duplicates,
        "counts": {name: len(values) for name, values in accepted.items()},
        "date_ranges_ns": {
            name: {
                "start": min(int(value["timestamp_ns"]) for value in values),
                "end": max(int(value["timestamp_ns"]) for value in values),
            }
            for name, values in accepted.items()
        },
        "training_duplicates_preserved_for_frequency": True,
        "message_text_persisted_in_report": False,
    }
    write_json(report_path, report)
    return report
