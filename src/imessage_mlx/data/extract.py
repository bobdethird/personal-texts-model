from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from imessage_mlx.data.attributed_body import decode_attributed_body
from imessage_mlx.data.contacts import resolve_macos_contact_names
from imessage_mlx.data.inspect_schema import column_names, inspect_schema
from imessage_mlx.data.normalize import normalize_text
from imessage_mlx.data.redact import load_or_create_key, pseudonym
from imessage_mlx.data.snapshot import open_readonly
from imessage_mlx.utils import ensure_private_dir, write_json, write_jsonl

APPLE_EPOCH_UNIX_SECONDS = 978_307_200


def apple_timestamp_to_unix_ns(value: object) -> int:
    if value in (None, ""):
        return 0
    raw = int(value)
    magnitude = abs(raw)
    if magnitude >= 10**17:  # nanoseconds since 2001
        seconds_since_2001 = raw / 1_000_000_000
    elif magnitude >= 10**14:  # microseconds since 2001
        seconds_since_2001 = raw / 1_000_000
    elif magnitude >= 10**11:  # milliseconds since 2001
        seconds_since_2001 = raw / 1_000
    else:
        seconds_since_2001 = raw
    unix_seconds = seconds_since_2001 + APPLE_EPOCH_UNIX_SECONDS
    result = int(unix_seconds * 1_000_000_000)
    if result and not (946_684_800_000_000_000 <= result <= 4_102_444_800_000_000_000):
        raise ValueError(f"Implausible converted Messages timestamp: {value}")
    return result


def _select(column: str, available: set[str], alias: str | None = None) -> str:
    alias = alias or column
    if column in available:
        return f'm."{column}" AS "{alias}"'
    return f'NULL AS "{alias}"'


def _build_query(message_columns: set[str], tables: set[str]) -> str:
    selected = [
        "m.ROWID AS message_rowid",
        _select("guid", message_columns),
        _select("text", message_columns),
        _select("attributedBody", message_columns, "attributed_body"),
        _select("is_from_me", message_columns),
        _select("handle_id", message_columns),
        _select("date", message_columns),
        _select("service", message_columns),
        _select("item_type", message_columns),
        _select("associated_message_type", message_columns),
        _select("reply_to_guid", message_columns),
        _select("thread_originator_guid", message_columns),
        _select("thread_originator_part", message_columns),
        _select("balloon_bundle_id", message_columns),
        _select("is_deleted", message_columns),
        _select("date_retracted", message_columns),
    ]
    if "chat_message_join" in tables:
        selected.append(
            "(SELECT cmj.chat_id FROM chat_message_join cmj "
            "WHERE cmj.message_id = m.ROWID ORDER BY cmj.chat_id LIMIT 1) AS chat_rowid"
        )
    else:
        selected.append("NULL AS chat_rowid")
    if "message_attachment_join" in tables:
        selected.append(
            "EXISTS(SELECT 1 FROM message_attachment_join maj "
            "WHERE maj.message_id = m.ROWID) AS has_attachment"
        )
    else:
        selected.append("0 AS has_attachment")
    return f"SELECT {', '.join(selected)} FROM message m ORDER BY m.date, m.ROWID"


def _lookup_map(connection: Any, table: str, value_column: str) -> dict[int, str]:
    try:
        rows = connection.execute(f'SELECT ROWID, "{value_column}" FROM "{table}"')
        return {int(row[0]): str(row[1] or row[0]) for row in rows}
    except Exception:
        return {}


def extract_messages(
    database: str | Path,
    output: str | Path,
    report_path: str | Path,
    key_path: str | Path,
    *,
    redaction: dict[str, bool] | None = None,
    include_attachment_marker: bool = True,
    minimum_body_recovery_rate: float = 0.90,
    include_contact_names: bool = False,
    contact_names: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    schema = inspect_schema(database)
    tables = set(schema["tables"])
    if "message" not in tables:
        raise RuntimeError("Snapshot has no message table")
    message_columns = column_names(schema, "message")
    query = _build_query(message_columns, tables)
    key = load_or_create_key(key_path)
    counters: Counter[str] = Counter()
    output_path = Path(output)
    ensure_private_dir(output_path.parent)

    def records() -> Iterator[dict[str, Any]]:
        with open_readonly(database) as connection:
            connection.row_factory = __import__("sqlite3").Row
            handle_map = _lookup_map(connection, "handle", "id") if "handle" in tables else {}
            resolved_contact_names: Mapping[str, str] = {}
            if include_contact_names:
                resolved_contact_names = (
                    dict(contact_names)
                    if contact_names is not None
                    else resolve_macos_contact_names(handle_map.values())
                )
                unique_handles = set(handle_map.values())
                counters["contact_handles_considered"] = len(unique_handles)
                counters["contact_handles_resolved"] = sum(
                    handle in resolved_contact_names for handle in unique_handles
                )
            chat_map = (
                _lookup_map(connection, "chat", "chat_identifier") if "chat" in tables else {}
            )
            group_counts: dict[int, int] = {}
            if "chat_handle_join" in tables:
                group_counts = {
                    int(chat_id): int(count)
                    for chat_id, count in connection.execute(
                        "SELECT chat_id, COUNT(*) FROM chat_handle_join GROUP BY chat_id"
                    )
                }

            for row in connection.execute(query):
                counters["total_rows"] += 1
                if row["is_deleted"] or row["date_retracted"]:
                    counters["excluded_deleted"] += 1
                    continue
                if row["associated_message_type"] not in (None, 0):
                    counters["excluded_reaction"] += 1
                    continue
                if row["item_type"] not in (None, 0):
                    counters["excluded_system_event"] += 1
                    continue

                raw_text = row["text"]
                body_source = "text"
                if not isinstance(raw_text, str) or not raw_text.strip():
                    raw_text = decode_attributed_body(row["attributed_body"])
                    body_source = "attributed_body"
                if not raw_text:
                    if row["has_attachment"]:
                        counters["excluded_attachment_only"] += 1
                    else:
                        counters["excluded_unrecoverable_body"] += 1
                    continue

                text = normalize_text(raw_text, redaction or {})
                if not text:
                    counters["excluded_empty_after_normalization"] += 1
                    continue
                if row["has_attachment"] and include_attachment_marker:
                    text = f"{text}\n<|attachment|>"

                chat_rowid = int(row["chat_rowid"] or 0)
                handle_rowid = int(row["handle_id"] or 0)
                is_from_me = bool(row["is_from_me"])
                chat_identity = chat_map.get(chat_rowid, str(chat_rowid or "unknown"))
                participant_identity = (
                    "me"
                    if is_from_me
                    else handle_map.get(handle_rowid, str(handle_rowid or "unknown"))
                )
                message_identity = row["guid"] or row["message_rowid"]
                reply_to_guid = row["reply_to_guid"]
                thread_originator_guid = row["thread_originator_guid"]
                timestamp_ns = apple_timestamp_to_unix_ns(row["date"])
                if body_source == "text":
                    counters["recovered_text"] += 1
                else:
                    counters["recovered_attributed_body"] += 1
                counters["retained_rows"] += 1
                counters["outgoing_rows" if is_from_me else "incoming_rows"] += 1
                if reply_to_guid:
                    counters["retained_reply_links"] += 1
                if thread_originator_guid:
                    counters["retained_thread_links"] += 1
                is_group = group_counts.get(chat_rowid, 0) > 1
                counters["group_rows" if is_group else "one_to_one_rows"] += 1
                record = {
                    "message_id": pseudonym(message_identity, key, "message"),
                    "reply_to_message_id": (
                        pseudonym(reply_to_guid, key, "message") if reply_to_guid else None
                    ),
                    "thread_root_message_id": (
                        pseudonym(thread_originator_guid, key, "message")
                        if thread_originator_guid
                        else None
                    ),
                    "thread_originator_part": (
                        str(row["thread_originator_part"])
                        if row["thread_originator_part"] is not None
                        else None
                    ),
                    "chat_id": pseudonym(chat_identity, key, "chat"),
                    "timestamp_ns": timestamp_ns,
                    "sender_role": "me" if is_from_me else "other",
                    "participant_id": pseudonym(participant_identity, key, "participant"),
                    "text": text,
                    "has_attachment": bool(row["has_attachment"]),
                    "is_group": is_group,
                    "service": str(row["service"] or "unknown"),
                }
                if include_contact_names:
                    sender_name = (
                        "Me" if is_from_me else resolved_contact_names.get(participant_identity)
                    )
                    record["sender_name"] = sender_name
                    if sender_name and not is_from_me:
                        counters["incoming_rows_with_contact_name"] += 1
                yield record

    written = write_jsonl(output_path, records())
    eligible = counters["retained_rows"] + counters["excluded_unrecoverable_body"]
    recovery_rate = counters["retained_rows"] / eligible if eligible else 0.0
    accounted = sum(
        count
        for name, count in counters.items()
        if name.startswith("excluded_") or name == "retained_rows"
    )
    report = {
        "counts": dict(sorted(counters.items())),
        "written_rows": written,
        "accounted_rows": accounted,
        "all_rows_accounted_for": accounted == counters["total_rows"],
        "body_recovery_rate": recovery_rate,
        "minimum_body_recovery_rate": minimum_body_recovery_rate,
        "schema_missing_expected_tables": schema["missing_expected_tables"],
    }
    write_json(report_path, report)
    if not report["all_rows_accounted_for"]:
        raise RuntimeError(f"Extraction accounting mismatch: {json.dumps(report, sort_keys=True)}")
    if eligible and recovery_rate < minimum_body_recovery_rate:
        raise RuntimeError(
            f"Body recovery rate {recovery_rate:.1%} is below the required "
            f"{minimum_body_recovery_rate:.1%}"
        )
    return report
