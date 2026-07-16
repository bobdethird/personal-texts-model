from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from imessage_mlx.data.redact import contains_obvious_pii
from imessage_mlx.utils import read_jsonl, write_json

HASH_RE = re.compile(r"^[0-9a-f]{24}$")
REQUIRED_KEYS = {
    "message_id",
    "reply_to_message_id",
    "thread_root_message_id",
    "thread_originator_part",
    "chat_id",
    "timestamp_ns",
    "sender_role",
    "participant_id",
    "text",
    "has_attachment",
    "is_group",
    "service",
}
OPTIONAL_KEYS = {"sender_name"}


def audit_extracted_messages(
    messages_path: str | Path, output_path: str | Path | None = None
) -> dict[str, Any]:
    counts = {
        "records": 0,
        "unexpected_schema_records": 0,
        "invalid_hashed_identifier_records": 0,
        "invalid_linked_identifier_records": 0,
        "invalid_role_records": 0,
        "invalid_sender_name_records": 0,
        "contact_name_records": 0,
        "obvious_pii_records": 0,
    }
    for record in read_jsonl(messages_path):
        counts["records"] += 1
        keys = set(record)
        if not REQUIRED_KEYS.issubset(keys) or keys - REQUIRED_KEYS - OPTIONAL_KEYS:
            counts["unexpected_schema_records"] += 1
        if not all(
            HASH_RE.fullmatch(str(record.get(name, "")))
            for name in ("message_id", "chat_id", "participant_id")
        ):
            counts["invalid_hashed_identifier_records"] += 1
        if any(
            record.get(name) is not None and HASH_RE.fullmatch(str(record.get(name, ""))) is None
            for name in ("reply_to_message_id", "thread_root_message_id")
        ):
            counts["invalid_linked_identifier_records"] += 1
        if record.get("sender_role") not in {"me", "other"}:
            counts["invalid_role_records"] += 1
        if "sender_name" in record:
            sender_name = record["sender_name"]
            if sender_name is not None:
                counts["contact_name_records"] += 1
                if not isinstance(sender_name, str) or not sender_name.strip():
                    counts["invalid_sender_name_records"] += 1
        if contains_obvious_pii(str(record.get("text", ""))):
            counts["obvious_pii_records"] += 1
    passed = counts["records"] > 0 and not any(
        counts[name]
        for name in (
            "unexpected_schema_records",
            "invalid_hashed_identifier_records",
            "invalid_linked_identifier_records",
            "invalid_role_records",
            "invalid_sender_name_records",
            "obvious_pii_records",
        )
    )
    report = {
        "passed": passed,
        "counts": counts,
        "raw_identifier_fields_persisted": False,
        "contact_names_persisted": counts["contact_name_records"] > 0,
        "attachment_paths_selected_by_extractor": False,
        "message_text_persisted_in_report": False,
    }
    if output_path is not None:
        write_json(output_path, report)
    return report
