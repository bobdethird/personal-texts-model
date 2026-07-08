from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from imessage_mlx.data.redact import contains_obvious_pii
from imessage_mlx.utils import read_jsonl, write_json

HASH_RE = re.compile(r"^[0-9a-f]{24}$")
EXPECTED_KEYS = {
    "message_id",
    "chat_id",
    "timestamp_ns",
    "sender_role",
    "participant_id",
    "text",
    "has_attachment",
    "is_group",
    "service",
}


def audit_extracted_messages(
    messages_path: str | Path, output_path: str | Path | None = None
) -> dict[str, Any]:
    counts = {
        "records": 0,
        "unexpected_schema_records": 0,
        "invalid_hashed_identifier_records": 0,
        "invalid_role_records": 0,
        "obvious_pii_records": 0,
    }
    for record in read_jsonl(messages_path):
        counts["records"] += 1
        if set(record) != EXPECTED_KEYS:
            counts["unexpected_schema_records"] += 1
        if not all(
            HASH_RE.fullmatch(str(record.get(name, "")))
            for name in ("message_id", "chat_id", "participant_id")
        ):
            counts["invalid_hashed_identifier_records"] += 1
        if record.get("sender_role") not in {"me", "other"}:
            counts["invalid_role_records"] += 1
        if contains_obvious_pii(str(record.get("text", ""))):
            counts["obvious_pii_records"] += 1
    passed = counts["records"] > 0 and not any(
        counts[name]
        for name in (
            "unexpected_schema_records",
            "invalid_hashed_identifier_records",
            "invalid_role_records",
            "obvious_pii_records",
        )
    )
    report = {
        "passed": passed,
        "counts": counts,
        "raw_identifier_fields_persisted": False,
        "attachment_paths_selected_by_extractor": False,
        "message_text_persisted_in_report": False,
    }
    if output_path is not None:
        write_json(output_path, report)
    return report
