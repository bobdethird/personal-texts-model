from __future__ import annotations

import csv
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from imessage_mlx.utils import ensure_private_dir, read_jsonl

CSV_COLUMNS = [
    "timestamp_local",
    "sender_name",
    "sender_role",
    "text",
    "chat_id",
    "is_group",
    "service",
    "has_attachment",
    "message_id",
    "participant_id",
    "reply_to_message_id",
    "thread_root_message_id",
    "thread_originator_part",
    "timestamp_ns",
]


def _local_timestamp(timestamp_ns: Any) -> str:
    value = int(timestamp_ns)
    return datetime.fromtimestamp(value / 1_000_000_000).astimezone().isoformat(timespec="seconds")


def export_messages_csv(
    messages_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Export extracted message rows as a private, spreadsheet-friendly CSV."""
    source = Path(messages_path)
    destination = Path(output_path)
    ensure_private_dir(destination.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    rows = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for record in read_jsonl(source):
                row = dict(record)
                row["timestamp_local"] = _local_timestamp(record["timestamp_ns"])
                writer.writerow(row)
                rows += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        destination.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "input": str(source),
        "output": str(destination),
        "rows": rows,
        "columns": CSV_COLUMNS,
    }
