import csv
from pathlib import Path

from imessage_mlx.data.table_export import CSV_COLUMNS, export_messages_csv
from imessage_mlx.utils import write_jsonl


def test_exports_spreadsheet_friendly_message_rows(tmp_path: Path) -> None:
    source = tmp_path / "messages.jsonl"
    output = tmp_path / "messages.csv"
    write_jsonl(
        source,
        [
            {
                "message_id": "message-1",
                "reply_to_message_id": None,
                "thread_root_message_id": None,
                "thread_originator_part": None,
                "chat_id": "chat-1",
                "timestamp_ns": 1_000_000_000,
                "sender_name": "Alice Example",
                "sender_role": "other",
                "participant_id": "participant-1",
                "text": "hello,\nworld",
                "has_attachment": False,
                "is_group": False,
                "service": "iMessage",
            }
        ],
    )

    report = export_messages_csv(source, output)

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert report["rows"] == 1
    assert report["columns"] == CSV_COLUMNS
    assert rows[0]["sender_name"] == "Alice Example"
    assert rows[0]["text"] == "hello,\nworld"
    assert rows[0]["timestamp_local"]
    assert rows[0]["message_id"] == "message-1"
    assert output.stat().st_mode & 0o777 == 0o600
