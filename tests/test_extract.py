import sqlite3
from pathlib import Path

from imessage_mlx.data.extract import extract_messages
from imessage_mlx.utils import read_jsonl


def test_extracts_redacted_records_and_accounts_for_every_row(
    synthetic_db: Path, tmp_path: Path
) -> None:
    output = tmp_path / "messages.jsonl"
    report = extract_messages(
        synthetic_db,
        output,
        tmp_path / "extraction.json",
        tmp_path / "private/key",
        redaction={"urls": True, "emails": True, "phone_numbers": True},
    )
    records = list(read_jsonl(output))

    assert report["all_rows_accounted_for"]
    assert report["counts"]["total_rows"] == 8
    assert report["counts"]["retained_rows"] == 4
    assert report["counts"]["recovered_attributed_body"] == 1
    assert any("from archived body" in record["text"] for record in records)
    assert any("<|attachment|>" in record["text"] for record in records)
    serialized = output.read_text(encoding="utf-8")
    assert "alice@example.com" not in serialized
    assert "+12125550199" not in serialized
    assert "chat-alice" not in serialized
    assert "/private/photo.jpg" not in serialized
    assert "<|email|>" in serialized
    assert "<|phone|>" in serialized
    assert "<|url|>" in serialized


def test_extracts_pseudonymized_reply_and_thread_lineage(
    synthetic_db: Path,
    tmp_path: Path,
) -> None:
    with sqlite3.connect(synthetic_db) as connection:
        connection.execute("ALTER TABLE message ADD COLUMN reply_to_guid TEXT")
        connection.execute("ALTER TABLE message ADD COLUMN thread_originator_guid TEXT")
        connection.execute("ALTER TABLE message ADD COLUMN thread_originator_part TEXT")
        connection.execute(
            "UPDATE message SET reply_to_guid = ?, thread_originator_guid = ?, "
            "thread_originator_part = ? WHERE guid = ?",
            ("guid-1", "guid-1", "0", "guid-2"),
        )
        connection.commit()

    output = tmp_path / "messages.jsonl"
    extract_messages(
        synthetic_db,
        output,
        tmp_path / "extraction.json",
        tmp_path / "private/key",
    )
    records = list(read_jsonl(output))
    parent = next(record for record in records if "email" in record["text"])
    reply = next(record for record in records if record["text"] == "sounds good 😊")

    assert reply["reply_to_message_id"] == parent["message_id"]
    assert reply["thread_root_message_id"] == parent["message_id"]
    assert reply["thread_originator_part"] == "0"
    assert "guid-1" not in output.read_text(encoding="utf-8")


def test_extracts_contact_names_without_persisting_raw_handles(
    synthetic_db: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "messages.jsonl"
    report = extract_messages(
        synthetic_db,
        output,
        tmp_path / "extraction.json",
        tmp_path / "private/key",
        include_contact_names=True,
        contact_names={"alice@example.com": "Alice Example"},
    )
    records = list(read_jsonl(output))

    alice = next(record for record in records if "email" in record["text"])
    outgoing = next(record for record in records if record["sender_role"] == "me")
    unmatched = next(record for record in records if record["text"].startswith("look at this"))

    assert alice["sender_name"] == "Alice Example"
    assert outgoing["sender_name"] == "Me"
    assert unmatched["sender_name"] is None
    assert report["counts"]["contact_handles_considered"] == 2
    assert report["counts"]["contact_handles_resolved"] == 1
    serialized = output.read_text(encoding="utf-8")
    assert "alice@example.com" not in serialized
    assert "+12125550199" not in serialized
