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
