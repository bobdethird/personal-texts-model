from pathlib import Path

from imessage_mlx.data.privacy_audit import audit_extracted_messages
from imessage_mlx.utils import write_jsonl


def _record(text: str) -> dict:
    return {
        "message_id": "a" * 24,
        "chat_id": "b" * 24,
        "timestamp_ns": 1,
        "sender_role": "other",
        "participant_id": "c" * 24,
        "text": text,
        "has_attachment": False,
        "is_group": False,
        "service": "iMessage",
    }


def test_privacy_audit_reports_only_aggregate_failures(tmp_path: Path) -> None:
    clean = tmp_path / "clean.jsonl"
    write_jsonl(clean, [_record("hello <|email|>")])
    clean_report = audit_extracted_messages(clean, tmp_path / "clean-report.json")
    assert clean_report["passed"]
    assert "hello" not in (tmp_path / "clean-report.json").read_text(encoding="utf-8")

    unsafe = tmp_path / "unsafe.jsonl"
    write_jsonl(unsafe, [_record("email me at private@example.com")])
    unsafe_report = audit_extracted_messages(unsafe)
    assert not unsafe_report["passed"]
    assert unsafe_report["counts"]["obvious_pii_records"] == 1
