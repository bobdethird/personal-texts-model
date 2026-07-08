import json
from pathlib import Path

from imessage_mlx.data.sessions import build_sessions
from imessage_mlx.utils import read_jsonl


def test_builds_role_conditioned_sessions(tmp_path: Path) -> None:
    source = Path(__file__).parent / "fixtures/synthetic_messages.jsonl"
    output = tmp_path / "sessions.jsonl"
    report = build_sessions(source, output, tmp_path / "report.json")
    sessions = list(read_jsonl(output))

    assert report["session_count"] == 3
    assert len(sessions) == 3
    assert all(session["text"].startswith("<|bos|><|conversation|>") for session in sessions)
    assert all("<|other|>" in session["text"] for session in sessions)
    assert all("<|me|>" in session["text"] for session in sessions)
    assert all(session["text"].endswith("<|eos|>") for session in sessions)
    assert "participant_id" not in json.dumps(sessions)
