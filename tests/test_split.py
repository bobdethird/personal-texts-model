from pathlib import Path

from imessage_mlx.data.split import split_sessions
from imessage_mlx.utils import read_jsonl, write_jsonl


def test_chronological_split_has_no_duplicate_content(tmp_path: Path) -> None:
    sessions = []
    for index in range(40):
        sessions.append(
            {
                "session_id": f"session-{index}",
                "start_ns": index * 10_000_000_000,
                "end_ns": index * 10_000_000_000 + 1,
                "turn_count": 2,
                "is_group": False,
                "text": f"<|bos|><|conversation|> unique {index} <|eos|>",
            }
        )
    sessions.append(dict(sessions[0], session_id="duplicate"))
    source = tmp_path / "sessions.jsonl"
    write_jsonl(source, sessions)

    report = split_sessions(source, tmp_path / "splits", tmp_path / "report.json", guard_days=0)

    assert report["removed_duplicates"] == 1
    assert report["cross_split_duplicate_hashes"] == 0
    train = list(read_jsonl(tmp_path / "splits/train.jsonl"))
    validation = list(read_jsonl(tmp_path / "splits/validation.jsonl"))
    test = list(read_jsonl(tmp_path / "splits/test.jsonl"))
    assert max(row["end_ns"] for row in train) < min(row["end_ns"] for row in validation)
    assert max(row["end_ns"] for row in validation) < min(row["end_ns"] for row in test)
