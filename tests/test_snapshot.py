from pathlib import Path

from imessage_mlx.data.snapshot import can_open_readonly, create_snapshot
from imessage_mlx.utils import sha256_file


def test_snapshot_is_consistent_and_does_not_change_source(
    synthetic_db: Path, tmp_path: Path
) -> None:
    source_hash = sha256_file(synthetic_db)
    source_stat = synthetic_db.stat()
    destination = tmp_path / "private/snapshot.db"

    manifest = create_snapshot(synthetic_db, destination)

    assert destination.exists()
    assert manifest["quick_check"] == "ok"
    assert manifest["source_open_mode"] == "read-only"
    assert sha256_file(synthetic_db) == source_hash
    assert synthetic_db.stat().st_mtime_ns == source_stat.st_mtime_ns
    assert can_open_readonly(destination) == (True, None)
    assert destination.stat().st_mode & 0o777 == 0o600
