from __future__ import annotations

import os
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

from imessage_mlx.utils import ensure_private_dir, sha256_file, write_json


class DatabaseAccessError(RuntimeError):
    """The Messages database cannot be read with the current macOS permissions."""


def readonly_uri(path: str | Path) -> str:
    return f"file:{quote(str(Path(path).expanduser().resolve()))}?mode=ro"


def open_readonly(path: str | Path) -> sqlite3.Connection:
    try:
        connection = sqlite3.connect(readonly_uri(path), uri=True, timeout=30)
        connection.execute("PRAGMA query_only=ON")
        connection.execute("SELECT name FROM sqlite_master LIMIT 1").fetchone()
        return connection
    except sqlite3.Error as error:
        raise DatabaseAccessError(
            "Cannot read the Messages database. Grant Full Disk Access to the application "
            "running this command, then retry. The source will only be opened read-only."
        ) from error


def can_open_readonly(path: str | Path) -> tuple[bool, str | None]:
    try:
        with open_readonly(path):
            return True, None
    except (DatabaseAccessError, OSError) as error:
        return False, str(error)


def create_snapshot(source: str | Path, destination: str | Path) -> dict[str, object]:
    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if source_path == destination_path:
        raise ValueError("Snapshot destination must differ from the live database")

    destination_dir = ensure_private_dir(destination_path.parent)
    source_before = source_path.stat()
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination_dir, prefix=".chat-snapshot.", suffix=".db"
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    temporary_path.chmod(0o600)

    try:
        with (
            open_readonly(source_path) as source_connection,
            sqlite3.connect(temporary_path) as destination_connection,
        ):
            source_connection.backup(destination_connection, pages=2048)
            destination_connection.commit()
        with sqlite3.connect(readonly_uri(temporary_path), uri=True) as check_connection:
            quick_check = check_connection.execute("PRAGMA quick_check").fetchone()[0]
        if quick_check != "ok":
            raise RuntimeError(f"Snapshot quick_check failed: {quick_check}")

        source_after = source_path.stat()
        if (source_before.st_size, source_before.st_mtime_ns) != (
            source_after.st_size,
            source_after.st_mtime_ns,
        ):
            raise RuntimeError(
                "Live database changed during backup; retry to obtain a clean snapshot"
            )

        os.replace(temporary_path, destination_path)
        destination_path.chmod(0o600)
        manifest = {
            "created_at": datetime.now(UTC).isoformat(),
            "source_size": source_before.st_size,
            "source_mtime_ns": source_before.st_mtime_ns,
            "snapshot_size": destination_path.stat().st_size,
            "snapshot_sha256": sha256_file(destination_path),
            "sqlite_version": sqlite3.sqlite_version,
            "quick_check": quick_check,
            "source_open_mode": "read-only",
        }
        write_json(destination_path.parent / "manifest.json", manifest)
        return manifest
    finally:
        temporary_path.unlink(missing_ok=True)
