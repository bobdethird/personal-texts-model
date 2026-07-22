from __future__ import annotations

import plistlib
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def synthetic_db(tmp_path: Path) -> Path:
    database = tmp_path / "chat.db"
    sql_path = Path(__file__).parent / "fixtures/synthetic_chat.sql"
    with sqlite3.connect(database) as connection:
        connection.executescript(sql_path.read_text(encoding="utf-8"))
        attributed = plistlib.dumps({"value": "from archived body ✨"}, fmt=plistlib.FMT_BINARY)
        connection.execute("UPDATE message SET attributedBody = ? WHERE ROWID = 3", (attributed,))
        connection.commit()
    return database
