from __future__ import annotations

from pathlib import Path
from typing import Any

from imessage_mlx.data.snapshot import open_readonly
from imessage_mlx.utils import write_json

EXPECTED_TABLES = {
    "message",
    "handle",
    "chat",
    "chat_message_join",
    "chat_handle_join",
    "attachment",
    "message_attachment_join",
}


def inspect_schema(database: str | Path, output: str | Path | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"tables": {}, "missing_expected_tables": []}
    with open_readonly(database) as connection:
        names = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        for table_name in names:
            escaped = table_name.replace('"', '""')
            columns = []
            for row in connection.execute(f'PRAGMA table_info("{escaped}")'):
                columns.append(
                    {
                        "position": row[0],
                        "name": row[1],
                        "type": row[2],
                        "not_null": bool(row[3]),
                        "primary_key": bool(row[5]),
                    }
                )
            result["tables"][table_name] = columns
    result["missing_expected_tables"] = sorted(EXPECTED_TABLES - set(result["tables"]))
    if output is not None:
        write_json(output, result)
    return result


def column_names(schema: dict[str, Any], table: str) -> set[str]:
    return {column["name"] for column in schema.get("tables", {}).get(table, [])}
