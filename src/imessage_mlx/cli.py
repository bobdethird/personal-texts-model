from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from imessage_mlx.data.extract import extract_messages
from imessage_mlx.data.inspect_schema import inspect_schema
from imessage_mlx.data.sessions import build_sessions
from imessage_mlx.data.sft import prepare_sft_dataset
from imessage_mlx.data.snapshot import can_open_readonly, create_snapshot
from imessage_mlx.data.table_export import export_messages_csv
from imessage_mlx.utils import ensure_private_dir

app = typer.Typer(
    no_args_is_help=True,
    help="Export a private local copy of Messages from macOS.",
)


def _emit(value: object) -> None:
    typer.echo(json.dumps(value, indent=2, sort_keys=True))


@app.command()
def doctor(
    source: Annotated[
        Path,
        typer.Option(help="Live macOS Messages database"),
    ] = Path("~/Library/Messages/chat.db"),
) -> None:
    """Check whether the Messages database is available read-only."""
    database = source.expanduser().resolve()
    readable, error = can_open_readonly(database)
    _emit(
        {
            "source": str(database),
            "exists": database.exists(),
            "readable_read_only": readable,
            "error": error,
        }
    )
    if not readable:
        raise typer.Exit(code=1)


@app.command()
def download(
    source: Annotated[
        Path,
        typer.Option(help="Live macOS Messages database"),
    ] = Path("~/Library/Messages/chat.db"),
    output: Annotated[
        Path,
        typer.Option(help="Private directory for the snapshot and exports"),
    ] = Path("work/imessages"),
    include_contact_names: Annotated[
        bool,
        typer.Option(help="Resolve sender names through the macOS Contacts app"),
    ] = False,
    redact_private_data: Annotated[
        bool,
        typer.Option(help="Replace URLs, email addresses, and phone numbers in message text"),
    ] = False,
    minimum_body_recovery_rate: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help="Fail below this recovered-body fraction"),
    ] = 0.90,
) -> None:
    """Snapshot Messages, then export message bodies to JSONL and CSV."""
    source_path = source.expanduser().resolve()
    output_path = ensure_private_dir(output.expanduser().resolve())
    snapshot_path = output_path / "chat.db"
    messages_path = output_path / "messages.jsonl"
    csv_path = output_path / "messages.csv"
    report_path = output_path / "extraction-report.json"
    key_path = output_path / "pseudonym-key"

    snapshot = create_snapshot(source_path, snapshot_path)
    schema = inspect_schema(snapshot_path, output_path / "schema.json")
    extraction = extract_messages(
        snapshot_path,
        messages_path,
        report_path,
        key_path,
        redaction={
            "urls": redact_private_data,
            "emails": redact_private_data,
            "phone_numbers": redact_private_data,
        },
        include_contact_names=include_contact_names,
        minimum_body_recovery_rate=minimum_body_recovery_rate,
    )
    csv_export = export_messages_csv(messages_path, csv_path)

    _emit(
        {
            "source": str(source_path),
            "output": str(output_path),
            "snapshot": snapshot,
            "schema_tables": len(schema["tables"]),
            "extraction": extraction,
            "csv": csv_export,
        }
    )


@app.command()
def chunk(
    messages: Annotated[
        Path,
        typer.Option(help="Extracted message JSONL"),
    ] = Path("work/imessages/messages.jsonl"),
    output: Annotated[
        Path,
        typer.Option(help="Conversation-session JSONL"),
    ] = Path("work/imessages/sessions.jsonl"),
    report: Annotated[
        Path,
        typer.Option(help="Sessionization report JSON"),
    ] = Path("work/imessages/session-report.json"),
    session_gap_minutes: Annotated[
        int,
        typer.Option(min=1, help="Start a new session after this many inactive minutes"),
    ] = 360,
    merge_gap_minutes: Annotated[
        int,
        typer.Option(min=0, help="Merge consecutive same-sender messages within this window"),
    ] = 2,
) -> None:
    """Group an existing message export into conversation sessions."""
    result = build_sessions(
        messages.expanduser().resolve(),
        output.expanduser().resolve(),
        report.expanduser().resolve(),
        session_gap_minutes=session_gap_minutes,
        merge_gap_minutes=merge_gap_minutes,
    )
    _emit(result)


@app.command("prepare-sft")
def prepare_sft(
    messages: Annotated[
        Path,
        typer.Option(help="Extracted message JSONL"),
    ] = Path("work/imessages/messages.jsonl"),
    output: Annotated[
        Path,
        typer.Option(help="Private directory for train, validation, and report files"),
    ] = Path("work/imessages/sft"),
    session_gap_minutes: Annotated[
        int,
        typer.Option(min=1, help="Start a new conversation after this many inactive minutes"),
    ] = 360,
    validation_fraction: Annotated[
        float,
        typer.Option(min=0.0, max=0.99, help="Fraction of whole sessions held out"),
    ] = 0.05,
    max_history_messages: Annotated[
        int,
        typer.Option(min=0, help="Maximum prior messages per target; 0 keeps all history"),
    ] = 0,
) -> None:
    """Build one conversational SFT example for every message sent by me."""
    output_path = output.expanduser().resolve()
    result = prepare_sft_dataset(
        messages.expanduser().resolve(),
        output_path / "train.jsonl",
        output_path / "validation.jsonl",
        output_path / "report.json",
        session_gap_minutes=session_gap_minutes,
        validation_fraction=validation_fraction,
        max_history_messages=max_history_messages or None,
    )
    _emit(result)
