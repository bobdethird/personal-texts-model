from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from imessage_mlx.data.extract import extract_messages
from imessage_mlx.data.inspect_schema import inspect_schema
from imessage_mlx.data.pairs import prepare_pair_dataset
from imessage_mlx.data.sessions import build_sessions
from imessage_mlx.data.sft import prepare_sft_dataset
from imessage_mlx.data.snapshot import can_open_readonly, create_snapshot
from imessage_mlx.data.table_export import export_messages_csv
from imessage_mlx.reporting import render_personalization_report_file
from imessage_mlx.retrieval import DEFAULT_EMBEDDING_MODEL, build_retrieval_index
from imessage_mlx.style_card import run_style_card
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
) -> None:
    """Build one multi-turn SFT example per conversation session."""
    output_path = output.expanduser().resolve()
    result = prepare_sft_dataset(
        messages.expanduser().resolve(),
        output_path / "train.jsonl",
        output_path / "validation.jsonl",
        output_path / "report.json",
        session_gap_minutes=session_gap_minutes,
        validation_fraction=validation_fraction,
    )
    _emit(result)


@app.command("prepare-pairs")
def prepare_pairs(
    sessions: Annotated[
        Path,
        typer.Option(help="Directory containing prepared SFT train and validation JSONL"),
    ] = Path("work/imessages/sft"),
    output: Annotated[
        Path,
        typer.Option(help="Private directory for pair train, validation, and report files"),
    ] = Path("work/imessages/pairs"),
) -> None:
    """Derive incoming-message to reply pairs from the whole-session SFT split."""
    sessions_path = sessions.expanduser().resolve()
    output_path = output.expanduser().resolve()
    result = prepare_pair_dataset(
        sessions_path / "train.jsonl",
        sessions_path / "validation.jsonl",
        output_path / "train.jsonl",
        output_path / "validation.jsonl",
        output_path / "report.json",
    )
    _emit(result)


@app.command("build-retrieval-index")
def build_retrieval(
    pairs: Annotated[
        Path,
        typer.Option(help="Prepared training-pair JSONL"),
    ] = Path("work/imessages/pairs/train.jsonl"),
    output: Annotated[
        Path,
        typer.Option(help="Private directory for vectors, metadata, and manifest"),
    ] = Path("work/imessages/retrieval"),
    model_name: Annotated[
        str,
        typer.Option(help="Sentence Transformers embedding model"),
    ] = DEFAULT_EMBEDDING_MODEL,
    batch_size: Annotated[
        int,
        typer.Option(min=1, help="Embedding batch size"),
    ] = 64,
) -> None:
    """Build a local semantic index over incoming messages in training pairs."""
    result = build_retrieval_index(
        pairs.expanduser().resolve(),
        output.expanduser().resolve(),
        model_name=model_name,
        batch_size=batch_size,
    )
    _emit(result)


@app.command("induce-style-card")
def induce_style_card(
    pairs: Annotated[
        Path,
        typer.Option(help="Prepared training-pair JSONL"),
    ] = Path("work/imessages/pairs/train.jsonl"),
    output: Annotated[
        Path,
        typer.Option(help="Private Markdown file for the induced style guide"),
    ] = Path("work/imessages/style-card.md"),
    model_name: Annotated[
        str,
        typer.Option(help="Instruction model used for style induction"),
    ] = "Qwen/Qwen3-4B-Instruct-2507",
    samples: Annotated[
        int,
        typer.Option(min=1, help="Maximum training pairs to analyze"),
    ] = 128,
    max_source_chars: Annotated[
        int,
        typer.Option(min=1, help="Maximum combined characters in source examples"),
    ] = 24_000,
    max_new_tokens: Annotated[
        int,
        typer.Option(min=1, help="Maximum generated style-card tokens"),
    ] = 768,
    temperature: Annotated[
        float,
        typer.Option(min=0.0, help="Style-card generation temperature"),
    ] = 0.2,
    seed: int = 42,
    force: Annotated[
        bool,
        typer.Option(help="Regenerate even when the output already exists"),
    ] = False,
) -> None:
    """Infer a private style guide from training pairs with one model call."""
    output_path = output.expanduser().resolve()
    result = run_style_card(
        {
            "pairs_path": str(pairs.expanduser().resolve()),
            "output_path": str(output_path),
            "report_path": str(output_path.with_suffix(".json")),
            "model_name": model_name,
            "samples": samples,
            "max_source_chars": max_source_chars,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "seed": seed,
            "force": force,
        }
    )
    _emit({key: value for key, value in result.items() if key != "style_card"})


@app.command("render-personalization-report")
def render_personalization_report(
    results: Annotated[
        Path,
        typer.Option(help="Personalization sample JSON from Modal"),
    ] = Path("outputs/personal-qwen/personalization/sample-examples.json"),
    output: Annotated[
        Path,
        typer.Option(help="Private standalone HTML report"),
    ] = Path("outputs/personal-qwen/personalization/report.html"),
) -> None:
    """Turn personalization sample JSON into a human-readable local report."""
    result = render_personalization_report_file(
        results.expanduser().resolve(),
        output.expanduser().resolve(),
    )
    _emit(result)
