from __future__ import annotations

import asyncio
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Annotated

import mlx.core as mx
import typer
from dotenv import load_dotenv

from imessage_mlx.adapter_runtime import (
    predict_adapter,
    rewrite_with_adapter,
    setup_adapter_environment,
    train_adapter,
)
from imessage_mlx.audit import completion_audit
from imessage_mlx.config import load_yaml, resolve_path
from imessage_mlx.data.censor import censor_llm_dataset, create_censor_review
from imessage_mlx.data.extract import extract_messages
from imessage_mlx.data.inspect_schema import inspect_schema
from imessage_mlx.data.llm_dataset import build_llm_dataset, create_llm_dataset_review
from imessage_mlx.data.privacy_audit import audit_extracted_messages
from imessage_mlx.data.sessions import build_sessions
from imessage_mlx.data.snapshot import can_open_readonly, create_snapshot
from imessage_mlx.data.split import split_sessions
from imessage_mlx.data.table_export import export_messages_csv
from imessage_mlx.dataset import encode_all_splits, select_model
from imessage_mlx.evaluate import evaluate_checkpoint
from imessage_mlx.export import export_model
from imessage_mlx.generate import stream_reply
from imessage_mlx.tokenizer.train import load_tokenizer, train_tokenizer
from imessage_mlx.train import train_model
from imessage_mlx.utils import ensure_private_dir, write_json

app = typer.Typer(
    no_args_is_help=True,
    help="Train and run a private, from-scratch iMessage language model with MLX.",
)


def _emit(value) -> None:
    typer.echo(json.dumps(value, indent=2, sort_keys=True))


def _require_api_key() -> str:
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise typer.BadParameter("Set OPENAI_API_KEY in the environment or ignored .env file")
    return api_key


@app.command()
def doctor(
    config: Annotated[Path, typer.Option(help="Data configuration file")] = Path(
        "configs/data.yaml"
    ),
) -> None:
    """Check the local MLX, privacy, disk, and Messages-database prerequisites."""
    settings = load_yaml(config)
    source = resolve_path(settings["source_db"])
    work_directory = ensure_private_dir(resolve_path(settings.get("work_dir", "work")))
    output_directory = ensure_private_dir(resolve_path(settings.get("output_dir", "outputs")))
    readable, access_error = can_open_readonly(source)
    disk = shutil.disk_usage(Path.cwd())

    def ignored(path: str) -> bool:
        result = subprocess.run(
            ["git", "check-ignore", "-q", path],
            cwd=Path.cwd(),
            check=False,
            capture_output=True,
        )
        return result.returncode == 0

    report = {
        "architecture": platform.machine(),
        "apple_silicon": platform.machine() == "arm64",
        "mlx_metal_available": bool(mx.metal.is_available()),
        "python_version": platform.python_version(),
        "source_database_exists": source.exists(),
        "source_database_readable_read_only": readable,
        "source_access_error": access_error,
        "free_disk_bytes": disk.free,
        "work_private_files_ignored": ignored("work/privacy-check.db"),
        "outputs_model_files_ignored": ignored("outputs/privacy-check.safetensors"),
        "private_directory_permissions": {
            "work": oct(work_directory.stat().st_mode & 0o777),
            "outputs": oct(output_directory.stat().st_mode & 0o777),
        },
        "safe_to_run_synthetic_pipeline": bool(mx.metal.is_available()),
        "safe_to_snapshot_real_data": readable,
    }
    _emit(report)


@app.command("snapshot")
def snapshot_command(
    config: Annotated[Path, typer.Option(help="Data configuration file")] = Path(
        "configs/data.yaml"
    ),
) -> None:
    """Create a consistent local backup without modifying the live Messages database."""
    settings = load_yaml(config)
    manifest = create_snapshot(
        resolve_path(settings["source_db"]), resolve_path(settings["snapshot_db"])
    )
    _emit(manifest)


@app.command("inspect-schema")
def inspect_schema_command(
    database: Annotated[Path, typer.Option(help="Read-only database snapshot")] = Path(
        "work/snapshot/chat.db"
    ),
    output: Annotated[Path, typer.Option(help="Private schema output")] = Path(
        "work/schema/schema.json"
    ),
) -> None:
    """Record table and column metadata without reading message bodies."""
    schema = inspect_schema(resolve_path(database), resolve_path(output))
    _emit(
        {
            "table_count": len(schema["tables"]),
            "missing_expected_tables": schema["missing_expected_tables"],
            "output": str(resolve_path(output)),
        }
    )


@app.command("prepare")
def prepare_command(
    config: Annotated[Path, typer.Option(help="Data configuration file")] = Path(
        "configs/data.yaml"
    ),
    database: Annotated[
        Path | None, typer.Option(help="Override snapshot database; useful for synthetic tests")
    ] = None,
) -> None:
    """Extract, pseudonymize, sessionize, and split the snapshot."""
    settings = load_yaml(config)
    work = resolve_path(settings.get("work_dir", "work"))
    ensure_private_dir(work)
    snapshot = resolve_path(database or settings["snapshot_db"])
    inspect_schema(snapshot, work / "schema/schema.json")
    extraction = extract_messages(
        snapshot,
        work / "extracted/messages.jsonl",
        work / "reports/extraction.json",
        work / "private/pseudonym-key",
        redaction=settings.get("redaction", {}),
        include_attachment_marker=bool(settings.get("include_attachment_marker", True)),
        minimum_body_recovery_rate=float(settings.get("minimum_body_recovery_rate", 0.90)),
        include_contact_names=bool(settings.get("include_contact_names", False)),
    )
    sessions = build_sessions(
        work / "extracted/messages.jsonl",
        work / "processed/sessions.jsonl",
        work / "reports/sessions.json",
        session_gap_minutes=int(settings.get("session_gap_minutes", 360)),
        merge_gap_minutes=int(settings.get("merge_gap_minutes", 2)),
    )
    split_settings = settings.get("split", {})
    split_report = split_sessions(
        work / "processed/sessions.jsonl",
        work / "splits",
        work / "reports/split-report.json",
        train_fraction=float(split_settings.get("train", 0.90)),
        validation_fraction=float(split_settings.get("validation", 0.05)),
        test_fraction=float(split_settings.get("test", 0.05)),
        guard_days=int(split_settings.get("guard_days", 7)),
    )
    _emit({"extraction": extraction, "sessions": sessions, "split": split_report})


@app.command("build-llm-dataset")
def build_llm_dataset_command(
    messages: Annotated[Path, typer.Option(help="Private extracted message JSONL")] = Path(
        "work/extracted/messages.jsonl"
    ),
    output: Annotated[Path, typer.Option(help="Private dataset run directory")] = Path(
        "work/llm/run"
    ),
    model: Annotated[str | None, typer.Option(help="OpenAI model ID")] = None,
    judge_model: Annotated[
        str | None, typer.Option(help="OpenAI judge model ID; defaults to the proposal model")
    ] = None,
    max_window_messages: Annotated[int, typer.Option(min=2, max=400)] = 40,
    limit_windows: Annotated[
        int | None, typer.Option(min=1, help="Pilot on a deterministic window sample")
    ] = None,
    concurrency: Annotated[int, typer.Option(min=1, max=20)] = 4,
    max_attempts: Annotated[int, typer.Option(min=1, max=5)] = 3,
) -> None:
    """Generate judge-screened rewrite pairs with hosted model decisions only."""
    api_key = _require_api_key()
    model_name = model or os.environ.get("OPENAI_MODEL", "gpt-5.5")
    report = asyncio.run(
        build_llm_dataset(
            resolve_path(messages),
            resolve_path(output),
            api_key=api_key,
            model=model_name,
            judge_model=judge_model,
            max_window_messages=max_window_messages,
            limit_windows=limit_windows,
            concurrency=concurrency,
            max_attempts=max_attempts,
        )
    )
    _emit(report)
    if report["windows"]["proposal_failed"] or report["windows"]["judge_failed"]:
        raise typer.Exit(code=1)


@app.command("censor-llm-dataset")
def censor_llm_dataset_command(
    results: Annotated[Path, typer.Option(help="Private dataset run results JSONL")] = Path(
        "work/llm/run/results.jsonl"
    ),
    output: Annotated[Path, typer.Option(help="Private dataset run directory")] = Path(
        "work/llm/run"
    ),
    model: Annotated[str | None, typer.Option(help="OpenAI censor model ID")] = None,
    batch_size: Annotated[int, typer.Option(min=1, max=50)] = 20,
    concurrency: Annotated[int, typer.Option(min=1, max=20)] = 4,
    max_attempts: Annotated[int, typer.Option(min=1, max=5)] = 3,
    valid_fraction: Annotated[float, typer.Option(min=0.0, max=0.4)] = 0.05,
    test_fraction: Annotated[float, typer.Option(min=0.0, max=0.4)] = 0.05,
) -> None:
    """Screen every accepted pair and publish only censor-allowed training splits."""
    api_key = _require_api_key()
    model_name = model or os.environ.get("OPENAI_MODEL", "gpt-5.5")
    report = asyncio.run(
        censor_llm_dataset(
            resolve_path(results),
            resolve_path(output),
            api_key=api_key,
            model=model_name,
            batch_size=batch_size,
            concurrency=concurrency,
            max_attempts=max_attempts,
            valid_fraction=valid_fraction,
            test_fraction=test_fraction,
        )
    )
    _emit(report)
    if report["pairs"]["screening_failures_excluded"]:
        raise typer.Exit(code=1)


@app.command("review-llm-dataset")
def review_llm_dataset_command(
    results: Annotated[Path, typer.Option(help="Private dataset run results JSONL")] = Path(
        "work/llm/run/results.jsonl"
    ),
    output: Annotated[Path, typer.Option(help="Private review Markdown")] = Path(
        "work/llm/run/review.md"
    ),
    sample_size: Annotated[int, typer.Option(min=1, max=500)] = 50,
) -> None:
    """Render a private sample of generated pairs for human review."""
    _emit(
        create_llm_dataset_review(
            resolve_path(results),
            resolve_path(output),
            sample_size=sample_size,
        )
    )


@app.command("review-censor")
def review_censor_command(
    censor: Annotated[Path, typer.Option(help="Private censor screening JSONL")] = Path(
        "work/llm/run/censor.jsonl"
    ),
    output: Annotated[Path, typer.Option(help="Private exclusion review Markdown")] = Path(
        "work/llm/run/censor-review.md"
    ),
    max_rows: Annotated[int, typer.Option(min=1, max=1000)] = 200,
) -> None:
    """Render the censor's excluded rows for human spot-checking."""
    _emit(
        create_censor_review(
            resolve_path(censor),
            resolve_path(output),
            max_rows=max_rows,
        )
    )


@app.command("setup-adapter-environment")
def setup_adapter_environment_command(
    architecture: Annotated[
        str,
        typer.Argument(help="Adapter architecture: bart, flan_t5, marian, or qwen"),
    ],
    output: Annotated[Path, typer.Option(help="Private isolated virtual environment")],
) -> None:
    """Install an isolated local adapter-training environment."""
    _emit(
        setup_adapter_environment(
            architecture,
            resolve_path(output),
            project_root=Path.cwd(),
        )
    )


@app.command("train-adapter")
def train_adapter_command(
    config: Annotated[Path, typer.Option(help="Adapter configuration YAML")],
    environment: Annotated[Path, typer.Option(help="Isolated adapter virtual environment")],
    data: Annotated[Path, typer.Option(help="Censored adapter dataset root")] = Path(
        "work/llm/run/dataset"
    ),
    output: Annotated[Path, typer.Option(help="Private adapter run directory")] = Path(
        "outputs/adapters/run"
    ),
    benchmark: Annotated[
        bool, typer.Option(help="Use the small architecture benchmark split")
    ] = False,
) -> None:
    """Train a local seq2seq LoRA or Qwen QLoRA adapter."""
    _emit(
        train_adapter(
            resolve_path(config),
            resolve_path(data),
            resolve_path(output),
            resolve_path(environment),
            benchmark=benchmark,
            project_root=Path.cwd(),
        )
    )


@app.command("predict-adapter")
def predict_adapter_command(
    config: Annotated[Path, typer.Option(help="Adapter configuration YAML")],
    environment: Annotated[Path, typer.Option(help="Isolated adapter virtual environment")],
    adapter: Annotated[Path, typer.Option(help="Private adapter run directory")],
    data: Annotated[Path, typer.Option(help="Censored adapter dataset root")] = Path(
        "work/llm/run/dataset"
    ),
    output: Annotated[Path, typer.Option(help="Private prediction JSONL")] = Path(
        "work/llm/evaluation/predictions.jsonl"
    ),
    benchmark: Annotated[
        bool, typer.Option(help="Use the small architecture benchmark split")
    ] = False,
    test_file: Annotated[
        Path | None, typer.Option(help="Optional explicit seq2seq or MLX test JSONL")
    ] = None,
) -> None:
    """Generate held-out predictions from a local adapter."""
    _emit(
        predict_adapter(
            resolve_path(config),
            resolve_path(data),
            resolve_path(adapter),
            resolve_path(output),
            resolve_path(environment),
            benchmark=benchmark,
            data_file=resolve_path(test_file) if test_file else None,
            project_root=Path.cwd(),
        )
    )


@app.command("rewrite-adapter")
def rewrite_adapter_command(
    neutral_draft: Annotated[str, typer.Argument(help="Neutral draft to rewrite")],
    config: Annotated[Path, typer.Option(help="Adapter configuration YAML")] = Path(
        "configs/adapter-bart-base.yaml"
    ),
    adapter: Annotated[Path, typer.Option(help="Private adapter run directory")] = Path(
        "outputs/adapters/run"
    ),
    environment: Annotated[Path, typer.Option(help="Isolated adapter environment")] = Path(
        "work/envs/bart"
    ),
) -> None:
    """Rewrite one draft with deterministic local adapter inference."""
    typer.echo(
        rewrite_with_adapter(
            neutral_draft,
            resolve_path(config),
            resolve_path(adapter),
            resolve_path(environment),
            project_root=Path.cwd(),
        )
    )


@app.command("train-tokenizer")
def train_tokenizer_command(
    train: Annotated[Path, typer.Option(help="Training JSONL only")] = Path(
        "work/splits/train.jsonl"
    ),
    output: Annotated[Path, typer.Option(help="Private tokenizer directory")] = Path(
        "outputs/tokenizer"
    ),
    vocab_size: Annotated[int, typer.Option(min=256)] = 4096,
) -> None:
    """Train a byte-level BPE tokenizer on the training split only."""
    _emit(train_tokenizer(resolve_path(train), resolve_path(output), vocab_size=vocab_size))


@app.command("privacy-audit")
def privacy_audit_command(
    messages: Annotated[Path, typer.Option(help="Pseudonymized extracted JSONL")] = Path(
        "work/extracted/messages.jsonl"
    ),
    output: Annotated[Path, typer.Option(help="Aggregate audit JSON")] = Path(
        "work/reports/privacy-audit.json"
    ),
) -> None:
    """Verify canonical hashed identifiers and obvious-PII redaction without printing text."""
    report = audit_extracted_messages(resolve_path(messages), resolve_path(output))
    _emit(report)
    if not report["passed"]:
        raise typer.Exit(code=1)


@app.command("export-messages-csv")
def export_messages_csv_command(
    messages: Annotated[Path, typer.Option(help="Extracted message JSONL")] = Path(
        "work/extracted/messages.jsonl"
    ),
    output: Annotated[Path, typer.Option(help="Private spreadsheet-friendly CSV")] = Path(
        "work/extracted/messages.csv"
    ),
) -> None:
    """Export one message per CSV row with a readable local timestamp."""
    _emit(export_messages_csv(resolve_path(messages), resolve_path(output)))


@app.command("corpus-stats")
def corpus_stats_command(
    splits: Annotated[Path, typer.Option(help="JSONL split directory")] = Path("work/splits"),
    tokenizer: Annotated[Path, typer.Option(help="Tokenizer directory")] = Path(
        "outputs/tokenizer"
    ),
    output: Annotated[Path, typer.Option(help="Encoded array directory")] = Path("work/tokens"),
) -> None:
    """Encode all splits, report token counts, and select a safe model size."""
    token_report = encode_all_splits(
        resolve_path(splits), resolve_path(tokenizer), resolve_path(output)
    )
    tokenizer_value = load_tokenizer(resolve_path(tokenizer))
    selection = select_model(
        int(token_report["train"]["tokens"]),
        tokenizer_value.get_vocab_size(),
        load_yaml("configs/model-1m.yaml"),
        load_yaml("configs/model-7m.yaml"),
    )
    report_path = resolve_path("work/reports/model-selection.json")
    write_json(report_path, selection)
    _emit({"tokens": token_report, "selection": selection})


@app.command("train")
def train_command(
    config: Annotated[Path, typer.Option(help="Model/training configuration")],
    data: Annotated[Path, typer.Option(help="Encoded token arrays")] = Path("work/tokens"),
    tokenizer: Annotated[Path, typer.Option(help="Tokenizer directory")] = Path(
        "outputs/tokenizer"
    ),
    output: Annotated[Path, typer.Option(help="Private run output directory")] = Path(
        "outputs/runs/model-v1"
    ),
    resume_from: Annotated[Path | None, typer.Option(help="Checkpoint to resume")] = None,
    compile_step: Annotated[bool, typer.Option(help="Compile the MLX update step")] = True,
    selection_report: Annotated[
        Path, typer.Option(help="Data-derived model-selection report")
    ] = Path("work/reports/model-selection.json"),
) -> None:
    """Train a decoder-only Transformer from random initialization with MLX."""
    training_config = load_yaml(config)
    selection_path = resolve_path(selection_report)
    tokenizer_path = resolve_path(tokenizer)
    if training_config.get("name") != "smoke":
        if not selection_path.exists():
            raise typer.BadParameter(
                "Run `imessage-mlx corpus-stats` before real training; the model size must be "
                "selected from the local token count."
            )
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if not selection.get("enough_tokens_to_train", False):
            raise typer.BadParameter(
                "The training split has fewer than one million tokens. The safety gate forbids "
                "claiming a meaningful from-scratch training run."
            )
        if training_config.get("name") != selection.get("selected"):
            raise typer.BadParameter(
                f"Model selection chose {selection.get('selected')!r}, but the supplied config is "
                f"{training_config.get('name')!r}."
            )
    summary = train_model(
        training_config,
        resolve_path(data),
        tokenizer_path,
        resolve_path(output),
        resume_from=resolve_path(resume_from) if resume_from else None,
        compile_step=compile_step,
    )
    _emit(summary)


@app.command("evaluate")
def evaluate_command(
    checkpoint: Annotated[Path, typer.Option(help="Best checkpoint directory")],
    data: Annotated[Path, typer.Option(help="Encoded token arrays")] = Path("work/tokens"),
    output: Annotated[Path, typer.Option(help="Aggregate private metrics JSON")] = Path(
        "outputs/evaluation.json"
    ),
) -> None:
    """Evaluate held-out perplexity and aggregate memorization indicators."""
    _emit(evaluate_checkpoint(resolve_path(checkpoint), resolve_path(data), resolve_path(output)))


@app.command("export")
def export_command(
    checkpoint: Annotated[Path, typer.Option(help="Best checkpoint directory")],
    output: Annotated[Path, typer.Option(help="Final private model directory")] = Path(
        "outputs/final"
    ),
    metrics: Annotated[Path | None, typer.Option(help="Evaluation metrics JSON")] = None,
    split_report: Annotated[Path, typer.Option(help="Data split report")] = Path(
        "work/reports/split-report.json"
    ),
    splits: Annotated[Path, typer.Option(help="Source JSONL split directory")] = Path(
        "work/splits"
    ),
) -> None:
    """Export the inference-only local artifact."""
    manifest = export_model(
        resolve_path(checkpoint),
        resolve_path(output),
        metrics_path=resolve_path(metrics) if metrics else None,
        split_report_path=resolve_path(split_report),
        split_dir=resolve_path(splits),
    )
    _emit(manifest)


@app.command("chat")
def chat_command(
    model: Annotated[Path, typer.Option(help="Exported local model directory")] = Path(
        "outputs/final"
    ),
    max_new_tokens: Annotated[int, typer.Option(min=1, max=512)] = 64,
    temperature: Annotated[float, typer.Option(min=0.0)] = 0.8,
    top_p: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.9,
    repetition_penalty: Annotated[float, typer.Option(min=0.1)] = 1.1,
    seed: int = 42,
) -> None:
    """Generate local reply suggestions; this command never sends messages."""
    model_path = resolve_path(model)
    history: list[tuple[str, str]] = []
    typer.echo("Local MLX reply generator. Type /quit to exit. Nothing will be sent.")
    while True:
        try:
            incoming = typer.prompt("other")
        except (EOFError, KeyboardInterrupt):
            typer.echo()
            break
        if incoming.strip() == "/quit":
            break
        typer.echo("me: ", nl=False)
        chunks = []
        for chunk in stream_reply(
            model_path,
            incoming,
            history=history,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        ):
            chunks.append(chunk)
            typer.echo(chunk, nl=False)
        typer.echo()
        history.extend([("other", incoming), ("me", "".join(chunks))])


@app.command("audit")
def audit_command(
    run: Annotated[Path, typer.Option(help="Completed training run directory")] = Path(
        "outputs/runs/model-1m-v2-clean"
    ),
    model: Annotated[Path, typer.Option(help="Final exported model directory")] = Path(
        "outputs/final"
    ),
) -> None:
    """Run the aggregate Gate A-D completion audit without printing private text."""
    report = completion_audit(Path.cwd(), run_dir=run, final_dir=model)
    _emit(report)
    if not report["ready"]:
        raise typer.Exit(code=1)
