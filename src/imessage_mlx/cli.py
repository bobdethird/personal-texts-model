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
    evaluate_semantics,
    predict_adapter,
    repair_pairs_locally,
    rewrite_with_adapter,
    setup_adapter_environment,
    train_adapter,
    validate_convergence_data_semantics,
)
from imessage_mlx.audit import completion_audit
from imessage_mlx.config import load_yaml, resolve_path
from imessage_mlx.convergence_evaluation import (
    compare_convergence_evaluations,
    create_convergence_comparison_review,
    evaluate_convergence_predictions,
)
from imessage_mlx.data.adapters import (
    prepare_adapter_datasets,
    prepare_convergence_adapter_datasets,
)
from imessage_mlx.data.convergence_generation import generate_openai_convergence_data
from imessage_mlx.data.extract import extract_messages
from imessage_mlx.data.inspect_schema import inspect_schema
from imessage_mlx.data.pair_generation import (
    create_rewrite_review,
    generate_openai_rewrite_pairs,
)
from imessage_mlx.data.privacy_audit import audit_extracted_messages
from imessage_mlx.data.rewrite import prepare_rewrite_dataset
from imessage_mlx.data.sessions import build_sessions
from imessage_mlx.data.snapshot import can_open_readonly, create_snapshot
from imessage_mlx.data.split import split_sessions
from imessage_mlx.dataset import (
    encode_all_rewrite_splits,
    encode_all_splits,
    select_model,
    select_rewrite_model,
)
from imessage_mlx.evaluate import evaluate_checkpoint
from imessage_mlx.export import export_adapter, export_model
from imessage_mlx.generate import generate_rewrite, stream_reply
from imessage_mlx.rewrite_evaluation import (
    compare_rewrite_evaluations,
    create_rewrite_comparison_review,
    evaluate_rewrite_predictions,
    predict_legacy_rewrite,
)
from imessage_mlx.tokenizer.train import load_tokenizer, train_tokenizer
from imessage_mlx.train import train_model
from imessage_mlx.utils import ensure_private_dir, write_json

app = typer.Typer(
    no_args_is_help=True,
    help="Train and run a private, from-scratch iMessage language model with MLX.",
)


def _emit(value) -> None:
    typer.echo(json.dumps(value, indent=2, sort_keys=True))


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


@app.command("prepare-rewrites")
def prepare_rewrites_command(
    pairs: Annotated[Path, typer.Option(help="Private neutral-to-styled pair JSONL")] = Path(
        "work/rewrite/pairs.jsonl"
    ),
    processed: Annotated[Path, typer.Option(help="Validated private pair JSONL")] = Path(
        "work/rewrite/processed/pairs.jsonl"
    ),
    splits: Annotated[Path, typer.Option(help="Chronological rewrite split directory")] = Path(
        "work/rewrite/splits"
    ),
    preparation_report: Annotated[
        Path, typer.Option(help="Aggregate pair preparation report")
    ] = Path("work/rewrite/reports/preparation.json"),
    split_report: Annotated[Path, typer.Option(help="Aggregate rewrite split report")] = Path(
        "work/rewrite/reports/split-report.json"
    ),
    guard_days: Annotated[int, typer.Option(min=0)] = 7,
) -> None:
    """Validate and chronologically split private neutral-to-styled rewrite pairs."""
    report = prepare_rewrite_dataset(
        resolve_path(pairs),
        resolve_path(processed),
        resolve_path(splits),
        resolve_path(preparation_report),
        resolve_path(split_report),
        guard_days=guard_days,
    )
    _emit(report)


@app.command("prepare-adapters")
def prepare_adapters_command(
    splits: Annotated[Path, typer.Option(help="Chronological rewrite split directory")] = Path(
        "work/rewrite/splits"
    ),
    output: Annotated[Path, typer.Option(help="Private adapter dataset directory")] = Path(
        "work/rewrite/adapters"
    ),
    report: Annotated[Path, typer.Option(help="Private adapter data report")] = Path(
        "work/rewrite/reports/adapter-data.json"
    ),
    benchmark_train_size: Annotated[int, typer.Option(min=1)] = 2_000,
    benchmark_eval_size: Annotated[int, typer.Option(min=1)] = 200,
    max_characters: Annotated[int, typer.Option(min=1)] = 512,
) -> None:
    """Build leakage-clean seq2seq and MLX-LM adapter datasets."""
    _emit(
        prepare_adapter_datasets(
            resolve_path(splits),
            resolve_path(output),
            resolve_path(report),
            benchmark_train_size=benchmark_train_size,
            benchmark_eval_size=benchmark_eval_size,
            max_characters=max_characters,
        )
    )


@app.command("generate-convergence-pilot")
def generate_convergence_pilot_command(
    splits: Annotated[Path, typer.Option(help="Accepted BART adapter split directory")] = Path(
        "work/rewrite/adapters/bart"
    ),
    output: Annotated[Path, typer.Option(help="Private convergence generation directory")] = Path(
        "work/rewrite/convergence/pilot"
    ),
    report: Annotated[Path, typer.Option(help="Private aggregate generation report")] = Path(
        "work/rewrite/convergence/pilot-report.json"
    ),
    model: Annotated[str | None, typer.Option(help="OpenAI model ID")] = None,
    train_targets: Annotated[int, typer.Option(min=0)] = 400,
    validation_targets: Annotated[int, typer.Option(min=0)] = 50,
    test_targets: Annotated[int, typer.Option(min=0)] = 50,
    reserve_per_split: Annotated[int | None, typer.Option(min=0)] = None,
    max_characters: Annotated[int, typer.Option(min=1)] = 512,
    concurrency: Annotated[int, typer.Option(min=1, max=20)] = 4,
    max_variant_attempts: Annotated[int, typer.Option(min=1, max=5)] = 2,
) -> None:
    """Generate resumable, wording-blind multi-register inputs through OpenAI."""
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise typer.BadParameter("Set OPENAI_API_KEY in the environment or ignored .env file")
    model_name = model or os.environ.get("OPENAI_MODEL", "gpt-5.5")
    generation_report = asyncio.run(
        generate_openai_convergence_data(
            resolve_path(splits),
            resolve_path(output),
            resolve_path(report),
            api_key=api_key,
            model=model_name,
            allocation={
                "train": train_targets,
                "valid": validation_targets,
                "test": test_targets,
            },
            reserve_per_split=reserve_per_split,
            max_characters=max_characters,
            concurrency=concurrency,
            max_variant_attempts=max_variant_attempts,
        )
    )
    _emit(generation_report)
    if generation_report["variants"]["incomplete_active_groups"] or (
        generation_report["selection"]["active"] != generation_report["selection"]["requested"]
    ):
        raise typer.Exit(code=1)


@app.command("validate-convergence-data-semantics")
def validate_convergence_data_semantics_command(
    environment: Annotated[Path, typer.Option(help="BART evaluation virtual environment")],
    pairs: Annotated[Path, typer.Option(help="Generated convergence pair JSONL")] = Path(
        "work/rewrite/convergence/pilot/published.jsonl"
    ),
    output: Annotated[Path, typer.Option(help="Pairs with local semantic scores")] = Path(
        "work/rewrite/convergence/pilot/semantic-validated.jsonl"
    ),
    report: Annotated[Path, typer.Option(help="Aggregate semantic validation report")] = Path(
        "work/rewrite/convergence/pilot-semantic-report.json"
    ),
    minimum_similarity: Annotated[float, typer.Option(min=-1.0, max=1.0)] = 0.25,
) -> None:
    """Score generated source-to-target meaning preservation locally."""
    _emit(
        validate_convergence_data_semantics(
            resolve_path(pairs),
            resolve_path(output),
            resolve_path(report),
            resolve_path(environment),
            minimum_similarity=minimum_similarity,
            project_root=Path.cwd(),
        )
    )


@app.command("prepare-convergence-adapters")
def prepare_convergence_adapters_command(
    base: Annotated[Path, typer.Option(help="Existing accepted BART data directory")] = Path(
        "work/rewrite/adapters/bart"
    ),
    pairs: Annotated[Path, typer.Option(help="Semantically scored convergence pairs")] = Path(
        "work/rewrite/convergence/pilot/semantic-validated.jsonl"
    ),
    output: Annotated[Path, typer.Option(help="Augmented private adapter data root")] = Path(
        "work/rewrite/convergence/adapters"
    ),
    report: Annotated[Path, typer.Option(help="Convergence adapter data report")] = Path(
        "work/rewrite/convergence/adapter-data-report.json"
    ),
    minimum_similarity: Annotated[float, typer.Option(min=-1.0, max=1.0)] = 0.25,
    minimum_group_mean: Annotated[float, typer.Option(min=-1.0, max=1.0)] = 0.50,
) -> None:
    """Merge complete convergence groups into leakage-clean BART data."""
    _emit(
        prepare_convergence_adapter_datasets(
            resolve_path(base),
            resolve_path(pairs),
            resolve_path(output),
            resolve_path(report),
            minimum_semantic_similarity=minimum_similarity,
            minimum_group_mean_similarity=minimum_group_mean,
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
    data: Annotated[Path, typer.Option(help="Prepared adapter dataset root")] = Path(
        "work/rewrite/adapters"
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
    data: Annotated[Path, typer.Option(help="Prepared adapter dataset root")] = Path(
        "work/rewrite/adapters"
    ),
    output: Annotated[Path, typer.Option(help="Private prediction JSONL")] = Path(
        "work/rewrite/evaluation/predictions.jsonl"
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


@app.command("evaluate-rewrite-adapter")
def evaluate_rewrite_adapter_command(
    predictions: Annotated[Path, typer.Option(help="Private adapter prediction JSONL")],
    train: Annotated[Path, typer.Option(help="Private original training split")] = Path(
        "work/rewrite/splits/train.jsonl"
    ),
    output: Annotated[Path, typer.Option(help="Private rewrite evaluation report")] = Path(
        "work/rewrite/evaluation/report.json"
    ),
    semantic_report: Annotated[
        Path | None, typer.Option(help="Optional private local semantic-similarity report")
    ] = None,
) -> None:
    """Evaluate content preservation, style transfer, fluency, and memorization."""
    _emit(
        evaluate_rewrite_predictions(
            resolve_path(predictions),
            resolve_path(train),
            resolve_path(output),
            semantic_report_path=resolve_path(semantic_report) if semantic_report else None,
        )
    )


@app.command("predict-legacy-rewrite")
def predict_legacy_rewrite_command(
    model: Annotated[Path, typer.Option(help="Exported legacy rewrite model")] = Path(
        "outputs/rewrite-final"
    ),
    test: Annotated[Path, typer.Option(help="Private chronological rewrite test split")] = Path(
        "work/rewrite/splits/test.jsonl"
    ),
    output: Annotated[Path, typer.Option(help="Private prediction JSONL")] = Path(
        "work/rewrite/evaluation/legacy-291k.jsonl"
    ),
    limit: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    """Generate deterministic held-out predictions from the current tiny rewrite model."""
    _emit(
        predict_legacy_rewrite(
            resolve_path(model),
            resolve_path(test),
            resolve_path(output),
            limit=limit,
        )
    )


@app.command("evaluate-rewrite-semantics")
def evaluate_rewrite_semantics_command(
    predictions: Annotated[Path, typer.Option(help="Private adapter prediction JSONL")],
    environment: Annotated[Path, typer.Option(help="BART evaluation virtual environment")],
    output: Annotated[Path, typer.Option(help="Private aggregate semantic report")] = Path(
        "work/rewrite/evaluation/semantic-report.json"
    ),
) -> None:
    """Compute private local embedding similarities without persisting message text."""
    _emit(
        evaluate_semantics(
            resolve_path(predictions),
            resolve_path(output),
            resolve_path(environment),
            project_root=Path.cwd(),
        )
    )


@app.command("evaluate-convergence-adapter")
def evaluate_convergence_adapter_command(
    predictions: Annotated[Path, typer.Option(help="Grouped convergence predictions")],
    train: Annotated[
        Path, typer.Option(help="Adapter training JSONL for memorization checks")
    ] = Path("work/rewrite/convergence/adapters/bart/train.jsonl"),
    output: Annotated[Path, typer.Option(help="Private convergence evaluation report")] = Path(
        "work/rewrite/convergence/evaluation/report.json"
    ),
    semantic_report: Annotated[
        Path | None, typer.Option(help="Optional local prediction semantic report")
    ] = None,
) -> None:
    """Evaluate four-register content, copying, style, and convergence."""
    _emit(
        evaluate_convergence_predictions(
            resolve_path(predictions),
            resolve_path(train),
            resolve_path(output),
            semantic_report_path=resolve_path(semantic_report) if semantic_report else None,
        )
    )


@app.command("compare-convergence-pilot")
def compare_convergence_pilot_command(
    baseline: Annotated[Path, typer.Option(help="Current adapter convergence report")],
    augmented: Annotated[Path, typer.Option(help="Pilot adapter convergence report")],
    output: Annotated[Path, typer.Option(help="Expansion gate report")] = Path(
        "work/rewrite/convergence/evaluation/pilot-comparison.json"
    ),
    baseline_semantic: Annotated[
        Path | None, typer.Option(help="Current adapter local semantic report")
    ] = None,
    augmented_semantic: Annotated[
        Path | None, typer.Option(help="Pilot adapter local semantic report")
    ] = None,
) -> None:
    """Apply explicit same-challenge pilot expansion gates."""
    _emit(
        compare_convergence_evaluations(
            resolve_path(baseline),
            resolve_path(augmented),
            resolve_path(output),
            baseline_semantic_report=(
                resolve_path(baseline_semantic) if baseline_semantic else None
            ),
            augmented_semantic_report=(
                resolve_path(augmented_semantic) if augmented_semantic else None
            ),
        )
    )


@app.command("review-convergence-models")
def review_convergence_models_command(
    baseline: Annotated[Path, typer.Option(help="Current adapter convergence predictions")],
    augmented: Annotated[Path, typer.Option(help="Pilot adapter convergence predictions")],
    output: Annotated[Path, typer.Option(help="Private grouped human review Markdown")] = Path(
        "work/rewrite/convergence/reviews/pilot-comparison.md"
    ),
    summary: Annotated[Path | None, typer.Option(help="Private review summary JSON")] = None,
    sample_size: Annotated[int, typer.Option(min=1, max=200)] = 50,
) -> None:
    """Create a private four-register comparison review."""
    _emit(
        create_convergence_comparison_review(
            {
                "Current BART LoRA": resolve_path(baseline),
                "Convergence Pilot BART LoRA": resolve_path(augmented),
            },
            resolve_path(output),
            summary_path=resolve_path(summary) if summary else None,
            sample_size=sample_size,
        )
    )


@app.command("compare-rewrite-evaluations")
def compare_rewrite_evaluations_command(
    bart: Annotated[Path, typer.Option(help="BART evaluation report")],
    qwen: Annotated[Path, typer.Option(help="Qwen evaluation report")],
    output: Annotated[Path, typer.Option(help="Private architecture decision report")] = Path(
        "work/rewrite/evaluation/architecture-selection.json"
    ),
    legacy: Annotated[Path | None, typer.Option(help="Optional legacy evaluation report")] = None,
    bart_training: Annotated[Path | None, typer.Option(help="BART training report")] = None,
    qwen_training: Annotated[Path | None, typer.Option(help="Qwen training report")] = None,
    bart_prediction_log: Annotated[Path | None, typer.Option(help="BART prediction log")] = None,
    qwen_prediction_log: Annotated[Path | None, typer.Option(help="Qwen prediction log")] = None,
) -> None:
    """Select an architecture using held-out content, style, and fluency gates."""
    reports: dict[str, Path] = {"bart": resolve_path(bart), "qwen": resolve_path(qwen)}
    if legacy is not None:
        reports["legacy_291k"] = resolve_path(legacy)
    training = {
        name: resolve_path(path)
        for name, path in (("bart", bart_training), ("qwen", qwen_training))
        if path is not None
    }
    prediction_logs = {
        name: resolve_path(path)
        for name, path in (
            ("bart", bart_prediction_log),
            ("qwen", qwen_prediction_log),
        )
        if path is not None
    }
    _emit(
        compare_rewrite_evaluations(
            reports,
            resolve_path(output),
            training_report_paths=training,
            prediction_log_paths=prediction_logs,
        )
    )


@app.command("compare-seq2seq-evaluations")
def compare_seq2seq_evaluations_command(
    bart: Annotated[Path, typer.Option(help="BART evaluation report")],
    flan: Annotated[Path, typer.Option(help="Flan-T5 evaluation report")],
    opus: Annotated[Path, typer.Option(help="OPUS-MT evaluation report")],
    output: Annotated[Path, typer.Option(help="Private model decision report")] = Path(
        "work/rewrite/evaluation/seq2seq-selection.json"
    ),
    bart_training: Annotated[Path | None, typer.Option(help="BART training report")] = None,
    flan_training: Annotated[Path | None, typer.Option(help="Flan-T5 training report")] = None,
    opus_training: Annotated[Path | None, typer.Option(help="OPUS-MT training report")] = None,
    bart_prediction_log: Annotated[Path | None, typer.Option(help="BART prediction log")] = None,
    flan_prediction_log: Annotated[Path | None, typer.Option(help="Flan-T5 prediction log")] = None,
    opus_prediction_log: Annotated[Path | None, typer.Option(help="OPUS-MT prediction log")] = None,
) -> None:
    """Select among BART, Flan-T5, and OPUS-MT using the same held-out gates."""
    reports = {
        "bart": resolve_path(bart),
        "flan_t5_small": resolve_path(flan),
        "opus_mt_gem_gem": resolve_path(opus),
    }
    training = {
        name: resolve_path(path)
        for name, path in (
            ("bart", bart_training),
            ("flan_t5_small", flan_training),
            ("opus_mt_gem_gem", opus_training),
        )
        if path is not None
    }
    prediction_logs = {
        name: resolve_path(path)
        for name, path in (
            ("bart", bart_prediction_log),
            ("flan_t5_small", flan_prediction_log),
            ("opus_mt_gem_gem", opus_prediction_log),
        )
        if path is not None
    }
    _emit(
        compare_rewrite_evaluations(
            reports,
            resolve_path(output),
            training_report_paths=training,
            prediction_log_paths=prediction_logs,
        )
    )


@app.command("review-rewrite-models")
def review_rewrite_models_command(
    adapter: Annotated[Path, typer.Option(help="Private adapter prediction JSONL")],
    legacy: Annotated[Path, typer.Option(help="Private legacy prediction JSONL")],
    output: Annotated[Path, typer.Option(help="Private human review Markdown")] = Path(
        "work/rewrite/reviews/model-comparison.md"
    ),
    sample_size: Annotated[int, typer.Option(min=1, max=200)] = 50,
) -> None:
    """Create a private stratified human comparison of adapter and legacy rewrites."""
    _emit(
        create_rewrite_comparison_review(
            {
                "Pretrained BART LoRA": resolve_path(adapter),
                "Legacy 291K Transformer": resolve_path(legacy),
            },
            resolve_path(output),
            sample_size=sample_size,
        )
    )


@app.command("review-seq2seq-models")
def review_seq2seq_models_command(
    bart: Annotated[Path, typer.Option(help="BART prediction JSONL")],
    flan: Annotated[Path, typer.Option(help="Flan-T5 prediction JSONL")],
    opus: Annotated[Path, typer.Option(help="OPUS-MT prediction JSONL")],
    output: Annotated[Path, typer.Option(help="Private human comparison Markdown")] = Path(
        "work/rewrite/reviews/seq2seq-comparison.md"
    ),
    sample_size: Annotated[int, typer.Option(min=1, max=200)] = 50,
) -> None:
    """Create a private blindable comparison of the three seq2seq candidates."""
    _emit(
        create_rewrite_comparison_review(
            {
                "BART Base LoRA": resolve_path(bart),
                "Flan-T5 Small LoRA": resolve_path(flan),
                "OPUS-MT gem-gem LoRA": resolve_path(opus),
            },
            resolve_path(output),
            sample_size=sample_size,
        )
    )


@app.command("promote-adapter")
def promote_adapter_command(
    adapter: Annotated[Path, typer.Option(help="Passing private adapter run directory")],
    evaluation: Annotated[Path, typer.Option(help="Passing rewrite evaluation report")],
    config: Annotated[Path, typer.Option(help="Pinned adapter configuration YAML")] = Path(
        "configs/adapter-bart-base.yaml"
    ),
    data_report: Annotated[Path, typer.Option(help="Private adapter data report")] = Path(
        "work/rewrite/reports/adapter-data.json"
    ),
    architecture_report: Annotated[
        Path, typer.Option(help="Private architecture selection report")
    ] = Path("work/rewrite/evaluation/architecture-selection.json"),
    output: Annotated[Path, typer.Option(help="Promoted private adapter artifact")] = Path(
        "outputs/rewrite-adapter-final"
    ),
) -> None:
    """Promote an adapter only after its rewrite evaluation passes."""
    _emit(
        export_adapter(
            resolve_path(adapter),
            resolve_path(evaluation),
            resolve_path(output),
            config_path=resolve_path(config),
            data_report_path=resolve_path(data_report),
            architecture_report_path=resolve_path(architecture_report),
        )
    )


@app.command("repair-rewrite-pairs-local")
def repair_rewrite_pairs_local_command(
    pairs: Annotated[Path, typer.Option(help="Private original rewrite pair JSONL")] = Path(
        "work/rewrite/pairs-v2.jsonl"
    ),
    output: Annotated[Path, typer.Option(help="Private targeted-repair pair JSONL")] = Path(
        "work/rewrite/pairs-targeted.jsonl"
    ),
    report: Annotated[Path, typer.Option(help="Private aggregate repair report")] = Path(
        "work/rewrite/reports/targeted-repair.json"
    ),
    config: Annotated[Path, typer.Option(help="Local Qwen configuration")] = Path(
        "configs/adapter-qwen3-0.6b.yaml"
    ),
    environment: Annotated[Path, typer.Option(help="Local MLX-LM environment")] = Path(
        "work/envs/mlx-lm"
    ),
    limit: Annotated[int, typer.Option(min=1, max=500)] = 500,
) -> None:
    """Pilot blind two-stage semantic reconstruction on low-signal pairs."""
    _emit(
        repair_pairs_locally(
            resolve_path(config),
            resolve_path(pairs),
            resolve_path(output),
            resolve_path(report),
            resolve_path(environment),
            limit=limit,
            project_root=Path.cwd(),
        )
    )


@app.command("rewrite-adapter")
def rewrite_adapter_command(
    neutral_draft: Annotated[str, typer.Argument(help="Neutral draft to rewrite")],
    config: Annotated[Path, typer.Option(help="Adapter configuration YAML")] = Path(
        "configs/adapter-bart-base.yaml"
    ),
    adapter: Annotated[Path, typer.Option(help="Promoted or training adapter directory")] = Path(
        "outputs/rewrite-adapter-final"
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


@app.command("generate-rewrite-pairs")
def generate_rewrite_pairs_command(
    messages: Annotated[Path, typer.Option(help="Private extracted message JSONL")] = Path(
        "work/extracted/messages.jsonl"
    ),
    output: Annotated[Path, typer.Option(help="Private generated pair JSONL")] = Path(
        "work/rewrite/pairs.jsonl"
    ),
    report: Annotated[Path, typer.Option(help="Aggregate generation report")] = Path(
        "work/rewrite/reports/pair-generation.json"
    ),
    model: Annotated[str | None, typer.Option(help="OpenAI model ID")] = None,
    limit: Annotated[int, typer.Option(min=1)] = 500,
    batch_size: Annotated[int, typer.Option(min=1, max=100)] = 20,
    concurrency: Annotated[int, typer.Option(min=1, max=20)] = 4,
    max_characters: Annotated[int, typer.Option(min=1)] = 1_000,
) -> None:
    """Generate resumable neutral-to-styled pairs with the OpenAI Responses API."""
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise typer.BadParameter("Set OPENAI_API_KEY in the environment or ignored .env file")
    model_name = model or os.environ.get("OPENAI_MODEL", "gpt-5.5")
    generation_report = asyncio.run(
        generate_openai_rewrite_pairs(
            resolve_path(messages),
            resolve_path(output),
            resolve_path(report),
            api_key=api_key,
            model=model_name,
            limit=limit,
            batch_size=batch_size,
            concurrency=concurrency,
            max_characters=max_characters,
        )
    )
    _emit(generation_report)
    if generation_report["failed_batches"]:
        raise typer.Exit(code=1)


@app.command("review-rewrite-pairs")
def review_rewrite_pairs_command(
    pairs: Annotated[Path, typer.Option(help="Private generated pair JSONL")] = Path(
        "work/rewrite/pairs.jsonl"
    ),
    output: Annotated[Path, typer.Option(help="Private Markdown review file")] = Path(
        "work/rewrite/pilot-review.md"
    ),
    sample_size: Annotated[int, typer.Option(min=1)] = 50,
) -> None:
    """Create a private, evenly sampled human-review document."""
    _emit(
        create_rewrite_review(
            resolve_path(pairs),
            resolve_path(output),
            sample_size=sample_size,
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


@app.command("rewrite-corpus-stats")
def rewrite_corpus_stats_command(
    splits: Annotated[Path, typer.Option(help="Rewrite JSONL split directory")] = Path(
        "work/rewrite/splits"
    ),
    tokenizer: Annotated[Path, typer.Option(help="Rewrite tokenizer directory")] = Path(
        "outputs/rewrite-tokenizer"
    ),
    output: Annotated[Path, typer.Option(help="Encoded rewrite array directory")] = Path(
        "work/rewrite/tokens"
    ),
    config: Annotated[Path, typer.Option(help="Rewrite model configuration")] = Path(
        "configs/model-rewrite-190k.yaml"
    ),
    selection_report: Annotated[Path, typer.Option(help="Rewrite model-selection report")] = Path(
        "work/rewrite/reports/model-selection.json"
    ),
) -> None:
    """Encode rewrite pairs with target masks and report supervised token counts."""
    training_config = load_yaml(config)
    tokenizer_path = resolve_path(tokenizer)
    tokenizer_report_path = tokenizer_path / "training-report.json"
    if not tokenizer_report_path.exists():
        raise typer.BadParameter("Rewrite tokenizer is missing its training report")
    tokenizer_training_report = json.loads(tokenizer_report_path.read_text(encoding="utf-8"))
    if int(tokenizer_training_report.get("requested_vocab_size", 0)) != 2048:
        raise typer.BadParameter(
            "Rewrite models require a tokenizer trained with `--vocab-size 2048`"
        )
    token_report = encode_all_rewrite_splits(
        resolve_path(splits),
        tokenizer_path,
        resolve_path(output),
        context_length=int(training_config["max_sequence_length"]),
    )
    tokenizer_value = load_tokenizer(tokenizer_path)
    candidate_configs = [
        load_yaml("configs/model-rewrite-190k.yaml"),
        load_yaml("configs/model-rewrite-290k.yaml"),
    ]
    selection = select_rewrite_model(
        int(token_report["train"]["supervised_tokens"]),
        int(token_report["train"]["pairs"]),
        tokenizer_value.get_vocab_size(),
        candidate_configs,
    )
    selection["token_basis"] = "supervised target tokens"
    selection["requested_vocab_size"] = int(tokenizer_training_report["requested_vocab_size"])
    write_json(resolve_path(selection_report), selection)
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
    expected_task = str(training_config.get("task", "reply"))
    if expected_task not in {"reply", "rewrite"}:
        raise typer.BadParameter(f"Unsupported training task {expected_task!r}.")
    expected_objective = "target_only" if expected_task == "rewrite" else "causal"
    if training_config.get("objective", "causal") != expected_objective:
        raise typer.BadParameter(
            f"Task {expected_task!r} requires objective {expected_objective!r}."
        )
    selection_path = resolve_path(selection_report)
    tokenizer_path = resolve_path(tokenizer)
    if training_config.get("name") != "smoke":
        if not selection_path.exists():
            stats_command = "rewrite-corpus-stats" if expected_task == "rewrite" else "corpus-stats"
            raise typer.BadParameter(
                f"Run `imessage-mlx {stats_command}` before real training; the model size must be "
                "selected from the local token count."
            )
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selected_task = str(selection.get("task", "reply"))
        if selected_task != expected_task:
            raise typer.BadParameter(
                f"Training config requires task {expected_task!r}, but the selection report is "
                f"for {selected_task!r}."
            )
        if expected_task == "rewrite":
            if (
                selection.get("schema_version") != 1
                or selection.get("selection_scope") != "training_split"
            ):
                raise typer.BadParameter(
                    "Rewrite selection report is outdated; rerun `imessage-mlx "
                    "rewrite-corpus-stats`."
                )
            if not selection.get("eligible_to_train", False):
                raise typer.BadParameter(
                    "Rewrite training data does not satisfy the recorded pair, supervised-token, "
                    "and tokens-per-parameter safety gates."
                )
            if int(training_config.get("vocab_size", 0)) != 2048:
                raise typer.BadParameter("Rewrite model configs require a 2,048-token vocabulary")
            tokenizer_training_report_path = tokenizer_path / "training-report.json"
            if not tokenizer_training_report_path.exists():
                raise typer.BadParameter("Rewrite tokenizer is missing its training report")
            tokenizer_training_report = json.loads(
                tokenizer_training_report_path.read_text(encoding="utf-8")
            )
            if int(tokenizer_training_report.get("requested_vocab_size", 0)) != 2048:
                raise typer.BadParameter(
                    "Rewrite models require a tokenizer trained with `--vocab-size 2048`"
                )
            actual_vocab_size = load_tokenizer(tokenizer_path).get_vocab_size()
            if int(selection.get("vocab_size", -1)) != actual_vocab_size:
                raise typer.BadParameter(
                    "Rewrite tokenizer differs from the model-selection report; rerun "
                    "`imessage-mlx rewrite-corpus-stats`."
                )
        elif not selection.get("enough_tokens_to_train", False):
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


@app.command("rewrite")
def rewrite_command(
    neutral_draft: Annotated[str, typer.Argument(help="Neutral draft to rewrite")],
    model: Annotated[Path, typer.Option(help="Exported rewrite model directory")] = Path(
        "outputs/rewrite-final"
    ),
    max_new_tokens: Annotated[int, typer.Option(min=1, max=512)] = 64,
    temperature: Annotated[float, typer.Option(min=0.0)] = 0.0,
    top_p: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.8,
    repetition_penalty: Annotated[float, typer.Option(min=0.1)] = 1.1,
    seed: int = 42,
) -> None:
    """Rewrite a neutral draft in the locally trained casual text style."""
    typer.echo(
        generate_rewrite(
            resolve_path(model),
            neutral_draft,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )
    )


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
