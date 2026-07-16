from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from imessage_mlx.config import load_yaml
from imessage_mlx.seq2seq_profile import SEQ2SEQ_ARCHITECTURES
from imessage_mlx.utils import (
    atomic_write_text,
    ensure_private_dir,
    sha256_file,
    write_json,
)

BART_PACKAGES = (
    "torch==2.13.0",
    "transformers==4.57.6",
    "peft==0.19.1",
    "sentence-transformers==5.6.0",
    "sentencepiece==0.2.2",
    "sacremoses==0.1.1",
)
QWEN_PACKAGES = ("mlx-lm[train]==0.31.3",)
ADAPTER_ARCHITECTURES = SEQ2SEQ_ARCHITECTURES | {"qwen"}


def _run(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "DO_NOT_TRACK": "1",
            "WANDB_DISABLED": "true",
            "PYTHONPATH": str(cwd / "src"),
        }
    )
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if log_path is not None:
        atomic_write_text(
            log_path,
            completed.stdout + ("\n" if completed.stdout else "") + completed.stderr,
        )
    if completed.returncode:
        raise RuntimeError(
            f"Adapter command failed with exit code {completed.returncode}; "
            f"see {log_path or 'captured output'}"
        )
    return completed


def environment_python(environment_dir: str | Path) -> Path:
    return Path(environment_dir) / "bin/python"


def _pinned_huggingface_snapshot(
    model_id: str,
    revision: str,
    python: Path,
    root: Path,
) -> Path:
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    snapshot = hub / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
    if not snapshot.exists():
        completed = _run(
            [
                str(python),
                "-c",
                (
                    "from huggingface_hub import snapshot_download; "
                    "import sys; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))"
                ),
                model_id,
                revision,
            ],
            cwd=root,
        )
        snapshot = Path(completed.stdout.strip())
    return snapshot


def setup_adapter_environment(
    architecture: str,
    environment_dir: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    if architecture not in ADAPTER_ARCHITECTURES:
        raise ValueError(
            "Adapter architecture must be one of: " + ", ".join(sorted(ADAPTER_ARCHITECTURES))
        )
    root = Path(project_root or Path.cwd()).resolve()
    destination = Path(environment_dir).resolve()
    ensure_private_dir(destination.parent)
    _run(
        [
            "uv",
            "venv",
            "--allow-existing",
            "--python",
            "3.11",
            str(destination),
        ],
        cwd=root,
    )
    packages = BART_PACKAGES if architecture in SEQ2SEQ_ARCHITECTURES else QWEN_PACKAGES
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(environment_python(destination)),
            *packages,
        ],
        cwd=root,
        log_path=destination / "install.log",
    )
    destination.chmod(0o700)
    report = {
        "schema_version": 1,
        "architecture": architecture,
        "python": str(environment_python(destination)),
        "packages": list(packages),
        "private_local_environment": True,
    }
    write_json(destination / "environment-report.json", report)
    return report


def train_adapter(
    config_path: str | Path,
    data_root: str | Path,
    output_dir: str | Path,
    environment_dir: str | Path,
    *,
    benchmark: bool = False,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root or Path.cwd()).resolve()
    config_file = Path(config_path).resolve()
    config = load_yaml(config_file)
    architecture = str(config.get("architecture"))
    if architecture not in ADAPTER_ARCHITECTURES:
        raise ValueError(
            "Adapter config architecture must be one of: "
            + ", ".join(sorted(ADAPTER_ARCHITECTURES))
        )
    python = Path(environment_dir).resolve() / "bin/python"
    if not python.exists():
        raise FileNotFoundError(f"Adapter environment is missing {python}")
    output = ensure_private_dir(output_dir)
    data = Path(data_root).resolve()
    if benchmark:
        data = data / "benchmark"
    started = time.perf_counter()

    if architecture in SEQ2SEQ_ARCHITECTURES:
        command = [
            str(python),
            "-m",
            "imessage_mlx.adapter_worker",
            "train-bart" if architecture == "bart" else "train-seq2seq",
            "--config",
            str(config_file),
            "--data",
            str(data / "bart"),
            "--output",
            str(output),
        ]
    else:
        adapter_path = ensure_private_dir(output / "adapter")
        iterations = int(config.get("iterations", 1200))
        if benchmark:
            iterations = min(iterations, 100)
        model_reference = _pinned_huggingface_snapshot(
            str(config["base_model"]),
            str(config["revision"]),
            python,
            root,
        )
        command = [
            str(python),
            "-m",
            "mlx_lm.lora",
            "--model",
            str(model_reference),
            "--train",
            "--data",
            str(data / "mlx"),
            "--adapter-path",
            str(adapter_path),
            "--iters",
            str(iterations),
            "--batch-size",
            str(config.get("batch_size", 1)),
            "--num-layers",
            str(config.get("num_layers", 16)),
            "--learning-rate",
            str(config.get("learning_rate", 1e-4)),
            "--grad-accumulation-steps",
            str(config.get("gradient_accumulation_steps", 8)),
            "--max-seq-length",
            str(config.get("max_length", 256)),
            "--val-batches",
            "-1",
            "--mask-prompt",
        ]
    completed = _run(command, cwd=root, log_path=output / "training.log")
    elapsed = time.perf_counter() - started
    report_path = output / "training-report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {
            "schema_version": 1,
            "architecture": architecture,
            "base_model": str(config["base_model"]),
            "base_revision": str(config["revision"]),
            "base_model_license": config.get("base_model_license"),
            "source_prefix": config.get("source_prefix", ""),
            "lora_target_modules": config.get("lora_target_modules"),
            "elapsed_seconds": elapsed,
            "benchmark": benchmark,
            "private_local_artifact": True,
            "stdout_tail": completed.stdout[-2_000:],
        }
        write_json(report_path, report)
    for path in output.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)
    output.chmod(0o700)
    return report


def predict_adapter(
    config_path: str | Path,
    data_root: str | Path,
    adapter_dir: str | Path,
    output_path: str | Path,
    environment_dir: str | Path,
    *,
    benchmark: bool = False,
    data_file: str | Path | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root or Path.cwd()).resolve()
    config_file = Path(config_path).resolve()
    config = load_yaml(config_file)
    architecture = str(config.get("architecture"))
    if architecture not in ADAPTER_ARCHITECTURES:
        raise ValueError(
            "Adapter config architecture must be one of: "
            + ", ".join(sorted(ADAPTER_ARCHITECTURES))
        )
    python = Path(environment_dir).resolve() / "bin/python"
    if not python.exists():
        raise FileNotFoundError(f"Adapter environment is missing {python}")
    data = Path(data_root).resolve()
    if benchmark:
        data = data / "benchmark"
    family = "bart" if architecture in SEQ2SEQ_ARCHITECTURES else "mlx"
    prediction_data = (
        Path(data_file).resolve() if data_file is not None else data / family / "test.jsonl"
    )
    command = [
        str(python),
        "-m",
        "imessage_mlx.adapter_worker",
        (
            "predict-bart"
            if architecture == "bart"
            else "predict-seq2seq"
            if architecture in SEQ2SEQ_ARCHITECTURES
            else "predict-qwen"
        ),
        "--config",
        str(config_file),
        "--data",
        str(prediction_data),
        "--adapter",
        str(Path(adapter_dir).resolve() / ("adapter" if architecture == "qwen" else "")),
        "--output",
        str(Path(output_path).resolve()),
    ]
    completed = _run(
        command,
        cwd=root,
        log_path=Path(output_path).with_suffix(".log"),
    )
    return {
        "architecture": architecture,
        "output": str(Path(output_path).resolve()),
        "worker_output": completed.stdout.strip(),
    }


def evaluate_semantics(
    predictions_path: str | Path,
    output_path: str | Path,
    environment_dir: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root or Path.cwd()).resolve()
    python = Path(environment_dir).resolve() / "bin/python"
    command = [
        str(python),
        "-m",
        "imessage_mlx.adapter_worker",
        "semantic-eval",
        "--data",
        str(Path(predictions_path).resolve()),
        "--output",
        str(Path(output_path).resolve()),
    ]
    _run(
        command,
        cwd=root,
        log_path=Path(output_path).with_suffix(".log"),
    )
    return json.loads(Path(output_path).read_text(encoding="utf-8"))


def validate_convergence_data_semantics(
    pairs_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    environment_dir: str | Path,
    *,
    minimum_similarity: float = 0.70,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root or Path.cwd()).resolve()
    python = Path(environment_dir).resolve() / "bin/python"
    command = [
        str(python),
        "-m",
        "imessage_mlx.adapter_worker",
        "convergence-data-semantics",
        "--data",
        str(Path(pairs_path).resolve()),
        "--output",
        str(Path(output_path).resolve()),
        "--report",
        str(Path(report_path).resolve()),
        "--minimum",
        str(minimum_similarity),
    ]
    _run(
        command,
        cwd=root,
        log_path=Path(report_path).with_suffix(".log"),
    )
    return json.loads(Path(report_path).read_text(encoding="utf-8"))


def repair_pairs_locally(
    config_path: str | Path,
    pairs_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    environment_dir: str | Path,
    *,
    limit: int = 500,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root or Path.cwd()).resolve()
    python = Path(environment_dir).resolve() / "bin/python"
    command = [
        str(python),
        "-m",
        "imessage_mlx.adapter_worker",
        "repair-qwen",
        "--config",
        str(Path(config_path).resolve()),
        "--data",
        str(Path(pairs_path).resolve()),
        "--output",
        str(Path(output_path).resolve()),
        "--report",
        str(Path(report_path).resolve()),
        "--limit",
        str(limit),
    ]
    _run(
        command,
        cwd=root,
        log_path=Path(report_path).with_suffix(".log"),
    )
    return json.loads(Path(report_path).read_text(encoding="utf-8"))


def rewrite_with_adapter(
    draft: str,
    config_path: str | Path,
    adapter_dir: str | Path,
    environment_dir: str | Path,
    *,
    project_root: str | Path | None = None,
) -> str:
    if not draft.strip():
        raise ValueError("Rewrite draft cannot be empty")
    root = Path(project_root or Path.cwd()).resolve()
    config_file = Path(config_path).resolve()
    config = load_yaml(config_file)
    architecture = str(config.get("architecture"))
    if architecture not in ADAPTER_ARCHITECTURES:
        raise ValueError(
            "Adapter config architecture must be one of: "
            + ", ".join(sorted(ADAPTER_ARCHITECTURES))
        )
    run = Path(adapter_dir).resolve()
    manifest_path = run / "data-manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "rewrite" not in manifest.get("capabilities", []):
            raise ValueError("Adapter artifact was not promoted for rewrite generation")
        run = run / "adapter"
    python = Path(environment_dir).resolve() / "bin/python"
    completed = _run(
        [
            str(python),
            "-m",
            "imessage_mlx.adapter_worker",
            (
                "rewrite-bart"
                if architecture == "bart"
                else "rewrite-seq2seq"
                if architecture in SEQ2SEQ_ARCHITECTURES
                else "rewrite-qwen"
            ),
            "--config",
            str(config_file),
            "--adapter",
            str(run),
            "--draft",
            draft,
        ],
        cwd=root,
    )
    return completed.stdout.strip()


def promote_adapter(
    adapter_dir: str | Path,
    evaluation_report_path: str | Path,
    destination_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    data_report_path: str | Path | None = None,
    architecture_report_path: str | Path | None = None,
    review_summary_path: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(adapter_dir)
    evaluation_path = Path(evaluation_report_path)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if not evaluation.get("ready_to_promote", False):
        raise ValueError("Adapter evaluation did not pass the promotion gate")
    if evaluation.get("semantic") is None:
        raise ValueError("Adapter promotion requires local semantic evaluation evidence")
    if review_summary_path is None:
        raise ValueError("Adapter promotion requires a completed human review summary")
    review_path = Path(review_summary_path)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("human_approved") is not True:
        raise ValueError("Human review summary is not approved")
    reviewed_groups = review.get("reviewed_target_groups")
    if (
        isinstance(reviewed_groups, bool)
        or not isinstance(reviewed_groups, int)
        or reviewed_groups < 20
    ):
        raise ValueError("Human review must cover at least 20 target groups")
    if review.get("changed_fact_failures") != 0:
        raise ValueError("Human review found changed facts or lacks a zero-failure attestation")
    destination = Path(destination_dir)
    ensure_private_dir(destination.parent)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    rollback: Path | None = None
    try:
        shutil.copytree(source, temporary / "adapter", dirs_exist_ok=True)
        shutil.copy2(evaluation_path, temporary / "evaluation.json")
        shutil.copy2(review_path, temporary / "human-review-summary.json")
        training_report_path = source / "training-report.json"
        training_report = (
            json.loads(training_report_path.read_text(encoding="utf-8"))
            if training_report_path.exists()
            else {}
        )
        adapter_config = load_yaml(config_path) if config_path is not None else {}
        if config_path is not None:
            shutil.copy2(config_path, temporary / "adapter-config.yaml")
        if data_report_path is not None:
            shutil.copy2(data_report_path, temporary / "adapter-data-report.json")
        if architecture_report_path is not None:
            shutil.copy2(
                architecture_report_path,
                temporary / "architecture-selection.json",
            )
        architecture = str(
            training_report.get(
                "architecture",
                adapter_config.get("architecture", "unknown"),
            )
        )
        generation = {
            "deterministic": True,
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": 64,
            "task": "rewrite",
        }
        write_json(temporary / "generation-config.json", generation)
        atomic_write_text(
            temporary / "README.md",
            "# Private local rewrite adapter\n\n"
            "This artifact contains a personal style adapter. Keep it local, review every "
            "output, and never connect it to automatic message sending.\n\n"
            "The base model is not bundled. Inference is deterministic by default. The adapter "
            "must not be fused, uploaded, or shared.\n",
        )
        manifest = {
            "schema_version": 1,
            "task": "rewrite",
            "capabilities": ["rewrite"],
            "private_local_artifact": True,
            "pretrained_weights_used": True,
            "pretrained_weights_bundled": False,
            "adapter_fused": False,
            "architecture": architecture,
            "base_model": training_report.get(
                "base_model",
                adapter_config.get("base_model"),
            ),
            "base_revision": training_report.get(
                "base_revision",
                adapter_config.get("revision"),
            ),
            "source_prefix": training_report.get(
                "source_prefix",
                adapter_config.get("source_prefix", ""),
            ),
            "lora_target_modules": training_report.get(
                "lora_target_modules",
                adapter_config.get("lora_target_modules"),
            ),
            "training_seed": adapter_config.get("seed"),
            "adapter_config_sha256": (
                sha256_file(config_path) if config_path is not None else None
            ),
            "adapter_data_report_sha256": (
                sha256_file(data_report_path) if data_report_path is not None else None
            ),
            "architecture_selection_sha256": (
                sha256_file(architecture_report_path)
                if architecture_report_path is not None
                else None
            ),
            "base_model_license": training_report.get(
                "base_model_license",
                adapter_config.get(
                    "base_model_license",
                    "not_declared_in_hugging_face_model_card" if architecture == "bart" else None,
                ),
            ),
            "deterministic_generation": generation,
            "no_automatic_sending": True,
            "upload_permitted": False,
            "adapter_files": {
                str(path.relative_to(temporary)): sha256_file(path)
                for path in temporary.rglob("*")
                if path.is_file()
            },
            "evaluation_ready_to_promote": True,
            "human_review_approved": True,
            "human_reviewed_target_groups": reviewed_groups,
            "human_review_summary_sha256": sha256_file(review_path),
        }
        write_json(temporary / "data-manifest.json", manifest)
        for path in temporary.rglob("*"):
            path.chmod(0o700 if path.is_dir() else 0o600)
        if destination.exists():
            suffix = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            rollback = destination.with_name(f"{destination.name}-rollback-{suffix}")
            counter = 1
            while rollback.exists():
                rollback = destination.with_name(f"{destination.name}-rollback-{suffix}-{counter}")
                counter += 1
            os.replace(destination, rollback)
        try:
            os.replace(temporary, destination)
        except Exception:
            if rollback is not None and rollback.exists() and not destination.exists():
                os.replace(rollback, destination)
            raise
        destination.chmod(0o700)
        manifest["rollback_artifact"] = str(rollback) if rollback is not None else None
        write_json(destination / "data-manifest.json", manifest)
        return manifest
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
