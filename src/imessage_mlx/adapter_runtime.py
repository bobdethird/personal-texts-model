from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from imessage_mlx.config import load_yaml
from imessage_mlx.seq2seq_profile import SEQ2SEQ_ARCHITECTURES
from imessage_mlx.utils import (
    atomic_write_text,
    ensure_private_dir,
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
