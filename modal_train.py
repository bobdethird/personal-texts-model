from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

APP_NAME = "imessage-next-message-sft"
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name("imessage-sft-data", create_if_missing=True)
artifact_volume = modal.Volume.from_name("imessage-sft-artifacts", create_if_missing=True)
model_cache = modal.Volume.from_name("imessage-sft-model-cache", create_if_missing=True)

training_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "accelerate==1.14.0",
        "datasets==5.0.0",
        "peft==0.19.1",
        "torch==2.13.0",
        "transformers==5.14.1",
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source("imessage_mlx")
)


@app.function(
    image=training_image,
    gpu="A100-80GB",
    cpu=8,
    memory=32_768,
    timeout=24 * 60 * 60,
    startup_timeout=20 * 60,
    volumes={
        "/data": data_volume,
        "/outputs": artifact_volume,
        "/cache": model_cache,
    },
)
def train(config: dict[str, Any]) -> dict[str, Any]:
    from imessage_mlx.training import run_training

    data_volume.reload()
    artifact_volume.reload()
    model_cache.reload()
    result = run_training(config)
    artifact_volume.commit()
    model_cache.commit()
    return result


def _safe_run_name(value: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip(".-")
    if not name:
        raise ValueError("run_name must contain at least one letter or number")
    return name


@app.local_entrypoint()
def main(
    dataset_dir: str = "work/imessages/sft",
    run_name: str = "",
    model_name: str = DEFAULT_MODEL,
    max_length: int = 4096,
    epochs: float = 3.0,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.01,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    logging_steps: int = 10,
    save_steps: int = 100,
    save_total_limit: int = 3,
    seed: int = 42,
    resume: bool = False,
) -> None:
    """Upload a prepared private dataset and train a masked LoRA on an A100."""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    train_path = dataset_dir / "train.jsonl"
    validation_path = dataset_dir / "validation.jsonl"
    if not train_path.is_file():
        raise FileNotFoundError(f"Missing prepared training data: {train_path}")
    if not validation_path.is_file():
        raise FileNotFoundError(f"Missing prepared validation data: {validation_path}")

    if not run_name:
        run_name = datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S")
    run_name = _safe_run_name(run_name)
    remote_dataset_dir = f"/{run_name}"
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(train_path, f"{remote_dataset_dir}/train.jsonl")
        batch.put_file(validation_path, f"{remote_dataset_dir}/validation.jsonl")
        report_path = dataset_dir / "report.json"
        if report_path.is_file():
            batch.put_file(report_path, f"{remote_dataset_dir}/report.json")

    config = {
        "run_name": run_name,
        "model_name": model_name,
        "train_path": f"/data/{run_name}/train.jsonl",
        "validation_path": f"/data/{run_name}/validation.jsonl",
        "output_dir": f"/outputs/{run_name}",
        "max_length": max_length,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "logging_steps": logging_steps,
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        "seed": seed,
        "resume": resume,
    }
    result = train.remote(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(
        "Artifacts are in the imessage-sft-artifacts Volume at "
        f"/{run_name}/final"
    )
