from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

# Resolve imessage_mlx straight from src/ so Modal's source packaging never
# depends on the venv's editable .pth file (iCloud Drive re-flags it hidden,
# which makes Python skip it and Modal fail with "imessage_mlx has no spec").
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

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
        "sentence-transformers==5.7.0",
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
    # Preemption/timeout should restart immediately into a fresh container and
    # pick up the last committed checkpoint (resume defaults to True).
    retries=modal.Retries(initial_delay=0.0, max_retries=10),
    single_use_containers=True,
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
    # Committing on every checkpoint save means a preempted container loses at
    # most save_steps worth of progress; the restart resumes from the last
    # committed checkpoint (https://modal.com/docs/guide/preemption).
    result = run_training(config, on_checkpoint_saved=artifact_volume.commit)
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
    resume: bool = True,
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
    # spawn().get() (not .remote()) so the FunctionCall survives beyond 24h and
    # works with `modal run --detach` for long GPU jobs.
    # https://modal.com/docs/examples/long-training
    call = train.spawn(config)
    print(f"Spawned training FunctionCall: {call.object_id}")
    print(
        "Artifacts will land in the imessage-sft-artifacts Volume at "
        f"/{run_name}/final"
    )
    print("Watch logs with: modal app logs -f")
    result = call.get()
    print(json.dumps(result, indent=2, sort_keys=True))


@app.function(
    image=training_image,
    gpu="A100-80GB",
    cpu=4,
    memory=24_576,
    timeout=30 * 60,
    startup_timeout=20 * 60,
    volumes={
        "/data": data_volume,
        "/outputs": artifact_volume,
        "/cache": model_cache,
    },
)
def sample(config: dict[str, Any]) -> dict[str, Any]:
    from imessage_mlx.sampling import run_sampling

    data_volume.reload()
    artifact_volume.reload()
    model_cache.reload()
    result = run_sampling(config)
    artifact_volume.commit()
    model_cache.commit()
    return result


@app.local_entrypoint()
def sample_main(
    run_name: str = "personal-qwen",
    model_name: str = DEFAULT_MODEL,
    samples: int = 8,
    temperature: float = 0.7,
    max_new_tokens: int = 128,
    min_context_turns: int = 1,
    seed: int = 42,
) -> None:
    """Generate held-out next-message samples from a trained adapter."""
    run_name = _safe_run_name(run_name)
    config = {
        "model_name": model_name,
        "validation_path": f"/data/{run_name}/validation.jsonl",
        "adapter_dir": f"/outputs/{run_name}/final",
        "output_path": f"/outputs/{run_name}/sample-examples.json",
        "samples": samples,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "min_context_turns": min_context_turns,
        "seed": seed,
    }
    result = sample.remote(config)
    # Drop bulky fields when printing to the local terminal; full JSON is on the volume.
    printable = {
        key: value
        for key, value in result.items()
        if key != "examples"
    }
    printable["example_ids"] = [example["example_id"] for example in result["examples"]]
    print(json.dumps(printable, indent=2, sort_keys=True))
    print(json.dumps(result["examples"], indent=2, ensure_ascii=False))


@app.function(
    image=training_image,
    gpu="A100-80GB",
    cpu=4,
    memory=24_576,
    timeout=30 * 60,
    startup_timeout=20 * 60,
    volumes={
        "/data": data_volume,
        "/outputs": artifact_volume,
        "/cache": model_cache,
    },
)
def induce_style(config: dict[str, Any]) -> dict[str, Any]:
    from imessage_mlx.style_card import run_style_card

    data_volume.reload()
    artifact_volume.reload()
    model_cache.reload()
    result = run_style_card(config)
    artifact_volume.commit()
    model_cache.commit()
    return result


@app.local_entrypoint()
def style_card_main(
    pairs_path: str = "work/imessages/pairs/train.jsonl",
    run_name: str = "personal-qwen",
    model_name: str = DEFAULT_MODEL,
    samples: int = 128,
    max_source_chars: int = 24_000,
    max_new_tokens: int = 768,
    temperature: float = 0.2,
    seed: int = 42,
    force: bool = False,
) -> None:
    """Upload private pairs and induce a cached style card on an A100."""
    run_name = _safe_run_name(run_name)
    local_pairs = Path(pairs_path).expanduser().resolve()
    if not local_pairs.is_file():
        raise FileNotFoundError(f"Missing prepared training pairs: {local_pairs}")
    remote_input = f"/{run_name}/personalization/train-pairs.jsonl"
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(local_pairs, remote_input)

    output_path = f"/outputs/{run_name}/personalization/style-card.md"
    result = induce_style.remote(
        {
            "pairs_path": f"/data{remote_input}",
            "output_path": output_path,
            "report_path": f"/outputs/{run_name}/personalization/style-card.json",
            "model_name": model_name,
            "samples": samples,
            "max_source_chars": max_source_chars,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "seed": seed,
            "force": force,
        }
    )
    printable = {key: value for key, value in result.items() if key != "style_card"}
    print(json.dumps(printable, indent=2, sort_keys=True))
    print(result["style_card"])


@app.function(
    image=training_image,
    gpu="A100-80GB",
    cpu=4,
    memory=24_576,
    timeout=60 * 60,
    startup_timeout=20 * 60,
    volumes={
        "/data": data_volume,
        "/outputs": artifact_volume,
        "/cache": model_cache,
    },
)
def personalize(config: dict[str, Any]) -> dict[str, Any]:
    from imessage_mlx.sampling import run_personalization_sampling

    data_volume.reload()
    artifact_volume.reload()
    model_cache.reload()
    result = run_personalization_sampling(config)
    artifact_volume.commit()
    model_cache.commit()
    return result


@app.local_entrypoint()
def personalize_main(
    dataset_dir: str = "work/imessages/sft",
    index_dir: str = "work/imessages/retrieval",
    style_card_path: str = "",
    run_name: str = "personal-qwen",
    model_name: str = DEFAULT_MODEL,
    samples: int = 8,
    top_k: int = 4,
    temperature: float = 0.7,
    max_new_tokens: int = 128,
    min_context_turns: int = 1,
    seed: int = 42,
) -> None:
    """Compare base, retrieval, and retrieval-plus-style on held-out messages."""
    run_name = _safe_run_name(run_name)
    validation_path = Path(dataset_dir).expanduser().resolve() / "validation.jsonl"
    local_index = Path(index_dir).expanduser().resolve()
    required_index_files = ("manifest.json", "metadata.jsonl", "embeddings.npy")
    if not validation_path.is_file():
        raise FileNotFoundError(f"Missing validation data: {validation_path}")
    for filename in required_index_files:
        if not (local_index / filename).is_file():
            raise FileNotFoundError(f"Missing retrieval index file: {local_index / filename}")

    remote_dir = f"/{run_name}/personalization"
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(validation_path, f"{remote_dir}/validation.jsonl")
        for filename in required_index_files:
            batch.put_file(local_index / filename, f"{remote_dir}/retrieval/{filename}")
        if style_card_path:
            local_style = Path(style_card_path).expanduser().resolve()
            if not local_style.is_file():
                raise FileNotFoundError(f"Missing style card: {local_style}")
            batch.put_file(local_style, f"{remote_dir}/style-card.md")
            remote_style_path = f"/data{remote_dir}/style-card.md"
        else:
            remote_style_path = f"/outputs/{run_name}/personalization/style-card.md"

    result = personalize.remote(
        {
            "model_name": model_name,
            "validation_path": f"/data{remote_dir}/validation.jsonl",
            "index_dir": f"/data{remote_dir}/retrieval",
            "style_card_path": remote_style_path,
            "output_path": (f"/outputs/{run_name}/personalization/sample-examples.json"),
            "samples": samples,
            "top_k": top_k,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "min_context_turns": min_context_turns,
            "seed": seed,
        }
    )
    printable = {key: value for key, value in result.items() if key != "examples"}
    printable["example_ids"] = [example["example_id"] for example in result["examples"]]
    print(json.dumps(printable, indent=2, sort_keys=True))
    print(json.dumps(result["examples"], indent=2, ensure_ascii=False))
