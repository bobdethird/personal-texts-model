from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from imessage_mlx.utils import ensure_private_dir, sha256_file, write_json


def export_model(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    metrics_path: str | Path | None = None,
    split_report_path: str | Path | None = None,
    split_dir: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint)
    destination = Path(output_dir)
    ensure_private_dir(destination.parent)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    temporary.chmod(0o700)
    try:
        for name in ("model.safetensors", "model-config.json"):
            shutil.copy2(checkpoint_path / name, temporary / name)
        shutil.copytree(checkpoint_path / "tokenizer", temporary / "tokenizer")
        generation_config = {
            "max_new_tokens": 64,
            "temperature": 0.8,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "stop_tokens": ["<|turn_end|>", "<|eos|>"],
        }
        write_json(temporary / "generation-config.json", generation_config)
        metrics: dict[str, Any] = {}
        if metrics_path is not None and Path(metrics_path).exists():
            metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        write_json(temporary / "metrics.json", metrics)
        manifest: dict[str, Any] = {
            "model_sha256": sha256_file(temporary / "model.safetensors"),
            "private_local_artifact": True,
            "pretrained_weights_used": False,
            "pretrained_tokenizer_used": False,
        }
        if split_report_path is not None and Path(split_report_path).exists():
            manifest["split_report"] = json.loads(
                Path(split_report_path).read_text(encoding="utf-8")
            )
        if split_dir is not None:
            manifest["split_hashes"] = {
                name: sha256_file(Path(split_dir) / f"{name}.jsonl")
                for name in ("train", "validation", "test")
            }
        write_json(temporary / "data-manifest.json", manifest)
        readme = (
            "# Private local iMessage model\n\n"
            "This artifact was trained from random initialization and is sensitive. "
            "Keep it local.\n\n"
            "Run from the project root:\n\n"
            "```bash\nuv run imessage-mlx chat --model outputs/final\n```\n"
        )
        (temporary / "README.md").write_text(readme, encoding="utf-8")
        (temporary / "README.md").chmod(0o600)
        for path in temporary.rglob("*"):
            path.chmod(0o700 if path.is_dir() else 0o600)
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary, destination)
        destination.chmod(0o700)
        return manifest
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
