from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import mlx.core as mx
import yaml
from mlx.utils import tree_flatten, tree_unflatten

from imessage_mlx.model.config import ModelConfig
from imessage_mlx.model.transformer import TransformerLM
from imessage_mlx.utils import ensure_private_dir, write_json


def _save_tree(path: Path, value: Any) -> None:
    flattened = dict(tree_flatten(value))
    mx.save_safetensors(str(path), flattened, metadata={"format": "mlx"})
    path.chmod(0o600)


def _load_tree(path: Path) -> Any:
    values = mx.load(str(path))
    return tree_unflatten(list(values.items()))


def save_checkpoint(
    destination: str | Path,
    model: TransformerLM,
    optimizer: Any,
    trainer_state: dict[str, Any],
    training_config: dict[str, Any],
    tokenizer_dir: str | Path | None = None,
) -> Path:
    destination_path = Path(destination)
    ensure_private_dir(destination_path.parent)
    temporary_path = Path(
        tempfile.mkdtemp(prefix=f".{destination_path.name}.", dir=destination_path.parent)
    )
    temporary_path.chmod(0o700)
    try:
        model.save_weights(str(temporary_path / "model.safetensors"))
        (temporary_path / "model.safetensors").chmod(0o600)
        _save_tree(temporary_path / "optimizer.safetensors", optimizer.state)
        _save_tree(temporary_path / "random-state.safetensors", mx.random.state)
        write_json(temporary_path / "model-config.json", model.config.to_dict())
        write_json(temporary_path / "training-config.json", training_config)
        (temporary_path / "training-config.yaml").write_text(
            yaml.safe_dump(training_config, sort_keys=True), encoding="utf-8"
        )
        (temporary_path / "training-config.yaml").chmod(0o600)
        write_json(temporary_path / "trainer-state.json", trainer_state)
        if tokenizer_dir is not None:
            shutil.copytree(tokenizer_dir, temporary_path / "tokenizer")
            for path in (temporary_path / "tokenizer").rglob("*"):
                path.chmod(0o700 if path.is_dir() else 0o600)
        if destination_path.exists():
            shutil.rmtree(destination_path)
        os.replace(temporary_path, destination_path)
        destination_path.chmod(0o700)
        return destination_path
    finally:
        shutil.rmtree(temporary_path, ignore_errors=True)


def load_model(checkpoint: str | Path) -> TransformerLM:
    import json

    checkpoint_path = Path(checkpoint)
    with (checkpoint_path / "model-config.json").open(encoding="utf-8") as handle:
        config = ModelConfig.from_dict(json.load(handle))
    model = TransformerLM(config)
    model.load_weights(str(checkpoint_path / "model.safetensors"), strict=True)
    mx.eval(model.parameters())
    model.eval()
    return model


def restore_training_state(
    checkpoint: str | Path, model: TransformerLM, optimizer: Any
) -> dict[str, Any]:
    import json

    checkpoint_path = Path(checkpoint)
    model.load_weights(str(checkpoint_path / "model.safetensors"), strict=True)
    optimizer.init(model.trainable_parameters())
    optimizer.state = _load_tree(checkpoint_path / "optimizer.safetensors")
    random_state_path = checkpoint_path / "random-state.safetensors"
    if random_state_path.exists():
        mx.random.state = _load_tree(random_state_path)
    mx.eval(model.parameters(), optimizer.state)
    with (checkpoint_path / "trainer-state.json").open(encoding="utf-8") as handle:
        return json.load(handle)
