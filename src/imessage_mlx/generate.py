from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import mlx.core as mx
import numpy as np

from imessage_mlx.checkpoint import load_model
from imessage_mlx.tokenizer.train import load_tokenizer


def format_reply_prompt(other_message: str, history: list[tuple[str, str]] | None = None) -> str:
    lines = ["<|bos|><|conversation|>"]
    for role, text in history or []:
        token = "<|me|>" if role == "me" else "<|other|>"
        lines.append(f"{token}{text}<|turn_end|>")
    lines.append(f"<|other|>{other_message}<|turn_end|>")
    lines.append("<|me|>")
    return "\n".join(lines)


def _model_capabilities(directory: Path) -> set[str]:
    manifest_path = directory / "data-manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        capabilities = manifest.get("capabilities")
        if isinstance(capabilities, list):
            return {str(value) for value in capabilities}
    training_config_path = directory / "training-config.json"
    if training_config_path.exists():
        config = json.loads(training_config_path.read_text(encoding="utf-8"))
        return {str(config.get("task", "reply"))}
    return {"reply"}


def _load_reply_model(directory: Path):
    if "reply" not in _model_capabilities(directory):
        raise ValueError("Model artifact was not trained for reply generation")
    return load_model(directory), load_tokenizer(directory / "tokenizer")


def _sample(
    logits: np.ndarray,
    generated: list[int],
    generator: np.random.Generator,
    *,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> int:
    scores = logits.astype(np.float64, copy=True)
    if repetition_penalty != 1.0:
        for token_id in set(generated):
            scores[token_id] = (
                scores[token_id] / repetition_penalty
                if scores[token_id] > 0
                else scores[token_id] * repetition_penalty
            )
    if temperature <= 0:
        return int(np.argmax(scores))
    scores /= temperature
    scores -= scores.max()
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum()
    if top_p < 1.0:
        order = np.argsort(probabilities)[::-1]
        cumulative = np.cumsum(probabilities[order])
        keep_count = max(1, int(np.searchsorted(cumulative, top_p, side="left")) + 1)
        keep = order[:keep_count]
        filtered = np.zeros_like(probabilities)
        filtered[keep] = probabilities[keep]
        probabilities = filtered / filtered.sum()
    return int(generator.choice(len(probabilities), p=probabilities))


def iter_generated_ids(
    model,
    prompt_ids: list[int],
    *,
    eos_ids: set[int],
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    seed: int = 42,
) -> Iterator[int]:
    generator = np.random.default_rng(seed)
    all_ids = list(prompt_ids)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        context = all_ids[-model.config.max_sequence_length :]
        logits = model(mx.array([context], dtype=mx.int32))[:, -1, :]
        mx.eval(logits)
        token_id = _sample(
            np.asarray(logits)[0],
            generated,
            generator,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        if token_id in eos_ids:
            break
        generated.append(token_id)
        all_ids.append(token_id)
        yield token_id


def generate_ids(
    model,
    prompt_ids: list[int],
    *,
    eos_ids: set[int],
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    seed: int = 42,
) -> list[int]:
    return list(
        iter_generated_ids(
            model,
            prompt_ids,
            eos_ids=eos_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )
    )


def generate_reply(
    model_dir: str | Path,
    other_message: str,
    *,
    history: list[tuple[str, str]] | None = None,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    seed: int = 42,
) -> str:
    directory = Path(model_dir)
    model, tokenizer = _load_reply_model(directory)
    prompt = format_reply_prompt(other_message, history)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    eos_ids = {
        token_id
        for token_id in (
            tokenizer.token_to_id("<|eos|>"),
            tokenizer.token_to_id("<|turn_end|>"),
        )
        if token_id is not None
    }
    generated = generate_ids(
        model,
        prompt_ids,
        eos_ids=eos_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        seed=seed,
    )
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def stream_reply(
    model_dir: str | Path,
    other_message: str,
    *,
    history: list[tuple[str, str]] | None = None,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    seed: int = 42,
) -> Iterator[str]:
    directory = Path(model_dir)
    model, tokenizer = _load_reply_model(directory)
    prompt_ids = tokenizer.encode(
        format_reply_prompt(other_message, history), add_special_tokens=False
    ).ids
    eos_ids = {
        token_id
        for token_id in (
            tokenizer.token_to_id("<|eos|>"),
            tokenizer.token_to_id("<|turn_end|>"),
        )
        if token_id is not None
    }
    generated: list[int] = []
    emitted = ""
    for token_id in iter_generated_ids(
        model,
        prompt_ids,
        eos_ids=eos_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        seed=seed,
    ):
        generated.append(token_id)
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        delta = decoded[len(emitted) :] if decoded.startswith(emitted) else decoded
        emitted = decoded
        if delta:
            yield delta
