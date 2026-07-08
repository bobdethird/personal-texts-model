from __future__ import annotations

import json
import math
import platform
import time
from functools import partial
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from imessage_mlx import __version__
from imessage_mlx.checkpoint import restore_training_state, save_checkpoint
from imessage_mlx.dataset import batch_indices, load_tokens, materialize_batch, window_count
from imessage_mlx.model.config import ModelConfig
from imessage_mlx.model.transformer import TransformerLM, causal_lm_loss, perplexity
from imessage_mlx.tokenizer.train import load_tokenizer
from imessage_mlx.utils import ensure_private_dir, write_json


def learning_rate_at_step(
    step: int, total_steps: int, warmup_steps: int, peak: float, minimum: float
) -> float:
    if warmup_steps and step < warmup_steps:
        return peak * max(1, step + 1) / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum + (peak - minimum) * cosine


def evaluate_loss(
    model: TransformerLM,
    tokens: np.ndarray,
    *,
    context_length: int,
    batch_size: int,
) -> float:
    model.eval()
    count = window_count(tokens, context_length)
    if not count:
        raise ValueError("Evaluation split has no complete context window")
    total_loss = 0.0
    total_examples = 0
    for start in range(0, count, batch_size):
        indices = np.arange(start, min(count, start + batch_size))
        inputs_np, targets_np = materialize_batch(tokens, indices, context_length)
        loss = causal_lm_loss(model, mx.array(inputs_np), mx.array(targets_np))
        mx.eval(loss)
        total_loss += float(loss.item()) * len(indices)
        total_examples += len(indices)
    return total_loss / total_examples


def _append_metric(path: Path, metric: dict[str, Any]) -> None:
    ensure_private_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metric, sort_keys=True) + "\n")
    path.chmod(0o600)


def train_model(
    training_config: dict[str, Any],
    data_dir: str | Path,
    tokenizer_dir: str | Path,
    output_dir: str | Path,
    *,
    resume_from: str | Path | None = None,
    compile_step: bool = True,
) -> dict[str, Any]:
    output = ensure_private_dir(output_dir)
    data_path = Path(data_dir)
    tokenizer = load_tokenizer(tokenizer_dir)
    actual_vocab_size = tokenizer.get_vocab_size()
    config_value = dict(training_config)
    config_value["vocab_size"] = actual_vocab_size
    model_config = ModelConfig.from_dict(config_value)
    context_length = model_config.max_sequence_length
    train_tokens = load_tokens(data_path / "train.npy")
    validation_tokens = load_tokens(data_path / "validation.npy")
    train_count = window_count(train_tokens, context_length)
    if not train_count:
        raise ValueError(
            f"Training data has fewer than {context_length + 1} tokens; cannot form a batch"
        )

    seed = int(training_config.get("seed", 42))
    mx.random.seed(seed)
    np.random.seed(seed)
    model = TransformerLM(model_config)
    optimizer = optim.AdamW(
        learning_rate=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.1)),
    )
    optimizer.init(model.trainable_parameters())
    mx.eval(model.parameters(), optimizer.state)

    state = {
        "epoch": 0,
        "batch_in_epoch": 0,
        "global_step": 0,
        "tokens_trained": 0,
        "best_validation_loss": math.inf,
        "evaluations_without_improvement": 0,
    }
    if resume_from is not None:
        state.update(restore_training_state(resume_from, model, optimizer))

    batch_size = int(training_config["batch_size"])
    epochs = int(training_config["epochs"])
    batches_per_epoch = math.ceil(train_count / batch_size)
    total_steps = max(1, epochs * batches_per_epoch)
    warmup_steps = min(int(training_config.get("warmup_steps", 100)), max(1, total_steps // 2))
    peak_lr = float(training_config["learning_rate"])
    minimum_lr = float(training_config.get("minimum_learning_rate", peak_lr * 0.1))
    gradient_clip = float(training_config.get("gradient_clip", 1.0))
    evaluation_interval = int(training_config.get("evaluation_interval", 250))
    checkpoint_interval = int(training_config.get("checkpoint_interval", 500))
    patience = int(training_config.get("early_stopping_patience", 5))

    value_and_grad = nn.value_and_grad(model, causal_lm_loss)
    compiled_state = [model.state, optimizer.state]

    def step(inputs: mx.array, targets: mx.array) -> tuple[mx.array, mx.array]:
        loss, gradients = value_and_grad(model, inputs, targets)
        gradients, gradient_norm = optim.clip_grad_norm(gradients, gradient_clip)
        optimizer.update(model, gradients)
        return loss, gradient_norm

    train_step = (
        partial(mx.compile, inputs=compiled_state, outputs=compiled_state)(step)
        if compile_step
        else step
    )
    metrics_path = output / "metrics.jsonl"
    started = time.perf_counter()
    stop_early = False

    def checkpoint(name: str) -> None:
        checkpoint_state = dict(state)
        checkpoint_state.update(
            {
                "total_steps": total_steps,
                "parameter_count": model.parameter_count,
                "mlx_version": mx.__version__,
                "project_version": __version__,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "compiled_step": compile_step,
                "initialized_from_checkpoint": resume_from is not None,
                "pretrained_weights_used": False,
            }
        )
        save_checkpoint(
            output / name,
            model,
            optimizer,
            checkpoint_state,
            training_config,
            tokenizer_dir,
        )

    for epoch in range(int(state["epoch"]), epochs):
        model.train()
        start_batch = int(state["batch_in_epoch"]) if epoch == int(state["epoch"]) else 0
        for batch_number, indices in enumerate(
            batch_indices(
                train_count,
                batch_size,
                seed=seed,
                epoch=epoch,
                start_batch=start_batch,
            ),
            start=start_batch,
        ):
            learning_rate = learning_rate_at_step(
                int(state["global_step"]), total_steps, warmup_steps, peak_lr, minimum_lr
            )
            optimizer.learning_rate = mx.array(learning_rate)
            inputs_np, targets_np = materialize_batch(train_tokens, indices, context_length)
            loss, gradient_norm = train_step(mx.array(inputs_np), mx.array(targets_np))
            mx.eval(loss, gradient_norm, model.parameters(), optimizer.state)
            state["global_step"] = int(state["global_step"]) + 1
            state["tokens_trained"] = int(state["tokens_trained"]) + int(targets_np.size)
            state["epoch"] = epoch
            state["batch_in_epoch"] = batch_number + 1
            elapsed = max(time.perf_counter() - started, 1e-9)
            metric = {
                "type": "train",
                "step": state["global_step"],
                "epoch": epoch,
                "tokens_trained": state["tokens_trained"],
                "learning_rate": learning_rate,
                "training_loss": float(loss.item()),
                "gradient_norm": float(gradient_norm.item()),
                "tokens_per_second": state["tokens_trained"] / elapsed,
                "peak_memory_bytes": int(mx.get_peak_memory()),
                "elapsed_seconds": elapsed,
            }
            _append_metric(metrics_path, metric)

            should_evaluate = state["global_step"] % evaluation_interval == 0
            is_last_batch = batch_number + 1 >= batches_per_epoch
            if should_evaluate or is_last_batch:
                validation_loss = evaluate_loss(
                    model,
                    validation_tokens,
                    context_length=context_length,
                    batch_size=batch_size,
                )
                _append_metric(
                    metrics_path,
                    {
                        "type": "validation",
                        "step": state["global_step"],
                        "validation_loss": validation_loss,
                        "validation_perplexity": perplexity(validation_loss),
                    },
                )
                if validation_loss < float(state["best_validation_loss"]):
                    state["best_validation_loss"] = validation_loss
                    state["evaluations_without_improvement"] = 0
                    checkpoint("best")
                else:
                    state["evaluations_without_improvement"] = (
                        int(state["evaluations_without_improvement"]) + 1
                    )
                model.train()
                if int(state["evaluations_without_improvement"]) >= patience:
                    stop_early = True
                    break
            if state["global_step"] % checkpoint_interval == 0:
                checkpoint("last")

        if stop_early:
            break
        state["epoch"] = epoch + 1
        state["batch_in_epoch"] = 0

    checkpoint("last")
    summary = {
        **state,
        "parameter_count": model.parameter_count,
        "train_windows": train_count,
        "context_length": context_length,
        "actual_vocab_size": actual_vocab_size,
        "early_stopped": stop_early,
        "initialized_from_checkpoint": resume_from is not None,
        "pretrained_weights_used": False,
        "output_dir": str(output),
    }
    write_json(output / "training-summary.json", summary)
    return summary
