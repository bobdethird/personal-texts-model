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
from imessage_mlx.dataset import (
    batch_indices,
    load_tokens,
    materialize_batch,
    materialize_masked_batch,
    window_count,
)
from imessage_mlx.model.config import ModelConfig
from imessage_mlx.model.transformer import (
    TransformerLM,
    causal_lm_loss,
    masked_causal_lm_loss,
    perplexity,
)
from imessage_mlx.tokenizer.train import load_tokenizer
from imessage_mlx.utils import ensure_private_dir, sha256_file, write_json


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
    loss_mask: np.ndarray | None = None,
) -> float:
    model.eval()
    count = window_count(tokens, context_length)
    if not count:
        raise ValueError("Evaluation split has no complete context window")
    total_loss = 0.0
    total_weight = 0.0
    for start in range(0, count, batch_size):
        indices = np.arange(start, min(count, start + batch_size))
        if loss_mask is None:
            inputs_np, targets_np = materialize_batch(tokens, indices, context_length)
            loss = causal_lm_loss(model, mx.array(inputs_np), mx.array(targets_np))
            batch_weight = float(len(indices))
        else:
            inputs_np, targets_np, mask_np = materialize_masked_batch(
                tokens, loss_mask, indices, context_length
            )
            loss = masked_causal_lm_loss(
                model,
                mx.array(inputs_np),
                mx.array(targets_np),
                mx.array(mask_np),
            )
            batch_weight = float(mask_np.sum())
        mx.eval(loss)
        total_loss += float(loss.item()) * batch_weight
        total_weight += batch_weight
    if not total_weight:
        raise ValueError("Evaluation split contains no supervised target tokens")
    return total_loss / total_weight


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
    tokenizer_path = Path(tokenizer_dir)
    task = str(training_config.get("task", "reply"))
    objective = str(training_config.get("objective", "causal"))
    expected_objectives = {"reply": "causal", "rewrite": "target_only"}
    if task not in expected_objectives:
        raise ValueError(f"Unsupported training task {task!r}")
    if objective != expected_objectives[task]:
        raise ValueError(
            f"Training task {task!r} requires objective {expected_objectives[task]!r}, "
            f"not {objective!r}"
        )
    if resume_from is not None:
        checkpoint_tokenizer = Path(resume_from) / "tokenizer/tokenizer.json"
        supplied_tokenizer = tokenizer_path / "tokenizer.json"
        if not checkpoint_tokenizer.exists() or not supplied_tokenizer.exists():
            raise ValueError("Cannot verify tokenizer compatibility for checkpoint resume")
        if sha256_file(checkpoint_tokenizer) != sha256_file(supplied_tokenizer):
            raise ValueError("Cannot resume with a tokenizer that differs from the checkpoint")
        checkpoint_training_path = Path(resume_from) / "training-config.json"
        if not checkpoint_training_path.exists():
            raise ValueError("Cannot verify training configuration for checkpoint resume")
        checkpoint_training = json.loads(checkpoint_training_path.read_text(encoding="utf-8"))
        checkpoint_task = str(checkpoint_training.get("task", "reply"))
        checkpoint_objective = str(checkpoint_training.get("objective", "causal"))
        if (checkpoint_task, checkpoint_objective) != (task, objective):
            raise ValueError("Cannot change task or objective when resuming a checkpoint")
    tokenizer = load_tokenizer(tokenizer_dir)
    actual_vocab_size = tokenizer.get_vocab_size()
    config_value = dict(training_config)
    config_value["vocab_size"] = actual_vocab_size
    model_config = ModelConfig.from_dict(config_value)
    if resume_from is not None:
        checkpoint_model_path = Path(resume_from) / "model-config.json"
        if not checkpoint_model_path.exists():
            raise ValueError("Cannot verify model configuration for checkpoint resume")
        checkpoint_model = json.loads(checkpoint_model_path.read_text(encoding="utf-8"))
        if checkpoint_model != model_config.to_dict():
            raise ValueError("Cannot change model configuration when resuming a checkpoint")
    context_length = model_config.max_sequence_length
    train_tokens = load_tokens(data_path / "train.npy")
    validation_tokens = load_tokens(data_path / "validation.npy")
    train_loss_mask = (
        load_tokens(data_path / "train-loss-mask.npy") if objective == "target_only" else None
    )
    validation_loss_mask = (
        load_tokens(data_path / "validation-loss-mask.npy") if objective == "target_only" else None
    )
    if train_loss_mask is not None and train_loss_mask.size != train_tokens.size:
        raise ValueError("Training token and loss-mask arrays must have identical lengths")
    if validation_loss_mask is not None and validation_loss_mask.size != validation_tokens.size:
        raise ValueError("Validation token and loss-mask arrays must have identical lengths")
    if objective == "target_only":
        token_report_path = data_path / "token-counts.json"
        if not token_report_path.exists():
            raise ValueError("Target-only training requires rewrite token-count metadata")
        token_report = json.loads(token_report_path.read_text(encoding="utf-8"))
        for split_name, tokens in (
            ("train", train_tokens),
            ("validation", validation_tokens),
        ):
            encoded_context = int(token_report.get(split_name, {}).get("context_length", -1))
            if encoded_context != context_length:
                raise ValueError(
                    f"Rewrite {split_name} data was encoded for context {encoded_context}, "
                    f"not model context {context_length}"
                )
            if tokens.size % (context_length + 1):
                raise ValueError(f"Rewrite {split_name} data does not contain complete blocks")
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
        "supervised_tokens_trained": 0,
        "best_validation_loss": math.inf,
        "evaluations_without_improvement": 0,
    }
    if resume_from is not None:
        state.update(restore_training_state(resume_from, model, optimizer))
        state.setdefault("supervised_tokens_trained", 0)

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

    compiled_state = [model.state, optimizer.state]
    if objective == "target_only":
        value_and_grad = nn.value_and_grad(model, masked_causal_lm_loss)

        def masked_step(
            inputs: mx.array, targets: mx.array, loss_mask: mx.array
        ) -> tuple[mx.array, mx.array]:
            loss, gradients = value_and_grad(model, inputs, targets, loss_mask)
            gradients, gradient_norm = optim.clip_grad_norm(gradients, gradient_clip)
            optimizer.update(model, gradients)
            return loss, gradient_norm

        train_step = (
            partial(mx.compile, inputs=compiled_state, outputs=compiled_state)(masked_step)
            if compile_step
            else masked_step
        )
    else:
        value_and_grad = nn.value_and_grad(model, causal_lm_loss)

        def causal_step(inputs: mx.array, targets: mx.array) -> tuple[mx.array, mx.array]:
            loss, gradients = value_and_grad(model, inputs, targets)
            gradients, gradient_norm = optim.clip_grad_norm(gradients, gradient_clip)
            optimizer.update(model, gradients)
            return loss, gradient_norm

        train_step = (
            partial(mx.compile, inputs=compiled_state, outputs=compiled_state)(causal_step)
            if compile_step
            else causal_step
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
            if train_loss_mask is None:
                inputs_np, targets_np = materialize_batch(train_tokens, indices, context_length)
                loss, gradient_norm = train_step(mx.array(inputs_np), mx.array(targets_np))
                supervised_tokens = int(targets_np.size)
            else:
                inputs_np, targets_np, mask_np = materialize_masked_batch(
                    train_tokens, train_loss_mask, indices, context_length
                )
                loss, gradient_norm = train_step(
                    mx.array(inputs_np), mx.array(targets_np), mx.array(mask_np)
                )
                supervised_tokens = int(mask_np.sum())
            mx.eval(loss, gradient_norm, model.parameters(), optimizer.state)
            state["global_step"] = int(state["global_step"]) + 1
            state["tokens_trained"] = int(state["tokens_trained"]) + int(targets_np.size)
            state["supervised_tokens_trained"] = (
                int(state["supervised_tokens_trained"]) + supervised_tokens
            )
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
                    loss_mask=validation_loss_mask,
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
        "objective": objective,
        "early_stopped": stop_early,
        "initialized_from_checkpoint": resume_from is not None,
        "pretrained_weights_used": False,
        "output_dir": str(output),
    }
    write_json(output / "training-summary.json", summary)
    return summary
