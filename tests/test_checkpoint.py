from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from imessage_mlx.checkpoint import load_model, restore_training_state, save_checkpoint
from imessage_mlx.model.config import ModelConfig
from imessage_mlx.model.transformer import TransformerLM, causal_lm_loss


def _model() -> TransformerLM:
    return TransformerLM(
        ModelConfig(
            vocab_size=32,
            hidden_size=16,
            num_layers=1,
            num_heads=4,
            intermediate_size=32,
            max_sequence_length=8,
        )
    )


def _update(model, optimizer, inputs, targets) -> None:
    loss_and_grad = nn.value_and_grad(model, causal_lm_loss)
    loss, gradients = loss_and_grad(model, inputs, targets)
    optimizer.update(model, gradients)
    mx.eval(loss, model.parameters(), optimizer.state)


def test_checkpoint_round_trip_and_resume_match(tmp_path: Path) -> None:
    mx.random.seed(7)
    model = _model()
    optimizer = optim.AdamW(learning_rate=0.01)
    inputs = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    targets = mx.array([[2, 3, 4, 5]], dtype=mx.int32)
    _update(model, optimizer, inputs, targets)
    before = model(inputs)
    mx.eval(before)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model, optimizer, {"global_step": 1}, {"seed": 7})
    assert (checkpoint / "training-config.yaml").exists()

    reloaded = load_model(checkpoint)
    after = reloaded(inputs)
    mx.eval(after)
    assert bool(mx.allclose(before, after, atol=1e-6).item())

    _update(model, optimizer, inputs, targets)
    resumed = _model()
    resumed_optimizer = optim.AdamW(learning_rate=0.01)
    state = restore_training_state(checkpoint, resumed, resumed_optimizer)
    _update(resumed, resumed_optimizer, inputs, targets)
    assert state["global_step"] == 1
    for (_, expected), (_, actual) in zip(
        tree_flatten(model.parameters()), tree_flatten(resumed.parameters()), strict=True
    ):
        assert bool(mx.allclose(expected, actual, atol=1e-6).item())
