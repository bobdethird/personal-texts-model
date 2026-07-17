import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from imessage_mlx.model.config import ModelConfig
from imessage_mlx.model.transformer import TransformerLM, causal_lm_loss


def tiny_model() -> TransformerLM:
    mx.random.seed(42)
    return TransformerLM(
        ModelConfig(
            vocab_size=64,
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            intermediate_size=64,
            max_sequence_length=16,
        )
    )


def test_causal_mask_prevents_future_token_leakage() -> None:
    model = tiny_model()
    model.eval()
    first = model(mx.array([[1, 2, 3, 4]], dtype=mx.int32))
    second = model(mx.array([[1, 2, 3, 5]], dtype=mx.int32))
    mx.eval(first, second)
    assert bool(mx.allclose(first[:, :3], second[:, :3], atol=1e-5).item())


def test_overfits_one_tiny_batch() -> None:
    model = tiny_model()
    model.train()
    inputs = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=mx.int32)
    targets = mx.array([[2, 3, 4, 5, 6, 7, 8, 9]], dtype=mx.int32)
    optimizer = optim.AdamW(learning_rate=0.03, weight_decay=0.0)
    loss_and_grad = nn.value_and_grad(model, causal_lm_loss)
    initial = causal_lm_loss(model, inputs, targets)
    mx.eval(initial)
    for _ in range(80):
        loss, gradients = loss_and_grad(model, inputs, targets)
        optimizer.update(model, gradients)
        mx.eval(loss, model.parameters(), optimizer.state)
    final = causal_lm_loss(model, inputs, targets)
    mx.eval(final)
    assert float(final.item()) < float(initial.item()) * 0.25
