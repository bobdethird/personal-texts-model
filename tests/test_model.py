from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from imessage_mlx.config import load_yaml
from imessage_mlx.dataset import estimate_parameter_count
from imessage_mlx.model.config import ModelConfig
from imessage_mlx.model.transformer import TransformerLM, causal_lm_loss, masked_causal_lm_loss


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


def test_masked_loss_ignores_prompt_targets() -> None:
    model = tiny_model()
    model.eval()
    inputs = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    baseline_targets = mx.array([[5, 6, 7, 8]], dtype=mx.int32)
    prompt_changed = mx.array([[9, 6, 7, 8]], dtype=mx.int32)
    styled_changed = mx.array([[5, 10, 7, 8]], dtype=mx.int32)
    mask = mx.array([[0, 1, 1, 1]], dtype=mx.float32)

    baseline = masked_causal_lm_loss(model, inputs, baseline_targets, mask)
    prompt_loss = masked_causal_lm_loss(model, inputs, prompt_changed, mask)
    styled_loss = masked_causal_lm_loss(model, inputs, styled_changed, mask)
    mx.eval(baseline, prompt_loss, styled_loss)

    assert bool(mx.allclose(baseline, prompt_loss, atol=1e-7).item())
    assert not bool(mx.allclose(baseline, styled_loss, atol=1e-7).item())


def test_rewrite_preset_parameter_counts_match_live_models() -> None:
    root = Path(__file__).parents[1]
    for name, expected in (
        ("model-rewrite-190k.yaml", 188_496),
        ("model-rewrite-290k.yaml", 291_264),
    ):
        config = load_yaml(root / f"configs/{name}")
        assert estimate_parameter_count(config, 2048) == expected
        model = TransformerLM(ModelConfig.from_dict(config))
        mx.eval(model.parameters())
        assert model.parameter_count == expected
