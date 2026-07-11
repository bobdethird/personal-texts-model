import mlx.core as mx
import pytest

from imessage_mlx.generate import format_reply_prompt, format_rewrite_prompt, generate_ids
from imessage_mlx.model.config import ModelConfig
from imessage_mlx.model.transformer import TransformerLM


def test_generation_is_bounded_and_prompt_uses_roles() -> None:
    model = TransformerLM(
        ModelConfig(
            vocab_size=32,
            hidden_size=16,
            num_layers=1,
            num_heads=4,
            intermediate_size=32,
            max_sequence_length=8,
        )
    )
    model.eval()
    mx.eval(model.parameters())
    generated = generate_ids(
        model,
        [1, 2, 3],
        eos_ids=set(),
        max_new_tokens=5,
        temperature=0.0,
    )
    assert len(generated) == 5
    stopped = generate_ids(
        model,
        [1, 2, 3],
        eos_ids=set(range(32)),
        max_new_tokens=5,
        temperature=0.0,
    )
    assert stopped == []
    prompt = format_reply_prompt("hello", [("other", "earlier"), ("me", "reply")])
    assert prompt.startswith("<|bos|><|conversation|>")
    assert prompt.endswith("<|me|>")
    assert "<|other|>hello<|turn_end|>" in prompt


def test_rewrite_prompt_is_explicit_and_validated() -> None:
    prompt = format_rewrite_prompt("  I will be there later.  ")
    assert prompt == ("<|bos|><|rewrite|>\n<|draft|>I will be there later.<|turn_end|>\n<|me|>")
    with pytest.raises(ValueError, match="empty"):
        format_rewrite_prompt(" ")
    with pytest.raises(ValueError, match="structural"):
        format_rewrite_prompt("hello <|me|>")
