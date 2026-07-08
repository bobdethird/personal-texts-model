from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from imessage_mlx.model.config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim**-0.5
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_base)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, hidden: mx.array, mask: mx.array) -> mx.array:
        batch, length, width = hidden.shape

        def heads(value: mx.array) -> mx.array:
            return value.reshape(batch, length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        queries = self.rope(heads(self.query(hidden)))
        keys = self.rope(heads(self.key(hidden)))
        values = heads(self.value(hidden))
        attended = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=self.scale,
            mask=mask,
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, length, width)
        return self.dropout(self.output(attended))


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, hidden: mx.array) -> mx.array:
        return self.dropout(self.down(nn.silu(self.gate(hidden)) * self.up(hidden)))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_epsilon)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_epsilon)
        self.mlp = SwiGLU(config)

    def __call__(self, hidden: mx.array, mask: mx.array) -> mx.array:
        hidden = hidden + self.attention(self.attention_norm(hidden), mask)
        return hidden + self.mlp(self.mlp_norm(hidden))


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.layers = [TransformerBlock(config) for _ in range(config.num_layers)]
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_epsilon)
        if not config.tie_embeddings:
            self.output_projection = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, input_ids: mx.array) -> mx.array:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        length = input_ids.shape[1]
        if length > self.config.max_sequence_length:
            raise ValueError(
                f"Sequence length {length} exceeds configured maximum "
                f"{self.config.max_sequence_length}"
            )
        hidden = self.embedding_dropout(self.token_embedding(input_ids))
        mask = nn.MultiHeadAttention.create_additive_causal_mask(length, dtype=hidden.dtype)
        for layer in self.layers:
            hidden = layer(hidden, mask)
        hidden = self.final_norm(hidden)
        if self.config.tie_embeddings:
            return self.token_embedding.as_linear(hidden)
        return self.output_projection(hidden)

    @property
    def parameter_count(self) -> int:
        from mlx.utils import tree_flatten

        return sum(value.size for _, value in tree_flatten(self.parameters()))


def causal_lm_loss(model: TransformerLM, inputs: mx.array, targets: mx.array) -> mx.array:
    logits = model(inputs)
    return nn.losses.cross_entropy(logits, targets, reduction="mean")


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 50.0))
