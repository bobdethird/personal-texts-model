from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int
    max_sequence_length: int
    dropout: float = 0.0
    tie_embeddings: bool = True
    rope_base: float = 10_000.0
    rms_norm_epsilon: float = 1e-5

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if (self.hidden_size // self.num_heads) % 2:
            raise ValueError("attention head dimension must be even for rotary embeddings")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.vocab_size <= 0 or self.num_layers <= 0 or self.max_sequence_length <= 0:
            raise ValueError("model dimensions must be positive")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ModelConfig:
        fields = cls.__dataclass_fields__
        return cls(**{key: value[key] for key in fields if key in value})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
