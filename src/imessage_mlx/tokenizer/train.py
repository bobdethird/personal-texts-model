from __future__ import annotations

from pathlib import Path
from typing import Any

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from imessage_mlx.utils import ensure_private_dir, read_jsonl, write_json

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    "<|conversation|>",
    "<|me|>",
    "<|other|>",
    "<|turn_end|>",
    "<|attachment|>",
    "<|url|>",
    "<|email|>",
    "<|phone|>",
    "<|rewrite|>",
    "<|draft|>",
]


def load_tokenizer(path: str | Path) -> Tokenizer:
    directory = Path(path)
    tokenizer_file = directory / "tokenizer.json" if directory.is_dir() else directory
    return Tokenizer.from_file(str(tokenizer_file))


def train_tokenizer(
    train_path: str | Path,
    output_dir: str | Path,
    *,
    vocab_size: int = 4096,
    minimum_frequency: int = 2,
) -> dict[str, Any]:
    output = ensure_private_dir(output_dir)
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=minimum_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )

    def text_iterator():
        for record in read_jsonl(train_path):
            yield str(record["text"])

    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
    tokenizer.save(str(output / "tokenizer.json"))
    (output / "tokenizer.json").chmod(0o600)
    special_ids = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    if any(value is None for value in special_ids.values()):
        raise RuntimeError("Tokenizer did not reserve every required special token")
    if len(set(special_ids.values())) != len(special_ids):
        raise RuntimeError("Tokenizer assigned duplicate special-token IDs")
    config = {
        "tokenizer_class": "TokenizersBackend",
        "model_max_length": 512,
        "unk_token": "<|unk|>",
        "pad_token": "<|pad|>",
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "additional_special_tokens": SPECIAL_TOKENS[4:],
        "vocab_size": tokenizer.get_vocab_size(),
    }
    write_json(output / "tokenizer_config.json", config)
    write_json(
        output / "special_tokens_map.json",
        {
            "unk_token": "<|unk|>",
            "pad_token": "<|pad|>",
            "bos_token": "<|bos|>",
            "eos_token": "<|eos|>",
            "additional_special_tokens": SPECIAL_TOKENS[4:],
            "ids": special_ids,
        },
    )
    report = {
        "training_source": "train split only",
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": tokenizer.get_vocab_size(),
        "minimum_frequency": minimum_frequency,
        "special_token_ids": special_ids,
    }
    write_json(output / "training-report.json", report)
    return report
