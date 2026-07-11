from pathlib import Path

import numpy as np
import pytest

from imessage_mlx.data.rewrite import build_rewrite_pairs
from imessage_mlx.dataset import (
    encode_rewrite_split,
    estimate_parameter_count,
    materialize_masked_batch,
    select_model,
    select_rewrite_model,
)
from imessage_mlx.tokenizer.train import load_tokenizer, train_tokenizer


def _prepared_pairs(tmp_path: Path) -> tuple[Path, Path]:
    source = Path(__file__).parent / "fixtures/synthetic_rewrites.jsonl"
    processed = tmp_path / "processed.jsonl"
    build_rewrite_pairs(source, processed, tmp_path / "report.json")
    tokenizer_dir = tmp_path / "tokenizer"
    train_tokenizer(processed, tokenizer_dir, vocab_size=256, minimum_frequency=1)
    return processed, tokenizer_dir


def test_rewrite_encoding_packs_complete_pairs_and_aligns_masks(tmp_path: Path) -> None:
    processed, tokenizer_dir = _prepared_pairs(tmp_path)
    report = encode_rewrite_split(
        processed,
        tokenizer_dir,
        tmp_path / "tokens.npy",
        tmp_path / "loss-mask.npy",
        context_length=96,
    )
    tokens = np.load(tmp_path / "tokens.npy", allow_pickle=False)
    loss_mask = np.load(tmp_path / "loss-mask.npy", allow_pickle=False)
    tokenizer = load_tokenizer(tokenizer_dir)

    assert tokens.size == loss_mask.size
    assert tokens.size % 97 == 0
    assert int(loss_mask.sum()) == report["supervised_tokens"]
    assert report["pairs"] == 8
    rewrite_id = tokenizer.token_to_id("<|rewrite|>")
    pad_id = tokenizer.token_to_id("<|pad|>")
    assert np.all(loss_mask[tokens == rewrite_id] == 0)
    assert np.all(tokens[loss_mask == 1] != pad_id)

    indices = np.arange(report["windows"])
    _, _, shifted_mask = materialize_masked_batch(tokens, loss_mask, indices, 96)
    assert int(shifted_mask.sum()) == report["supervised_tokens"]


def test_rewrite_encoding_rejects_pair_larger_than_context(tmp_path: Path) -> None:
    processed, tokenizer_dir = _prepared_pairs(tmp_path)
    with pytest.raises(ValueError, match="exceeding block size"):
        encode_rewrite_split(
            processed,
            tokenizer_dir,
            tmp_path / "tokens.npy",
            tmp_path / "loss-mask.npy",
            context_length=8,
        )


def _candidate(name: str, hidden_size: int) -> dict[str, object]:
    return {
        "name": name,
        "hidden_size": hidden_size,
        "num_layers": 1,
        "intermediate_size": hidden_size * 3,
        "tie_embeddings": True,
    }


def test_rewrite_selection_enforces_training_pair_and_token_minimums() -> None:
    candidate = _candidate("small", 16)
    too_few_tokens = select_rewrite_model(99_999, 10_000, 256, [candidate])
    too_few_pairs = select_rewrite_model(100_000, 9_999, 256, [candidate])

    assert too_few_tokens["selected"] is None
    assert (
        "insufficient_supervised_target_tokens"
        in too_few_tokens["candidates"][0]["rejection_reasons"]
    )
    assert too_few_pairs["selected"] is None
    assert "insufficient_training_pairs" in too_few_pairs["candidates"][0]["rejection_reasons"]


def test_rewrite_selection_uses_exact_ratio_and_largest_eligible_candidate() -> None:
    small = _candidate("small", 16)
    large = _candidate("large", 24)
    large_parameters = estimate_parameter_count(large, 256)
    exact_tokens = large_parameters * 2

    report = select_rewrite_model(
        exact_tokens,
        10_000,
        256,
        [small, large],
        minimum_train_tokens=0,
    )

    assert report["selected"] == "large"
    assert report["eligible_to_train"] is True
    assert report["candidates"][1]["supervised_target_tokens_per_parameter"] == 2.0
    below_ratio = select_rewrite_model(
        exact_tokens - 1,
        10_000,
        256,
        [large],
        minimum_train_tokens=0,
    )
    assert below_ratio["selected"] is None


def test_reply_model_selection_policy_is_unchanged() -> None:
    small = _candidate("model-1m", 16)
    large = _candidate("model-7m", 24)
    report = select_model(999_999, 256, small, large)

    assert report["minimum_tokens_per_parameter"] == 10
    assert report["enough_tokens_to_train"] is False
    assert report["selected"] == "model-7m"
    fallback = select_model(0, 256, small, large)
    assert fallback["selected"] == "model-1m"
