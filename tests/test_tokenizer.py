from pathlib import Path

from imessage_mlx.tokenizer.train import SPECIAL_TOKENS, load_tokenizer, train_tokenizer
from imessage_mlx.utils import write_jsonl


def test_trains_local_tokenizer_with_unique_special_tokens(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    write_jsonl(
        train,
        [
            {
                "text": "<|bos|><|conversation|>\n<|other|>héllo 😊<|turn_end|>\n"
                "<|me|>hi there<|turn_end|>\n<|eos|>"
            }
            for _ in range(10)
        ],
    )
    report = train_tokenizer(train, tmp_path / "tokenizer", vocab_size=256, minimum_frequency=1)
    tokenizer = load_tokenizer(tmp_path / "tokenizer")

    ids = [tokenizer.token_to_id(token) for token in SPECIAL_TOKENS]
    assert all(token_id is not None for token_id in ids)
    assert len(ids) == len(set(ids))
    assert report["training_source"] == "train split only"
    assert tokenizer.encode("<|turn_end|>", add_special_tokens=False).ids == [
        tokenizer.token_to_id("<|turn_end|>")
    ]
    text = "héllo 😊\nsecond line"
    assert tokenizer.decode(tokenizer.encode(text).ids) == text
    unseen = "unseen bytes: 🧪漢字"
    assert tokenizer.decode(tokenizer.encode(unseen).ids) == unseen
