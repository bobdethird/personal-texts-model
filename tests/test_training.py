from typing import Any

from imessage_mlx.data.sft import SFT_FORMAT
from imessage_mlx.training import encode_sft_record, render_example_parts


class CharacterTokenizer:
    bos_token_id = 1

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) + 10 for character in text]


def _record(target: str = "My reply") -> dict[str, Any]:
    return {
        "format": SFT_FORMAT,
        "example_id": "example",
        "target_index": 2,
        "messages": [
            {
                "role": "user",
                "content": "Incoming",
                "participant": "person-a",
            },
            {"role": "assistant", "content": "Earlier reply"},
            {"role": "assistant", "content": target},
        ],
    }


def test_renderer_keeps_the_target_out_of_the_prompt() -> None:
    prompt, target = render_example_parts(_record())

    assert "Incoming" in prompt
    assert "Earlier reply" in prompt
    assert "[participant:person-a]" in prompt
    assert "My reply" not in prompt
    assert prompt.endswith("<|assistant|>")
    assert target == "My reply<|turn_end|>"


def test_encoder_masks_history_and_supervises_only_target() -> None:
    tokenizer = CharacterTokenizer()
    chunks = encode_sft_record(_record(), tokenizer, max_length=512)
    expected_target = tokenizer.encode("My reply<|turn_end|>", add_special_tokens=False)

    assert len(chunks) == 1
    chunk = chunks[0]
    supervised = [label for label in chunk["labels"] if label != -100]
    assert supervised == expected_target
    assert chunk["labels"][:-len(expected_target)] == [-100] * (
        len(chunk["labels"]) - len(expected_target)
    )
    assert len(chunk["input_ids"]) == len(chunk["labels"])


def test_long_targets_are_chunked_without_dropping_or_repeating_loss_tokens() -> None:
    tokenizer = CharacterTokenizer()
    target_text = "abcdefghijklmnopqrstuvwxyz" * 3
    chunks = encode_sft_record(_record(target_text), tokenizer, max_length=24)
    expected_target = tokenizer.encode(
        f"{target_text}<|turn_end|>", add_special_tokens=False
    )

    supervised = [
        label
        for chunk in chunks
        for label in chunk["labels"]
        if label != -100
    ]
    assert len(chunks) > 1
    assert supervised == expected_target
    assert all(len(chunk["input_ids"]) <= 24 for chunk in chunks)
    assert all(chunk["labels"][0] == -100 for chunk in chunks)
