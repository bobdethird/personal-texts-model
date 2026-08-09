from typing import Any

from imessage_mlx.data.sft import SFT_FORMAT
from imessage_mlx.training import encode_sft_record, render_session_turns


class CharacterTokenizer:
    bos_token_id = 1

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) + 10 for character in text]


def _record() -> dict[str, Any]:
    return {
        "format": SFT_FORMAT,
        "example_id": "example",
        "session_id": "session",
        "supervised_indexes": [1, 2],
        "messages": [
            {
                "role": "user",
                "content": "Incoming",
                "participant": "person-a",
            },
            {"role": "assistant", "content": "First reply"},
            {"role": "assistant", "content": "Second reply"},
        ],
    }


def test_renderer_marks_every_assistant_turn_as_supervised() -> None:
    turns = render_session_turns(_record())

    assert [turn["supervised"] for turn in turns] == [False, True, True]
    assert "[participant:person-a]" in turns[0]["text"]
    assert turns[0]["text"].startswith("<|im_start|>user\n")
    assert turns[1]["text"].startswith("<|im_start|>assistant\n")
    assert turns[1]["text"].endswith("<|im_end|>\n")


def test_encoder_masks_user_turns_and_supervises_every_assistant_turn() -> None:
    tokenizer = CharacterTokenizer()
    chunks = encode_sft_record(_record(), tokenizer, max_length=1024)

    assert len(chunks) == 1
    chunk = chunks[0]
    expected_first = tokenizer.encode(
        "<|im_start|>assistant\nFirst reply<|im_end|>\n", add_special_tokens=False
    )
    expected_second = tokenizer.encode(
        "<|im_start|>assistant\nSecond reply<|im_end|>\n", add_special_tokens=False
    )
    supervised = [label for label in chunk["labels"] if label != -100]
    assert supervised == expected_first + expected_second
    assert len(chunk["input_ids"]) == len(chunk["labels"])
    assert chunk["labels"][0] == -100


def test_long_sessions_are_windowed_without_dropping_supervised_tokens() -> None:
    tokenizer = CharacterTokenizer()
    long_reply = "abcdefghijklmnopqrstuvwxyz" * 4
    record = {
        "format": SFT_FORMAT,
        "example_id": "long",
        "session_id": "session",
        "supervised_indexes": [1, 3],
        "messages": [
            {"role": "user", "content": "Start"},
            {"role": "assistant", "content": long_reply},
            {"role": "user", "content": "Continue"},
            {"role": "assistant", "content": "Short"},
        ],
    }
    chunks = encode_sft_record(record, tokenizer, max_length=256)
    expected = []
    for turn in render_session_turns(record):
        if turn["supervised"]:
            expected.extend(tokenizer.encode(turn["text"], add_special_tokens=False))

    supervised = [
        label for chunk in chunks for label in chunk["labels"] if label != -100
    ]
    assert len(chunks) > 1
    assert supervised == expected
    assert all(len(chunk["input_ids"]) <= 256 for chunk in chunks)
    assert all(chunk["labels"][0] == -100 for chunk in chunks)
