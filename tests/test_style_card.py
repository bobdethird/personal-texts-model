from pathlib import Path

from imessage_mlx.data.pairs import PAIR_FORMAT
from imessage_mlx.style_card import (
    build_style_card_prompt,
    run_style_card,
    select_style_pairs,
)
from imessage_mlx.utils import (
    atomic_write_text,
    sha256_file,
    write_json,
    write_jsonl,
)


def _pair(
    pair_id: str,
    query: str,
    reply: str,
    *,
    split: str = "train",
) -> dict[str, str]:
    return {
        "format": PAIR_FORMAT,
        "pair_id": pair_id,
        "split": split,
        "query": query,
        "reply": reply,
    }


def test_style_pair_selection_is_seeded_and_budgeted() -> None:
    pairs = [
        _pair("a", "a" * 10, "A" * 10),
        _pair("b", "b" * 10, "B" * 10),
        _pair("c", "c" * 10, "C" * 10),
    ]

    first = select_style_pairs(pairs, samples=3, seed=7, max_source_chars=40)
    second = select_style_pairs(pairs, samples=3, seed=7, max_source_chars=40)

    assert [pair["pair_id"] for pair in first] == [pair["pair_id"] for pair in second]
    assert len(first) == 2


def test_style_pair_selection_excludes_validation_pairs() -> None:
    pairs = [
        _pair("train", "free?", "after 7"),
        _pair("validation", "dinner?", "sure", split="validation"),
    ]

    selected = select_style_pairs(pairs, samples=2, seed=7, max_source_chars=100)

    assert [pair["pair_id"] for pair in selected] == ["train"]


def test_style_card_prompt_requests_behavior_not_private_facts() -> None:
    prompt = build_style_card_prompt([_pair("a", "Are you free?", "yeah after 7")])

    assert prompt.startswith("<|im_start|>system\n")
    assert "Do not quote examples, repeat private facts, mention names" in prompt
    assert "Incoming: Are you free?" in prompt
    assert "Reply: yeah after 7" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_style_card_reuses_cache_only_with_matching_inputs(tmp_path: Path) -> None:
    pairs_path = tmp_path / "pairs.jsonl"
    output_path = tmp_path / "style-card.md"
    report_path = tmp_path / "style-card.json"
    write_jsonl(pairs_path, [_pair("a", "Are you free?", "yeah after 7")])
    atomic_write_text(output_path, "- concise\n")
    write_json(
        report_path,
        {
            "model_name": "fake-model",
            "pairs_sha256": sha256_file(pairs_path),
            "samples": 128,
            "seed": 42,
            "max_source_chars": 24_000,
            "max_new_tokens": 768,
            "temperature": 0.2,
            "output_path": str(output_path),
        },
    )

    result = run_style_card(
        {
            "pairs_path": str(pairs_path),
            "output_path": str(output_path),
            "report_path": str(report_path),
            "model_name": "fake-model",
        }
    )

    assert result["cached"] is True
    assert result["style_card"] == "- concise"
