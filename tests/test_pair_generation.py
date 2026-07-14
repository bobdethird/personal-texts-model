from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from imessage_mlx.data import pair_generation
from imessage_mlx.data.pair_generation import (
    NEUTRALIZATION_INSTRUCTIONS,
    NeutralizedBatch,
    NeutralizedItem,
    create_rewrite_review,
    generate_openai_rewrite_pairs,
    select_outgoing_messages,
)
from imessage_mlx.utils import read_jsonl, write_jsonl


def test_neutralization_prompt_has_corpus_specific_abbreviations() -> None:
    assert '"sm" means "something"' in NEUTRALIZATION_INSTRUCTIONS
    assert '"ts" means either' in NEUTRALIZATION_INSTRUCTIONS
    assert 'preserve "ts" rather than guessing' in NEUTRALIZATION_INSTRUCTIONS


def _messages(path: Path) -> None:
    write_jsonl(
        path,
        [
            {
                "message_id": "incoming",
                "timestamp_ns": 1,
                "sender_role": "other",
                "text": "incoming",
            },
            {
                "message_id": "old",
                "timestamp_ns": 2,
                "sender_role": "me",
                "text": "same text",
            },
            {
                "message_id": "structural",
                "timestamp_ns": 3,
                "sender_role": "me",
                "text": "look <|url|>",
            },
            {
                "message_id": "new-duplicate",
                "timestamp_ns": 4,
                "sender_role": "me",
                "text": "same text",
            },
            {
                "message_id": "latest",
                "timestamp_ns": 5,
                "sender_role": "me",
                "text": "omw rn",
            },
        ],
    )


def test_selects_recent_unique_outgoing_messages_without_structural_tokens(
    tmp_path: Path,
) -> None:
    messages = tmp_path / "messages.jsonl"
    _messages(messages)

    selected, counts = select_outgoing_messages(messages, limit=2)

    assert [value["pair_id"] for value in selected] == ["new-duplicate", "latest"]
    assert counts["outgoing_records"] == 4
    assert counts["skipped_structural_tokens"] == 1
    assert counts["selected_records"] == 2


def test_openai_pair_generation_is_structured_private_and_resumable(
    tmp_path: Path, monkeypatch
) -> None:
    messages = tmp_path / "messages.jsonl"
    output = tmp_path / "pairs.jsonl"
    _messages(messages)

    class FakeResponses:
        def __init__(self):
            self.calls = 0

        async def parse(self, *, input, **_options):
            self.calls += 1
            payload = json.loads(input)
            parsed = NeutralizedBatch(
                items=[
                    NeutralizedItem(
                        pair_id=value["pair_id"],
                        neutral_text=f"Neutral version {index}",
                    )
                    for index, value in enumerate(payload)
                ]
            )
            return SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type="message",
                        content=[SimpleNamespace(parsed=parsed)],
                    )
                ],
                usage=SimpleNamespace(input_tokens=20, output_tokens=10, total_tokens=30),
            )

    responses = FakeResponses()

    class FakeClient:
        def __init__(self, **_options):
            self.responses = responses

        async def close(self):
            return None

    monkeypatch.setattr(pair_generation, "AsyncOpenAI", FakeClient)
    report = asyncio.run(
        generate_openai_rewrite_pairs(
            messages,
            output,
            tmp_path / "report.json",
            api_key="test-key",
            model="test-model",
            limit=2,
            batch_size=2,
        )
    )

    pairs = list(read_jsonl(output))
    assert report["generated_pairs"] == 2
    assert report["failed_batches"] == 0
    assert report["usage"]["total_tokens"] == 30
    assert report["message_text_persisted_in_report"] is False
    assert [pair["styled_text"] for pair in pairs] == ["same text", "omw rn"]
    assert all(pair["neutral_text"].startswith("Neutral version") for pair in pairs)

    resumed = asyncio.run(
        generate_openai_rewrite_pairs(
            messages,
            output,
            tmp_path / "resumed-report.json",
            api_key="test-key",
            model="test-model",
            limit=2,
            batch_size=2,
        )
    )
    assert resumed["already_present"] == 2
    assert resumed["generated_pairs"] == 0
    assert responses.calls == 1

    review = create_rewrite_review(output, tmp_path / "review.md", sample_size=2)
    review_text = (tmp_path / "review.md").read_text(encoding="utf-8")
    assert review["sampled_pairs"] == 2
    assert "Neutral version" in review_text
    assert "Original styled target" in review_text
