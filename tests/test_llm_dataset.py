import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from imessage_mlx.data import llm_dataset
from imessage_mlx.data.llm_dataset import (
    ExampleVerdict,
    LlmDatasetResumeError,
    ProposedExample,
    WindowJudgement,
    WindowProposal,
    WindowResponseError,
    build_llm_dataset,
    build_windows,
    create_llm_dataset_review,
    example_id,
    load_accepted_examples,
    validate_window_judgement,
    validate_window_proposal,
)
from imessage_mlx.utils import read_jsonl, write_jsonl

SECOND_NS = 1_000_000_000


def _message(
    number: int,
    *,
    chat: str = "chat-1",
    role: str = "other",
    participant: str = "p-1",
    name: str | None = "Colin Hu",
    at_seconds: int,
    text: str = "hello",
    group: bool = False,
) -> dict:
    return {
        "message_id": f"m-{number}",
        "chat_id": chat,
        "timestamp_ns": at_seconds * SECOND_NS,
        "sender_role": role,
        "sender_name": "Me" if role == "me" else name,
        "participant_id": "p-me" if role == "me" else participant,
        "text": text,
        "is_group": group,
        "service": "iMessage",
    }


def _conversation() -> list[dict]:
    return [
        _message(1, at_seconds=0, text="are you coming tonight?"),
        _message(2, role="me", at_seconds=10, text="ya omw"),
        _message(3, role="me", at_seconds=14, text="be there in 10"),
        _message(4, at_seconds=30, text="ok cool"),
    ]


class FakeResponses:
    """Proposes one turn per run of 'Me' messages, then accepts or rejects them all."""

    def __init__(self, *, accept: bool = True, broken_proposals: bool = False) -> None:
        self.accept = accept
        self.broken_proposals = broken_proposals
        self.calls: list[dict] = []

    async def parse(self, **options):
        self.calls.append(options)
        payload = json.loads(str(options["input"]))
        window_id = payload["window_id"]
        if options["text_format"] is WindowProposal:
            if self.broken_proposals:
                parsed = WindowProposal(
                    window_id=window_id,
                    examples=[ProposedExample(message_indices=[99], draft="broken", note="broken")],
                )
            else:
                groups: list[list[int]] = []
                for entry in payload["messages"]:
                    if entry["who"] != "Me":
                        continue
                    if groups and entry["i"] == groups[-1][-1] + 1:
                        groups[-1].append(entry["i"])
                    else:
                        groups.append([entry["i"]])
                parsed = WindowProposal(
                    window_id=window_id,
                    examples=[
                        ProposedExample(
                            message_indices=group,
                            draft=f"Draft for {'/'.join(map(str, group))}.",
                            note="owner confirms plans",
                        )
                        for group in groups
                    ],
                )
        else:
            parsed = WindowJudgement(
                window_id=window_id,
                verdicts=[
                    ExampleVerdict(
                        example_id=candidate["example_id"],
                        accept=self.accept,
                        reason="meaning preserved" if self.accept else "meaning changed",
                    )
                    for candidate in payload["candidates"]
                ],
            )
        return SimpleNamespace(
            output=[SimpleNamespace(type="message", content=[SimpleNamespace(parsed=parsed)])],
            usage=SimpleNamespace(input_tokens=20, output_tokens=8, total_tokens=28),
        )


def _patch_client(monkeypatch, responses: FakeResponses) -> None:
    class FakeClient:
        def __init__(self, **_options):
            self.responses = responses

        async def close(self):
            return None

    monkeypatch.setattr(llm_dataset, "AsyncOpenAI", FakeClient)


def _run(messages: Path, output: Path, **overrides) -> dict:
    options = {"api_key": "test-key", "model": "gpt-test", **overrides}
    return asyncio.run(build_llm_dataset(messages, output, **options))


def test_windows_are_mechanical_slices_with_labels() -> None:
    messages = _conversation() + [
        _message(9, chat="chat-2", participant="p-9", name=None, at_seconds=0),
        _message(10, chat="chat-2", participant="p-9", name=None, at_seconds=5),
        _message(11, chat="chat-2", participant="p-8", name="Colin Diff", at_seconds=9),
    ]
    windows = build_windows(messages, max_window_messages=2)

    assert [len(window["rows"]) for window in windows] == [2, 2, 2, 1]
    first = windows[0]["payload"]["messages"]
    assert [entry["who"] for entry in first] == ["Colin", "Me"]
    assert all("at" in entry and "text" in entry for entry in first)
    chat_two = windows[2]["payload"]["messages"]
    assert [entry["who"] for entry in chat_two] == ["P1", "P1"]
    assert windows[0]["has_me"] is True
    assert windows[2]["has_me"] is False


def test_proposal_validation_is_mechanical_only() -> None:
    window = build_windows(_conversation())[0]

    valid = validate_window_proposal(
        window,
        WindowProposal(
            window_id=str(window["window_id"]),
            examples=[ProposedExample(message_indices=[1, 2], draft="I am on my way.", note="eta")],
        ),
    )
    assert valid[0]["example_id"] == example_id(str(window["window_id"]), [1, 2])
    assert valid[0]["draft"] == "I am on my way."

    for bad in (
        [ProposedExample(message_indices=[], draft="x", note="")],
        [ProposedExample(message_indices=[0], draft="x", note="")],
        [ProposedExample(message_indices=[2, 1], draft="x", note="")],
        [ProposedExample(message_indices=[1, 99], draft="x", note="")],
        [ProposedExample(message_indices=[1], draft=" ", note="")],
        [
            ProposedExample(message_indices=[1], draft="x", note=""),
            ProposedExample(message_indices=[1, 2], draft="y", note=""),
        ],
    ):
        with pytest.raises(WindowResponseError):
            validate_window_proposal(
                window,
                WindowProposal(window_id=str(window["window_id"]), examples=bad),
            )

    with pytest.raises(WindowResponseError, match="window identifier"):
        validate_window_proposal(
            window,
            WindowProposal(window_id="other", examples=[]),
        )


def test_judgement_must_cover_every_candidate_exactly_once() -> None:
    window = build_windows(_conversation())[0]
    examples = [
        {"example_id": example_id(str(window["window_id"]), [1, 2]), "message_indices": [1, 2]}
    ]

    verdicts = validate_window_judgement(
        window,
        WindowJudgement(
            window_id=str(window["window_id"]),
            verdicts=[
                ExampleVerdict(example_id=str(examples[0]["example_id"]), accept=True, reason="ok")
            ],
        ),
        examples,
    )
    assert verdicts[str(examples[0]["example_id"])]["accept"] is True

    for verdict_list in (
        [],
        [ExampleVerdict(example_id="unknown", accept=True, reason="")],
        [
            ExampleVerdict(example_id=str(examples[0]["example_id"]), accept=True, reason=""),
            ExampleVerdict(example_id=str(examples[0]["example_id"]), accept=False, reason=""),
        ],
    ):
        with pytest.raises(WindowResponseError):
            validate_window_judgement(
                window,
                WindowJudgement(window_id=str(window["window_id"]), verdicts=verdict_list),
                examples,
            )


def test_build_llm_dataset_records_accepted_pairs(tmp_path: Path, monkeypatch) -> None:
    messages = tmp_path / "messages.jsonl"
    output = tmp_path / "run"
    write_jsonl(
        messages,
        _conversation()
        + [_message(9, chat="chat-2", participant="p-9", name="Sarah Sun", at_seconds=0)],
    )
    responses = FakeResponses()
    _patch_client(monkeypatch, responses)

    report = _run(messages, output)

    assert report["windows"]["total"] == 2
    assert report["windows"]["skipped_no_owner_messages"] == 1
    assert report["windows"]["selected"] == 1
    assert report["windows"]["completed"] == 1
    assert report["examples"] == {
        "proposed": 1,
        "accepted": 1,
        "rejected": 0,
        "acceptance_rate": 1.0,
    }
    assert len(responses.calls) == 2
    assert "dataset" not in {path.name for path in output.iterdir()}

    accepted = load_accepted_examples(output / "results.jsonl")
    assert len(accepted) == 1
    assert accepted[0]["target"] == "ya omw\nbe there in 10"
    assert accepted[0]["source"] == "Draft for 1/2."
    assert accepted[0]["message_ids"] == ["m-2", "m-3"]

    report_text = (output / "report.json").read_text(encoding="utf-8")
    assert "ya omw" not in report_text

    fresh = FakeResponses()
    _patch_client(monkeypatch, fresh)
    resumed = _run(messages, output)
    assert fresh.calls == []
    assert resumed["windows"]["resumed_completed"] == 1

    with pytest.raises(LlmDatasetResumeError, match="incompatible"):
        _run(messages, output, model="different-model")


def test_rejected_pairs_are_recorded_but_not_accepted(tmp_path: Path, monkeypatch) -> None:
    messages = tmp_path / "messages.jsonl"
    output = tmp_path / "run"
    write_jsonl(messages, _conversation())
    _patch_client(monkeypatch, FakeResponses(accept=False))

    report = _run(messages, output)

    assert report["examples"]["accepted"] == 0
    assert report["examples"]["rejected"] == 1
    results = list(read_jsonl(output / "results.jsonl"))
    assert results[0]["examples"][0]["accepted"] is False
    assert results[0]["examples"][0]["judge_reason"] == "meaning changed"
    assert load_accepted_examples(output / "results.jsonl") == []


def test_failed_windows_are_reported_without_fallback(tmp_path: Path, monkeypatch) -> None:
    messages = tmp_path / "messages.jsonl"
    output = tmp_path / "run"
    write_jsonl(messages, _conversation())
    responses = FakeResponses(broken_proposals=True)
    _patch_client(monkeypatch, responses)

    report = _run(messages, output, max_attempts=2)

    assert len(responses.calls) == 2
    second_payload = json.loads(str(responses.calls[1]["input"]))
    assert "WindowResponseError" in second_payload["previous_attempt_error"]
    assert report["windows"]["proposal_failed"] == 1
    assert report["windows"]["completed"] == 0
    results = list(read_jsonl(output / "results.jsonl"))
    assert results[0]["status"] == "proposal_failed"
    assert results[0]["examples"] == []
    assert load_accepted_examples(output / "results.jsonl") == []


def test_resume_rejects_changed_messages(tmp_path: Path, monkeypatch) -> None:
    messages = tmp_path / "messages.jsonl"
    output = tmp_path / "run"
    write_jsonl(messages, _conversation())
    _patch_client(monkeypatch, FakeResponses())
    _run(messages, output)

    changed = _conversation()
    changed[1]["text"] = "completely different"
    write_jsonl(messages, changed)
    with pytest.raises(LlmDatasetResumeError, match="unknown windows"):
        _run(messages, output)


def test_review_renders_sample_without_leaking_into_summary(tmp_path: Path, monkeypatch) -> None:
    messages = tmp_path / "messages.jsonl"
    output = tmp_path / "run"
    write_jsonl(messages, _conversation())
    _patch_client(monkeypatch, FakeResponses())
    _run(messages, output)

    summary = create_llm_dataset_review(
        output / "results.jsonl",
        output / "review.md",
        sample_size=10,
    )

    assert summary["examples"] == 1
    assert summary["accepted"] == 1
    review_text = (output / "review.md").read_text(encoding="utf-8")
    assert "ya omw" in review_text
    assert "Draft for 1/2." in review_text
    assert "owner confirms plans" in review_text
