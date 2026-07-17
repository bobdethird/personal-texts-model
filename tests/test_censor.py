import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from imessage_mlx.data import censor
from imessage_mlx.data.censor import (
    CensorBatch,
    CensorResponseError,
    CensorResumeError,
    PairScreening,
    batch_identifier,
    censor_llm_dataset,
    create_censor_review,
    validate_censor_batch,
)
from imessage_mlx.utils import read_jsonl, write_jsonl

SECOND_NS = 1_000_000_000


def _results_record(examples: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "proposal_prompt_version": "llm-dataset-proposal-v1",
        "judge_prompt_version": "llm-dataset-judge-v1",
        "model": "gpt-test",
        "judge_model": "gpt-test",
        "window_id": "w-1",
        "window_fingerprint": "fp-1",
        "chat_id": "chat-1",
        "is_group": False,
        "window_message_count": 8,
        "status": "completed",
        "failure_reason": None,
        "examples": examples,
    }


def _example(number: int, *, accepted: bool = True, target: str = "ya omw") -> dict:
    return {
        "example_id": f"pair-{number}",
        "pair_id": f"pair-{number}",
        "message_indices": [number],
        "message_ids": [f"m-{number}"],
        "timestamp_ns": number * SECOND_NS,
        "target": target,
        "source": f"Draft {number}.",
        "note": f"note {number}",
        "accepted": accepted,
        "judge_reason": "ok",
    }


class FakeCensorResponses:
    """Excludes rows whose draft or original contains marker words; else allows."""

    def __init__(self, *, broken: bool = False, always_fail: bool = False) -> None:
        self.broken = broken
        self.always_fail = always_fail
        self.calls: list[dict] = []

    async def parse(self, **options):
        self.calls.append(options)
        if self.always_fail:
            raise RuntimeError("simulated transport failure")
        payload = json.loads(str(options["input"]))
        rows = payload["rows"]
        if self.broken:
            verdicts = [
                PairScreening(
                    pair_id="unknown-pair",
                    allowed=True,
                    category="none",
                    reason="broken",
                )
            ]
        else:
            verdicts = []
            for row in rows:
                text = f"{row['draft']} {row['original']}".lower()
                if "hunter2" in text:
                    verdicts.append(
                        PairScreening(
                            pair_id=row["pair_id"],
                            allowed=False,
                            category="sensitive_secret",
                            reason="contains a password value",
                        )
                    )
                elif "explicit" in text:
                    verdicts.append(
                        PairScreening(
                            pair_id=row["pair_id"],
                            allowed=False,
                            category="adult_content",
                            reason="sexual material",
                        )
                    )
                else:
                    verdicts.append(
                        PairScreening(
                            pair_id=row["pair_id"],
                            allowed=True,
                            category="none",
                            reason="clean",
                        )
                    )
        parsed = CensorBatch(batch_id=payload["batch_id"], verdicts=verdicts)
        return SimpleNamespace(
            output=[SimpleNamespace(type="message", content=[SimpleNamespace(parsed=parsed)])],
            usage=SimpleNamespace(input_tokens=30, output_tokens=10, total_tokens=40),
        )


def _patch_client(monkeypatch, responses: FakeCensorResponses) -> None:
    class FakeClient:
        def __init__(self, **_options):
            self.responses = responses

        async def close(self):
            return None

    monkeypatch.setattr(censor, "AsyncOpenAI", FakeClient)


def _run(results: Path, output: Path, **overrides) -> dict:
    options = {"api_key": "test-key", "model": "censor-test", **overrides}
    return asyncio.run(censor_llm_dataset(results, output, **options))


def test_batch_validation_is_mechanical_only() -> None:
    pairs = [{"pair_id": "pair-1"}, {"pair_id": "pair-2"}]
    batch_id = batch_identifier(["pair-1", "pair-2"])

    def batch(verdicts: list[PairScreening]) -> CensorBatch:
        return CensorBatch(batch_id=batch_id, verdicts=verdicts)

    allowed = PairScreening(pair_id="pair-1", allowed=True, category="none", reason="ok")
    excluded = PairScreening(
        pair_id="pair-2", allowed=False, category="adult_content", reason="bad"
    )
    verdicts = validate_censor_batch(
        batch([allowed, excluded]), expected_batch_id=batch_id, pairs=pairs
    )
    assert verdicts["pair-1"]["allowed"] is True
    assert verdicts["pair-2"]["category"] == "adult_content"

    for bad in (
        [],
        [allowed],
        [allowed, allowed],
        [
            allowed,
            PairScreening(pair_id="pair-9", allowed=True, category="none", reason=""),
        ],
        [
            allowed,
            PairScreening(pair_id="pair-2", allowed=True, category="adult_content", reason=""),
        ],
        [
            allowed,
            PairScreening(pair_id="pair-2", allowed=False, category="none", reason=""),
        ],
    ):
        with pytest.raises(CensorResponseError):
            validate_censor_batch(batch(bad), expected_batch_id=batch_id, pairs=pairs)

    with pytest.raises(CensorResponseError, match="batch identifier"):
        validate_censor_batch(
            CensorBatch(batch_id="other", verdicts=[allowed, excluded]),
            expected_batch_id=batch_id,
            pairs=pairs,
        )


def test_censor_excludes_by_category_and_publishes_only_allowed(
    tmp_path: Path, monkeypatch
) -> None:
    results = tmp_path / "results.jsonl"
    output = tmp_path / "run"
    write_jsonl(
        results,
        [
            _results_record(
                [
                    _example(1),
                    _example(2, target="my wifi password is hunter2"),
                    _example(3, target="that was explicit last night"),
                    _example(4, accepted=False, target="rejected by judge"),
                    _example(5),
                ]
            )
        ],
    )
    responses = FakeCensorResponses()
    _patch_client(monkeypatch, responses)

    report = _run(results, output, valid_fraction=0.0, test_fraction=0.0)

    assert report["pairs"]["accepted_input"] == 4
    assert report["pairs"]["allowed"] == 2
    assert report["pairs"]["excluded"] == 2
    assert report["pairs"]["excluded_by_category"] == {
        "adult_content": 1,
        "sensitive_secret": 1,
    }
    assert report["pairs"]["screening_failures_excluded"] == 0
    assert report["splits"] == {"train": 2, "valid": 0, "test": 0}
    assert report["published_splits_censored_only"] is True

    published = list(read_jsonl(output / "dataset/train.jsonl"))
    assert [row["pair_id"] for row in published] == ["pair-1", "pair-5"]
    assert all("hunter2" not in row["target"] for row in published)
    bart_rows = list(read_jsonl(output / "dataset/bart/train.jsonl"))
    assert {row["pair_id"] for row in bart_rows} == {"pair-1", "pair-5"}
    mlx_rows = list(read_jsonl(output / "dataset/mlx/train.jsonl"))
    assert all(row["prompt"].startswith("Rewrite the draft") for row in mlx_rows)

    report_text = (output / "censor-report.json").read_text(encoding="utf-8")
    assert "hunter2" not in report_text
    review_text = (output / "censor-review.md").read_text(encoding="utf-8")
    assert "hunter2" in review_text
    assert "sensitive_secret" in review_text
    assert "adult_content" in review_text
    assert "Draft 1." not in review_text


def test_censor_fails_closed_and_resumes(tmp_path: Path, monkeypatch) -> None:
    results = tmp_path / "results.jsonl"
    output = tmp_path / "run"
    write_jsonl(results, [_results_record([_example(1), _example(2)])])
    _patch_client(monkeypatch, FakeCensorResponses(always_fail=True))

    report = _run(results, output, max_attempts=2)

    assert report["pairs"]["allowed"] == 0
    assert report["pairs"]["screening_failures_excluded"] == 2
    assert report["pairs"]["excluded_by_category"] == {"unscreened_failure": 2}
    assert report["splits"] == {"train": 0, "valid": 0, "test": 0}
    assert list(read_jsonl(output / "dataset/train.jsonl")) == []

    # A transient failure must not permanently exclude a pair: a later healthy run
    # re-screens the previously failed pairs rather than skipping them.
    fresh = FakeCensorResponses()
    _patch_client(monkeypatch, fresh)
    resumed = _run(results, output, max_attempts=2)
    assert fresh.calls  # failed pairs were re-screened, not skipped
    assert resumed["pairs"]["allowed"] == 2
    assert resumed["pairs"]["screening_failures_excluded"] == 0
    assert resumed["splits"] == {"train": 2, "valid": 0, "test": 0}

    with pytest.raises(CensorResumeError, match="incompatible"):
        _run(results, output, model="different-censor")


def test_broken_responses_are_retried_with_error_feedback(tmp_path: Path, monkeypatch) -> None:
    results = tmp_path / "results.jsonl"
    output = tmp_path / "run"
    write_jsonl(results, [_results_record([_example(1)])])
    responses = FakeCensorResponses(broken=True)
    _patch_client(monkeypatch, responses)

    report = _run(results, output, max_attempts=2)

    assert len(responses.calls) == 2
    second_payload = json.loads(str(responses.calls[1]["input"]))
    assert "CensorResponseError" in second_payload["previous_attempt_error"]
    assert report["pairs"]["screening_failures_excluded"] == 1


def test_review_caps_rendered_rows(tmp_path: Path) -> None:
    censor_file = tmp_path / "censor.jsonl"
    write_jsonl(
        censor_file,
        [
            {
                "pair_id": f"pair-{number}",
                "timestamp_ns": number * SECOND_NS,
                "source": f"draft {number}",
                "target": f"target {number}",
                "allowed": False,
                "category": "adult_content",
                "reason": "test",
            }
            for number in range(5)
        ],
    )

    summary = create_censor_review(censor_file, tmp_path / "review.md", max_rows=3)

    assert summary["excluded"] == 5
    assert summary["shown"] == 3
    assert summary["excluded_by_category"] == {"adult_content": 5}
