import json
from pathlib import Path
from types import SimpleNamespace

from imessage_mlx.data.sessions import iter_message_sessions
from imessage_mlx.data.stamp import (
    DEFAULT_JUDGE_MODEL,
    BoundaryDecision,
    Bubble,
    CandidateChain,
    ChainDecision,
    OpenAIModelJudge,
    find_candidate_chains,
    prepare_stamp_corpus,
    stamp_session_id,
)
from imessage_mlx.utils import read_jsonl, sha256_text, write_jsonl

MINUTE_NS = 60 * 1_000_000_000


def _message(
    message_id: str,
    *,
    timestamp_ns: int,
    sender_role: str = "me",
    text: str,
    chat_id: str = "chat-a",
) -> dict[str, object]:
    return {
        "message_id": message_id,
        "chat_id": chat_id,
        "timestamp_ns": timestamp_ns,
        "sender_role": sender_role,
        "participant_id": "me" if sender_role == "me" else "person-a",
        "text": text,
        "is_group": False,
    }


def _session_ids(messages: list[dict[str, object]]) -> list[str]:
    return [
        stamp_session_id(chat_id, session)
        for chat_id, session in iter_message_sessions(messages)
    ]


def _write_sft(
    tmp_path: Path,
    *,
    train_sessions: list[str],
    heldout_sessions: list[str] | None = None,
) -> tuple[Path, Path]:
    train = tmp_path / "sft-train.jsonl"
    validation = tmp_path / "sft-validation.jsonl"
    write_jsonl(train, [{"session_id": session_id} for session_id in train_sessions])
    write_jsonl(
        validation,
        [{"session_id": session_id} for session_id in (heldout_sessions or [])],
    )
    return train, validation


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "train": tmp_path / "stamp-train.jsonl",
        "validation": tmp_path / "stamp-validation.jsonl",
        "test": tmp_path / "stamp-test.jsonl",
        "report": tmp_path / "stamp-report.json",
        "artifact": tmp_path / "stamp-judge.jsonl",
    }


def _prepare(
    tmp_path: Path,
    messages: list[dict[str, object]],
    *,
    judge: object | None = None,
    train_sessions: list[str] | None = None,
    heldout_sessions: list[str] | None = None,
    **kwargs: object,
) -> tuple[dict[str, object], dict[str, Path]]:
    source = tmp_path / "messages.jsonl"
    write_jsonl(source, messages)
    all_session_ids = _session_ids(messages)
    sft_train, sft_validation = _write_sft(
        tmp_path,
        train_sessions=train_sessions if train_sessions is not None else all_session_ids,
        heldout_sessions=heldout_sessions,
    )
    paths = _paths(tmp_path)
    report = prepare_stamp_corpus(
        source,
        sft_train,
        sft_validation,
        paths["train"],
        paths["validation"],
        paths["test"],
        paths["report"],
        judge=judge,
        judge_artifact_path=paths["artifact"],
        **kwargs,
    )
    return report, paths


class FakeJudge:
    cache_key = "tests.fake-judge-v1"

    def __init__(
        self,
        rules: dict[tuple[str, ...], tuple[BoundaryDecision, ...]],
    ) -> None:
        self.rules = rules
        self.calls = 0

    def judge(self, candidates: list[CandidateChain]) -> list[ChainDecision]:
        self.calls += 1
        return [
            ChainDecision(
                candidate_id=candidate.candidate_id,
                boundaries=self.rules[
                    tuple(bubble.text for bubble in candidate.bubbles)
                ],
            )
            for candidate in candidates
        ]


class FailingJudge:
    cache_key = "tests.failing-judge-v1"

    def judge(self, candidates: list[CandidateChain]) -> object:
        raise RuntimeError(f"unavailable for {len(candidates)} candidates")


def test_candidate_prefilter_requires_adjacent_close_outgoing_messages() -> None:
    messages = [
        _message("a", timestamp_ns=0, text="first"),
        _message("b", timestamp_ns=MINUTE_NS, text="continuation"),
        _message(
            "incoming",
            timestamp_ns=2 * MINUTE_NS,
            sender_role="other",
            text="interrupting",
        ),
        _message("c", timestamp_ns=3 * MINUTE_NS, text="after incoming"),
        _message("d", timestamp_ns=10 * MINUTE_NS, text="too late"),
        _message("e", timestamp_ns=11 * MINUTE_NS, text="<|attachment|>"),
        _message("f", timestamp_ns=12 * MINUTE_NS, text="not adjacent after cleanup"),
    ]

    candidates = find_candidate_chains(messages, merge_gap_minutes=2)

    assert len(candidates) == 1
    assert [bubble.message_id for bubble in candidates[0].bubbles] == ["a", "b"]
    assert candidates[0].session_id == stamp_session_id("chat-a", messages)


def test_model_driven_boundaries_create_merged_and_split_units(tmp_path: Path) -> None:
    messages = [
        _message("a", timestamp_ns=0, text="first thought"),
        _message("b", timestamp_ns=MINUTE_NS, text="continued"),
        _message("c", timestamp_ns=2 * MINUTE_NS, text="separate thought"),
    ]
    judge = FakeJudge(
        {
            ("first thought", "continued", "separate thought"): (
                BoundaryDecision(True, "continues the same utterance"),
                BoundaryDecision(False, "new conversational act"),
            )
        }
    )

    report, paths = _prepare(tmp_path, messages, judge=judge)
    records = list(read_jsonl(paths["train"]))

    assert judge.calls == 1
    assert report["merged_boundaries"] == 1
    assert [record["target"] for record in records] == [
        "first thought\ncontinued",
        "separate thought",
    ]
    merged = next(record for record in records if len(record["bubbles"]) == 2)
    assert merged["bubbles"] == ["first thought", "continued"]
    assert merged["merge_reasons"] == ["continues the same utterance"]


def test_judge_failure_fails_closed_to_individual_bubbles(tmp_path: Path) -> None:
    messages = [
        _message("a", timestamp_ns=0, text="one"),
        _message("b", timestamp_ns=MINUTE_NS, text="two"),
        _message("c", timestamp_ns=2 * MINUTE_NS, text="three"),
    ]

    report, paths = _prepare(tmp_path, messages, judge=FailingJudge())
    records = list(read_jsonl(paths["train"]))
    artifacts = list(read_jsonl(paths["artifact"]))

    assert sorted(record["target"] for record in records) == ["one", "three", "two"]
    assert all(record["merge_reasons"] == [] for record in records)
    assert report["failed_batches"] == 1
    assert report["merged_boundaries"] == 0
    assert artifacts[0]["status"] == "failed"
    assert all(
        not boundary["merge"]
        for decision in artifacts[0]["decisions"]
        for boundary in decision["boundaries"]
    )


def test_failed_judge_batch_is_retried_on_resume(tmp_path: Path) -> None:
    messages = [
        _message("a", timestamp_ns=0, text="one"),
        _message("b", timestamp_ns=MINUTE_NS, text="two"),
    ]
    _prepare(tmp_path, messages, judge=FailingJudge())
    recovered = FakeJudge(
        {("one", "two"): (BoundaryDecision(True, "same continuing thought"),)}
    )

    report, paths = _prepare(tmp_path, messages, judge=recovered)

    assert recovered.calls == 1
    assert report["failed_batches"] == 0
    assert [record["target"] for record in read_jsonl(paths["train"])] == ["one\ntwo"]
    artifacts = list(read_jsonl(paths["artifact"]))
    assert [record["status"] for record in artifacts] == ["failed", "completed"]


def test_dedupe_prefers_heldout_test_over_train(tmp_path: Path) -> None:
    messages = [
        _message("train", chat_id="chat-train", timestamp_ns=0, text="same exact reply"),
        _message("heldout", chat_id="chat-heldout", timestamp_ns=0, text="same exact reply"),
    ]
    session_ids = dict(
        (chat_id, stamp_session_id(chat_id, session))
        for chat_id, session in iter_message_sessions(messages)
    )

    report, paths = _prepare(
        tmp_path,
        messages,
        train_sessions=[session_ids["chat-train"]],
        heldout_sessions=[session_ids["chat-heldout"]],
        heldout_test_fraction=1,
    )

    assert list(read_jsonl(paths["train"])) == []
    test_records = list(read_jsonl(paths["test"]))
    assert len(test_records) == 1
    assert test_records[0]["session_id"] == session_ids["chat-heldout"]
    assert report["duplicates_dropped"] == {"train": 1, "validation": 0, "test": 0}
    assert report["configuration"]["dedupe_policy"].startswith("exact cleaned target")


def test_heldout_session_division_is_deterministic(tmp_path: Path) -> None:
    messages = [
        _message(
            f"message-{index}",
            chat_id=f"chat-{index:02d}",
            timestamp_ns=0,
            text=f"unique reply {index}",
        )
        for index in range(24)
    ]
    heldout = _session_ids(messages)

    _, paths = _prepare(
        tmp_path,
        messages,
        train_sessions=[],
        heldout_sessions=heldout,
        heldout_test_fraction=0.5,
    )
    first_validation = paths["validation"].read_bytes()
    first_test = paths["test"].read_bytes()

    _prepare(
        tmp_path,
        messages,
        train_sessions=[],
        heldout_sessions=heldout,
        heldout_test_fraction=0.5,
    )

    assert paths["validation"].read_bytes() == first_validation
    assert paths["test"].read_bytes() == first_test
    assert list(read_jsonl(paths["validation"]))
    assert list(read_jsonl(paths["test"]))
    assert not (
        {record["target"] for record in read_jsonl(paths["validation"])}
        & {record["target"] for record in read_jsonl(paths["test"])}
    )


def test_short_replies_survive_by_default_and_min_chars_is_optional(tmp_path: Path) -> None:
    messages = [
        _message("short", timestamp_ns=0, text="k"),
        _message(
            "separator",
            timestamp_ns=MINUTE_NS,
            sender_role="other",
            text="question",
        ),
        _message("placeholder", timestamp_ns=2 * MINUTE_NS, text="<|attachment|>"),
        _message(
            "separator-2",
            timestamp_ns=3 * MINUTE_NS,
            sender_role="other",
            text="another question",
        ),
        _message("emoji", timestamp_ns=4 * MINUTE_NS, text="👍"),
    ]

    report, paths = _prepare(tmp_path, messages)

    assert {record["target"] for record in read_jsonl(paths["train"])} == {"k", "👍"}
    assert report["dropped_empty_bubbles"] == 1
    assert report["dropped_below_min_chars"] == 0

    filtered_dir = tmp_path / "filtered"
    filtered_report, filtered_paths = _prepare(filtered_dir, messages, min_chars=2)

    assert list(read_jsonl(filtered_paths["train"])) == []
    assert filtered_report["dropped_empty_bubbles"] == 1
    assert filtered_report["dropped_below_min_chars"] == 2


def test_report_and_resume_artifact_are_private_and_text_free(tmp_path: Path) -> None:
    secret_a = "PRIVATE MESSAGE ALPHA"
    secret_b = "PRIVATE MESSAGE BETA"
    messages = [
        _message("a", timestamp_ns=0, text=secret_a),
        _message("b", timestamp_ns=MINUTE_NS, text=secret_b),
    ]
    judge = FakeJudge(
        {
            (secret_a, secret_b): (
                BoundaryDecision(True, "same utterance"),
            )
        }
    )

    first_report, paths = _prepare(
        tmp_path,
        messages,
        judge=judge,
        judge_batch_size=1,
    )
    artifact_before = paths["artifact"].read_bytes()
    second_report, _ = _prepare(
        tmp_path,
        messages,
        judge=judge,
        judge_batch_size=1,
    )

    assert judge.calls == 1
    assert second_report["judge_batches"] == 0
    assert second_report["resumed_batches"] == 1
    assert first_report["hashes"]["judge_artifact_sha256"] == second_report["hashes"][
        "judge_artifact_sha256"
    ]
    assert paths["artifact"].read_bytes() == artifact_before
    assert paths["report"].stat().st_mode & 0o777 == 0o600
    assert paths["artifact"].stat().st_mode & 0o777 == 0o600
    report_text = paths["report"].read_text(encoding="utf-8")
    artifact_text = paths["artifact"].read_text(encoding="utf-8")
    assert secret_a not in report_text
    assert secret_b not in report_text
    assert secret_a not in artifact_text
    assert secret_b not in artifact_text


def test_openai_judge_uses_configurable_strict_structured_output() -> None:
    candidate = CandidateChain(
        candidate_id="candidate",
        session_id="session",
        bubbles=(
            Bubble("a", 0, "one"),
            Bubble("b", MINUTE_NS, "two"),
        ),
    )

    class Responses:
        def __init__(self) -> None:
            self.request: dict[str, object] = {}

        def create(self, **kwargs: object) -> SimpleNamespace:
            self.request = kwargs
            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "decisions": [
                            {
                                "candidate_id": "candidate",
                                "boundaries": [{"merge": True, "reason": "continuation"}],
                            }
                        ]
                    }
                )
            )

    responses = Responses()
    judge = OpenAIModelJudge(client=SimpleNamespace(responses=responses))

    decisions = judge.judge([candidate])

    assert judge.model == DEFAULT_JUDGE_MODEL
    assert decisions[0].boundaries == (BoundaryDecision(True, "continuation"),)
    assert responses.request["model"] == DEFAULT_JUDGE_MODEL
    text_config = responses.request["text"]
    assert isinstance(text_config, dict)
    assert text_config["format"]["type"] == "json_schema"
    assert text_config["format"]["strict"] is True
    assert sha256_text(responses.request["input"])  # Prompt was supplied without persisting it.
