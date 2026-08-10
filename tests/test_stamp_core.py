import json
import stat
from pathlib import Path

import pytest

from imessage_mlx.stamp.classifier import (
    binary_classification_metrics,
    build_classifier_examples,
)
from imessage_mlx.stamp.evaluate import (
    evaluate_candidate_rows,
    evaluate_model_candidates,
    structural_correlation,
)
from imessage_mlx.stamp.neutralize import (
    NEUTRAL_PAIR_FORMAT,
    NeutralValidationConfig,
    build_neutralization_messages,
    make_neutral_pair,
    normalized_numbers,
    parse_neutral_output,
    run_neutralization,
    token_budget_slices,
    validate_neutral,
)
from imessage_mlx.stamp.preference import (
    CPO_PAIR_FORMAT,
    cpo_pair_loss,
    make_cpo_pair,
    make_cpo_pair_from_selection,
)
from imessage_mlx.stamp.rewards import (
    CandidateReward,
    RewardExponents,
    dynamic_reward_exponents,
    normalized_base_model_likelihood,
    select_hope_and_fear,
)
from imessage_mlx.stamp.sft import (
    build_sft_example,
    parse_rewrite_output,
)
from imessage_mlx.utils import read_jsonl


def _neutral_pair(pair_id: str = "pair-1") -> dict[str, object]:
    return make_neutral_pair(
        pair_id=pair_id,
        original_bubbles=["cant make it at 7", "sry <|attachment|>"],
        neutral_bubbles=["I cannot make it at 7.", "Sorry. <|attachment|>"],
        split="train",
    )


def _reward(
    candidate_id: str,
    *,
    text: str | None = None,
    style: float = 0.8,
    semantic: float = 0.8,
    likelihood: float = 0.8,
    length: float = 1.0,
) -> CandidateReward:
    return CandidateReward(
        candidate_id=candidate_id,
        text=text or candidate_id,
        style_probability=style,
        semantic_similarity=semantic,
        base_model_likelihood=likelihood,
        length_ratio=length,
    )


def test_neutralization_prompt_combines_a_multi_bubble_thought() -> None:
    messages = build_neutralization_messages(
        ["cant go at 7", "sry"],
        context_bubbles=["are you coming?"],
    )
    assert [message["role"] for message in messages] == ["system", "user"]
    assert '"cant go at 7", "sry"' in messages[1]["content"]
    assert "2 TARGET bubble(s)" in messages[1]["content"]
    assert "Combine the burst into one neutral draft" in messages[1]["content"]
    assert parse_neutral_output(
        '```json\n{"neutral_text":"I cannot go at 7, and I am sorry."}\n```'
    ) == ["I cannot go at 7, and I am sorry."]


def test_multi_bubble_original_can_use_one_neutral_draft() -> None:
    original = ["probably after six", "want to grab dinner?"]
    neutral = ["Would you like to have dinner sometime after six?"]
    validation = validate_neutral(original, neutral)
    pair = make_neutral_pair(
        pair_id="burst",
        original_bubbles=original,
        neutral_bubbles=neutral,
        validation=validation,
    )

    example = build_sft_example(pair)

    assert validation.valid
    assert len(example["prompt"]) == 2
    assert json.loads(example["completion"][0]["content"]) == {
        "original_bubbles": original
    }


def test_neutral_validation_checks_content_invariants() -> None:
    valid = validate_neutral(
        ["I won't pay $3,000 for <|attachment|>"],
        ["I will not pay 3k for <|attachment|>."],
    )
    assert valid.valid
    assert normalized_numbers("3k at 10:00") == normalized_numbers("3,000 at 10")

    invalid = validate_neutral(
        ["I won't pay $3,000 for <|attachment|>"],
        [""],
    )
    assert {
        "empty_bubble",
        "numbers_changed",
        "negation_changed",
        "placeholders_changed",
        "length_out_of_bounds",
    } <= set(invalid.errors)


def test_neutral_validation_rejects_lost_first_person_and_reported_speech() -> None:
    lost_person = validate_neutral(
        ["So like I'm not really up that much"],
        ["The overall change is not significant"],
    )
    assert "person_changed" in lost_person.errors

    narrated = validate_neutral(
        ["Sarah can I have money"],
        ["Sarah, the sender asks if he can have money"],
    )
    assert "reported_speech" in narrated.errors

    kept = validate_neutral(
        ["So like I'm not really up that much"],
        ["I have not gained very much"],
    )
    assert kept.valid


def test_neutral_validation_reports_copies_without_rejecting_them() -> None:
    copied = validate_neutral(["Collin has your card"], ["Collin has your card"])
    assert copied.valid
    assert copied.unchanged

    # Punctuation and casing alone are not a rewrite.
    restyled = validate_neutral(["on the elevator"], ["On the elevator."])
    assert restyled.unchanged

    rewritten = validate_neutral(["on the elevator"], ["I am in the elevator."])
    assert rewritten.valid
    assert not rewritten.unchanged

    strict = validate_neutral(
        ["Collin has your card"],
        ["Collin has your card"],
        config=NeutralValidationConfig(reject_unchanged=True),
    )
    assert strict.errors == ("unchanged_text",)


def test_neutral_validation_optional_semantic_threshold() -> None:
    config = NeutralValidationConfig(min_semantic_similarity=0.9)
    missing = validate_neutral("hello there", "hello", config=config)
    assert missing.errors == ("semantic_score_missing",)
    low = validate_neutral(
        "hello there",
        "hello",
        config=config,
        semantic_scorer=lambda _left, _right: 0.4,
    )
    assert "semantic_similarity_too_low" in low.errors


def test_neutralization_runner_resumes_and_writes_private_file(tmp_path: Path) -> None:
    output = tmp_path / "private" / "pairs.jsonl"
    calls = 0

    def generator(_messages: list[dict[str, str]]) -> str:
        nonlocal calls
        calls += 1
        return '{"neutral_bubbles":["I cannot come at 7."]}'

    source = [{"pair_id": "p1", "reply": "cant come at 7", "split": "train"}]
    first = run_neutralization(source, output, generator)
    second = run_neutralization(source, output, generator)

    assert first["generated"] == 1
    assert second["skipped_existing"] == 1
    assert calls == 1
    assert list(read_jsonl(output))[0]["format"] == NEUTRAL_PAIR_FORMAT
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_batched_neutralization_groups_records_into_one_call(tmp_path: Path) -> None:
    output = tmp_path / "private" / "pairs.jsonl"
    batch_sizes: list[int] = []
    progress: list[int] = []

    def batch_generator(prompts: list[list[dict[str, str]]]) -> list[str]:
        batch_sizes.append(len(prompts))
        return ['{"neutral_bubbles":["I cannot come at 7."]}'] * len(prompts)

    source = [
        {"pair_id": f"p{index}", "reply": "cant come at 7", "split": "train"}
        for index in range(5)
    ]
    summary = run_neutralization(
        source,
        output,
        batch_generator=batch_generator,
        batch_size=2,
        on_progress=lambda state: progress.append(int(state["generated"])),
    )

    assert summary["generated"] == 5
    assert batch_sizes == [2, 2, 1]
    assert progress == [2, 4, 5]
    assert len(list(read_jsonl(output))) == 5
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_batched_neutralization_retries_only_the_invalid_records(tmp_path: Path) -> None:
    output = tmp_path / "pairs.jsonl"
    seen: list[int] = []

    def batch_generator(prompts: list[list[dict[str, str]]]) -> list[str]:
        seen.append(len(prompts))
        # The first pass drops a required number from one record, so only that
        # record should be retried.
        if len(seen) == 1:
            return [
                '{"neutral_bubbles":["I cannot come at 7."]}',
                '{"neutral_bubbles":["I cannot come."]}',
            ]
        return ['{"neutral_bubbles":["I cannot come at 7."]}'] * len(prompts)

    source = [
        {"pair_id": "p0", "reply": "cant come at 7", "split": "train"},
        {"pair_id": "p1", "reply": "cant come at 7", "split": "train"},
    ]
    summary = run_neutralization(
        source,
        output,
        batch_generator=batch_generator,
        batch_size=2,
    )

    assert seen == [2, 1]
    assert summary["generated"] == 2
    assert summary["failed"] == 0


def test_batched_neutralization_varies_retries_and_keeps_rejected_drafts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "pairs.jsonl"
    attempts: list[int] = []

    def batch_generator(
        prompts: list[list[dict[str, str]]],
        *,
        attempt: int,
    ) -> list[str]:
        attempts.append(attempt)
        if attempt == 0:
            return ['{"neutral_bubbles":["I cannot come at 8."]}'] * len(prompts)
        return ['{"neutral_bubbles":["I cannot come at 7."]}'] * len(prompts)

    source = [{"pair_id": "p0", "reply": "cant come at 7", "split": "train"}]
    summary = run_neutralization(
        source,
        output,
        batch_generator=batch_generator,
        batch_size=1,
    )

    assert attempts == [0, 1]
    assert summary["generated"] == 1
    assert summary["failed"] == 0


def test_failures_record_the_rejected_draft(tmp_path: Path) -> None:
    output = tmp_path / "pairs.jsonl"

    def batch_generator(prompts: list[list[dict[str, str]]]) -> list[str]:
        return ['{"neutral_bubbles":["I cannot come at 7 and 8."]}'] * len(prompts)

    summary = run_neutralization(
        [{"pair_id": "p0", "reply": "cant come at 7", "split": "train"}],
        output,
        batch_generator=batch_generator,
        batch_size=1,
        max_attempts=1,
    )

    failure = summary["failures"][0]
    assert "numbers_changed" in failure["errors"]
    assert failure["original"] == ["cant come at 7"]
    assert failure["rejected"] == ["I cannot come at 7 and 8."]


def test_token_budget_slices_bound_the_padded_batch_cost() -> None:
    # Uniform short prompts pack together, since 4 * 100 stays under budget.
    assert token_budget_slices([100] * 4, 1000) == [(0, 4)]

    # One long prompt pads everything beside it, so it must batch alone.
    spans = token_budget_slices([100, 900, 100, 100], 1000)
    assert spans == [(0, 1), (1, 2), (2, 4)]
    for start, end in spans:
        lengths = [100, 900, 100, 100][start:end]
        assert max(lengths) * len(lengths) <= 1000

    # A single prompt over budget still has to run, rather than looping forever.
    assert token_budget_slices([5000], 1000) == [(0, 1)]
    assert token_budget_slices([], 1000) == []

    with pytest.raises(ValueError):
        token_budget_slices([10], 0)


def test_unchanged_pairs_are_kept_but_capped(tmp_path: Path) -> None:
    output = tmp_path / "pairs.jsonl"

    def batch_generator(prompts: list[list[dict[str, str]]]) -> list[str]:
        outputs = []
        for prompt in prompts:
            target = json.loads(prompt[-1]["content"].split("TARGET:\n")[1].split("\n\n")[0])
            reply = target[0]
            # Half the corpus is already neutral and echoes back unchanged.
            draft = reply if reply.startswith("plain") else reply.replace("msg", "message")
            outputs.append(json.dumps({"neutral_bubbles": [draft]}))
        return outputs

    source = [
        {
            "pair_id": f"p{index}",
            "reply": ("plain" if index % 2 else "msg") + f" {chr(97 + index)}",
            "split": "train",
        }
        for index in range(20)
    ]
    summary = run_neutralization(
        source,
        output,
        batch_generator=batch_generator,
        batch_size=4,
        max_attempts=1,
        max_unchanged_ratio=0.25,
    )

    stored = [json.loads(line) for line in output.read_text().splitlines()]
    unchanged = [pair for pair in stored if pair["validation"]["unchanged"]]

    assert len(stored) == summary["generated"]
    assert len(unchanged) == summary["unchanged_pairs"]
    # Every real rewrite is kept, while copies are held to the configured share.
    assert summary["generated"] - len(unchanged) == 10
    assert 0 < len(unchanged) <= 0.25 * len(stored) + 1
    assert all("unchanged_over_cap" in f["errors"] for f in summary["failures"])


def test_batched_neutralization_records_persistently_invalid_records(tmp_path: Path) -> None:
    output = tmp_path / "pairs.jsonl"

    def batch_generator(prompts: list[list[dict[str, str]]]) -> list[str]:
        return ['{"neutral_bubbles":["I cannot come."]}'] * len(prompts)

    source = [{"pair_id": "p0", "reply": "cant come at 7", "split": "train"}]
    summary = run_neutralization(
        source,
        output,
        batch_generator=batch_generator,
        batch_size=4,
        max_attempts=2,
    )

    assert summary["generated"] == 0
    assert summary["failed"] == 1
    assert summary["failures"][0]["pair_id"] == "p0"
    assert not output.exists()


def test_neutralization_requires_a_generator(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="generator"):
        run_neutralization([], tmp_path / "pairs.jsonl")


def test_classifier_examples_are_balanced_and_metrics_include_auc() -> None:
    examples = build_classifier_examples([_neutral_pair()])
    assert [example["label"] for example in examples] == [1, 0]
    assert examples[0]["text"].startswith("cant make it")
    metrics = binary_classification_metrics([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_sft_example_is_conversational_prompt_completion() -> None:
    example = build_sft_example(_neutral_pair())
    assert [message["role"] for message in example["prompt"]] == ["system", "user"]
    completion = json.loads(example["completion"][0]["content"])
    assert completion["original_bubbles"] == [
        "cant make it at 7",
        "sry <|attachment|>",
    ]
    assert parse_rewrite_output(
        example["completion"][0]["content"],
        expected_bubbles=2,
    ) == completion["original_bubbles"]


def test_rewards_are_bounded_and_dynamic_exponents_follow_reversals() -> None:
    assert normalized_base_model_likelihood(-2.0, token_count=2) == pytest.approx(
        0.367879,
        rel=1e-5,
    )
    preferred = _reward("preferred", style=0.9, semantic=0.3, likelihood=0.8)
    rejected = _reward("rejected", style=0.2, semantic=0.9, likelihood=0.7)
    exponents = dynamic_reward_exponents([(preferred, rejected)], max_exponent=4)
    assert exponents.semantic == 4
    assert exponents.style == 1
    assert 0 <= preferred.aggregate(exponents) <= 1


def test_hope_fear_selection_is_distinct_and_deterministic_on_ties() -> None:
    candidates = [_reward("c"), _reward("a"), _reward("b")]
    selection = select_hope_and_fear(candidates)
    assert selection.hope.candidate_id == "a"
    assert selection.fear.candidate_id == "b"


def test_cpo_pair_uses_standard_string_columns() -> None:
    selection = select_hope_and_fear(
        [
            _reward("hope", text="yo", style=0.9),
            _reward("fear", text="Hello.", style=0.1),
        ]
    )
    record = make_cpo_pair_from_selection(
        pair_id="p1",
        neutral_bubbles=["Hello."],
        selection=selection,
        exponents=RewardExponents(style=2),
    )
    assert record["format"] == CPO_PAIR_FORMAT
    assert all(isinstance(record[key], str) for key in ("prompt", "chosen", "rejected"))
    assert json.loads(record["chosen"]) == {"original_bubbles": ["yo"]}
    with pytest.raises(ValueError, match="must differ"):
        make_cpo_pair(
            pair_id="bad",
            neutral_bubbles="hello",
            chosen_bubbles="same",
            rejected_bubbles="same",
        )


def test_reference_free_cpo_loss_prefers_a_larger_chosen_margin() -> None:
    weak_margin = cpo_pair_loss(-4.0, -4.1, chosen_nll=1.0)
    strong_margin = cpo_pair_loss(-2.0, -5.0, chosen_nll=1.0)

    assert strong_margin < weak_margin
    assert cpo_pair_loss(-2.0, -5.0, chosen_nll=0.5) < strong_margin


def test_evaluation_reports_rewards_content_and_structural_anchoring() -> None:
    rows = [
        {
            "pair_id": "one",
            "neutral": ["Tell Dad it causes motion sickness."],
            "original": ["it causes motion sickness", "tell dad"],
            "generated": ["Tell dad it causes motion sickness."],
            "style_probability": 0.7,
            "semantic_similarity": 0.9,
            "base_model_likelihood": 0.8,
            "length_ratio": 1.0,
        },
        {
            "pair_id": "two",
            "neutral": ["Where are you at 7?"],
            "original": ["wya at 7"],
            "generated": ["wya at 7"],
            "style_probability": 0.95,
            "semantic_similarity": 0.95,
            "base_model_likelihood": 0.7,
            "length_ratio": 0.6,
        },
    ]
    report = evaluate_candidate_rows(rows)
    assert report["rows"] == 2
    assert report["reward_rows"] == 2
    assert report["number_agreement_rate"] == 1.0
    assert report["structural_anchoring"]["opener_disagreements"] == 2
    assert report["structural_anchoring"]["split_disagreements"] == 1
    assert structural_correlation(
        "I will be there at seven",
        "I will be there at 7",
    ) > structural_correlation("What do you mean", "wdym")


def test_model_evaluation_requires_identical_pair_ids(tmp_path: Path) -> None:
    row = {
        "pair_id": "one",
        "neutral": "hello",
        "original": "yo",
        "generated": "yo",
    }
    output = tmp_path / "evaluation.json"
    report = evaluate_model_candidates(
        {"base": [row], "sft": [row], "cpo": [row]},
        output_path=output,
    )
    assert report["models"]["cpo"]["target_f1"] == 1.0
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    with pytest.raises(ValueError, match="identical pair IDs"):
        evaluate_model_candidates(
            {
                "base": [row],
                "cpo": [{**row, "pair_id": "different"}],
            }
        )
