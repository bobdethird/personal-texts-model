from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from imessage_mlx.data import convergence_generation
from imessage_mlx.data.convergence_generation import (
    GeneratedVariant,
    GeneratedVariantBatch,
    ResolvedEntity,
    ResumeCompatibilityError,
    SemanticExtraction,
    SemanticFactConflictError,
    SlangInterpretation,
    StructuredOutputMissingError,
    StyleIdentifierMismatchError,
    TargetIdentifierMismatchError,
    deterministic_pair_id,
    generate_openai_convergence_data,
    select_convergence_targets,
    semantic_repeats_original_wording,
    validate_generated_source,
    validate_generated_variant_batch,
    validate_semantic_extraction,
)
from imessage_mlx.utils import read_jsonl, write_jsonl

STYLED_TARGET = "ZXQUNIQUEORIGINALPHRASE asks us to discuss the plan"


def _bart_record(index: int, *, target: str | None = None) -> dict[str, str]:
    return {
        "pair_id": f"pair-{index}",
        "source": f"Please discuss topic {index}" + ("?" if index % 2 else "."),
        "target": target or f"Can we discuss topic {index}" + ("?" if index % 2 else "."),
    }


def _write_splits(
    root: Path,
    *,
    train: list[dict[str, str]],
    valid: list[dict[str, str]] | None = None,
    test: list[dict[str, str]] | None = None,
) -> None:
    write_jsonl(root / "train.jsonl", train)
    write_jsonl(root / "valid.jsonl", valid or [])
    write_jsonl(root / "test.jsonl", test or [])


def _semantic(
    target_id: str,
    *,
    eligible: bool = True,
    context_dependent: bool = False,
) -> SemanticExtraction:
    return SemanticExtraction(
        target_id=target_id,
        speech_act="request",
        resolved_paraphrase="I would like us to talk through the plan together.",
        atomic_propositions=["The speaker wants to discuss a plan."],
        entities=[],
        protected_literals=[],
        time_references=[],
        numbers=[],
        negation=False,
        modality_uncertainty=[],
        question_intent=None,
        emotion="neutral",
        intensity="normal",
        eligible=eligible,
        context_dependent=context_dependent,
        ineligibility_reason=None if eligible else "Unknown conversational referent.",
    )


def _variant_batch(
    target_id: str,
    *,
    bad_terse_text: str | None = None,
) -> GeneratedVariantBatch:
    values = {
        "formal_professional": "Would you be available for a discussion when convenient?",
        "neutral_everyday": "Can we talk when you have a minute?",
        "verbose_indirect": (
            "When you find some time, I was hoping we could sit down and discuss this."
        ),
        "terse_conversational": bad_terse_text or "Got a sec to chat?",
    }
    return GeneratedVariantBatch(
        target_id=target_id,
        variants=[
            GeneratedVariant(
                target_id=target_id,
                variant_kind=style,
                source_text=source,
            )
            for style, source in values.items()
        ],
    )


def _response(parsed, *, input_tokens: int, output_tokens: int):
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(parsed=parsed)],
            )
        ],
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )


class FakeResponses:
    def __init__(
        self,
        *,
        original_target: str,
        bad_first_variant_call: bool = False,
        ineligible_semantic_calls: set[int] | None = None,
    ) -> None:
        self.original_target = original_target
        self.bad_first_variant_call = bad_first_variant_call
        self.ineligible_semantic_calls = ineligible_semantic_calls or set()
        self.calls: list[dict[str, object]] = []
        self.stage_counts: Counter[str] = Counter()

    async def parse(self, **options):
        self.calls.append(options)
        payload = json.loads(str(options["input"]))
        text_format = options["text_format"]
        if text_format is SemanticExtraction:
            self.stage_counts["stage_a"] += 1
            call_number = self.stage_counts["stage_a"]
            eligible = call_number not in self.ineligible_semantic_calls
            return _response(
                _semantic(
                    payload["target_id"],
                    eligible=eligible,
                    context_dependent=not eligible,
                ),
                input_tokens=13,
                output_tokens=7,
            )
        if text_format is GeneratedVariantBatch:
            self.stage_counts["stage_b"] += 1
            assert self.original_target not in str(options["input"])
            bad_text = (
                self.original_target
                if self.bad_first_variant_call and self.stage_counts["stage_b"] == 1
                else None
            )
            return _response(
                _variant_batch(payload["target_id"], bad_terse_text=bad_text),
                input_tokens=31,
                output_tokens=9,
            )
        raise AssertionError(f"Unexpected structured output type: {text_format}")


def _patch_client(monkeypatch, responses: FakeResponses) -> None:
    class FakeClient:
        def __init__(self, **_options):
            self.responses = responses

        async def close(self):
            return None

    monkeypatch.setattr(convergence_generation, "AsyncOpenAI", FakeClient)


def _run(
    splits: Path,
    output: Path,
    report: Path,
    *,
    model: str = "test-model",
    max_variant_attempts: int = 1,
):
    return asyncio.run(
        generate_openai_convergence_data(
            splits,
            output,
            report,
            api_key="test-key",
            model=model,
            allocation={"train": 1, "valid": 0, "test": 0},
            max_variant_attempts=max_variant_attempts,
        )
    )


def test_two_stage_generation_is_blind_private_structured_and_separately_metered(
    tmp_path: Path,
    monkeypatch,
) -> None:
    splits = tmp_path / "splits"
    output = tmp_path / "output"
    report_path = tmp_path / "report.json"
    _write_splits(splits, train=[_bart_record(1, target=STYLED_TARGET)])
    responses = FakeResponses(original_target=STYLED_TARGET)
    _patch_client(monkeypatch, responses)

    report = _run(splits, output, report_path)

    assert len(responses.calls) == 2
    assert all(call["store"] is False for call in responses.calls)
    stage_a_call = next(
        call for call in responses.calls if call["text_format"] is SemanticExtraction
    )
    stage_b_call = next(
        call for call in responses.calls if call["text_format"] is GeneratedVariantBatch
    )
    assert STYLED_TARGET in str(stage_a_call["input"])
    assert STYLED_TARGET not in str(stage_b_call["input"])
    assert set(json.loads(str(stage_a_call["input"]))) == {
        "context_bundle",
        "styled_target",
        "target_id",
    }
    assert set(json.loads(str(stage_b_call["input"]))) == {"target_id", "semantic"}

    published = list(read_jsonl(output / "published.jsonl"))
    assert [record["variant_kind"] for record in published] == list(
        convergence_generation.STYLE_KINDS
    )
    assert len({record["target_id"] for record in published}) == 1
    assert all(record["generation_fingerprint"] for record in published)
    assert all(
        record["pair_id"] == deterministic_pair_id(record["target_id"], record["variant_kind"])
        for record in published
    )
    assert report["usage"]["stage_a"] == {
        "input_tokens": 13,
        "output_tokens": 7,
        "total_tokens": 20,
    }
    assert report["usage"]["stage_b"] == {
        "input_tokens": 31,
        "output_tokens": 9,
        "total_tokens": 40,
    }
    assert report["variants"]["complete_groups"] == 1
    assert report["variants"]["published_rows"] == 4
    assert report["message_text_persisted_in_report"] is False
    assert STYLED_TARGET not in report_path.read_text(encoding="utf-8")
    assert (output / "manifest.json").stat().st_mode & 0o777 == 0o600
    assert (output / "semantics.jsonl").stat().st_mode & 0o777 == 0o600


def test_exact_target_and_style_identifiers_are_required() -> None:
    batch = _variant_batch("expected")
    assert validate_generated_variant_batch(batch, expected_target_id="expected") is batch

    with pytest.raises(TargetIdentifierMismatchError):
        validate_generated_variant_batch(batch, expected_target_id="other")

    missing_style = batch.model_copy(update={"variants": batch.variants[:-1]})
    with pytest.raises(StyleIdentifierMismatchError):
        validate_generated_variant_batch(missing_style, expected_target_id="expected")

    wrong_item_id = batch.model_copy(
        update={
            "variants": [
                batch.variants[0].model_copy(update={"target_id": "other"}),
                *batch.variants[1:],
            ]
        }
    )
    with pytest.raises(TargetIdentifierMismatchError):
        validate_generated_variant_batch(wrong_item_id, expected_target_id="expected")


def test_resolved_entities_must_cite_context_bundle_evidence() -> None:
    semantic = _semantic("target").model_copy(
        update={
            "resolved_entities": [
                ResolvedEntity(
                    term="Cercor",
                    interpretation="The author's AI-agent project.",
                    evidence_ids=["message-evidence"],
                    confidence="high",
                )
            ]
        }
    )
    context_bundle = {
        "retrieved_evidence": [{"message_id": "message-evidence"}],
        "exact_links": [],
        "recent_turns": [],
        "glossary_entries": [],
    }

    assert (
        validate_semantic_extraction(
            semantic,
            expected_target_id="target",
            context_bundle=context_bundle,
        )
        is semantic
    )
    with pytest.raises(SemanticFactConflictError):
        validate_semantic_extraction(
            semantic,
            expected_target_id="target",
            context_bundle={**context_bundle, "retrieved_evidence": []},
        )


def test_resolved_paraphrase_is_required_and_must_not_copy_target_wording() -> None:
    with pytest.raises(StructuredOutputMissingError, match="resolved paraphrase"):
        validate_semantic_extraction(
            _semantic("target").model_copy(update={"resolved_paraphrase": "  "}),
            expected_target_id="target",
        )

    target = "we could get dinner after the movie"
    copied = _semantic("target").model_copy(
        update={"resolved_paraphrase": "We could get dinner, after the movie!"}
    )
    assert semantic_repeats_original_wording(copied, target)

    reworded = _semantic("target").model_copy(
        update={"resolved_paraphrase": "I suggest having dinner once the movie ends."}
    )
    assert not semantic_repeats_original_wording(reworded, target)


def test_slang_interpretations_are_scoped_grounded_and_target_anchored() -> None:
    target = "ong we should demo cercor tonight brodie"

    widespread = _semantic("target").model_copy(
        update={
            "slang_interpretations": [
                SlangInterpretation(
                    expression="ong",
                    interpretation="A sincerity emphasis similar to 'honestly'.",
                    scope="widespread",
                ),
                SlangInterpretation(
                    expression="brodie",
                    interpretation="A generic friendly address term like 'bro'.",
                    scope="widespread",
                ),
            ]
        }
    )
    assert (
        validate_semantic_extraction(
            widespread,
            expected_target_id="target",
            styled_target=target,
        )
        is widespread
    )

    ungrounded = _semantic("target").model_copy(
        update={
            "slang_interpretations": [
                SlangInterpretation(
                    expression="cercor",
                    interpretation="The author's AI-agent project.",
                    scope="in_group",
                )
            ]
        }
    )
    with pytest.raises(SemanticFactConflictError, match="grounding evidence"):
        validate_semantic_extraction(
            ungrounded,
            expected_target_id="target",
            styled_target=target,
        )

    grounded = _semantic("target").model_copy(
        update={
            "slang_interpretations": [
                SlangInterpretation(
                    expression="cercor",
                    interpretation="The author's AI-agent project that invokes tools.",
                    scope="in_group",
                    evidence_ids=["glossary-cercor"],
                )
            ]
        }
    )
    bundle = {
        "exact_links": [],
        "recent_turns": [],
        "retrieved_evidence": [],
        "glossary_entries": [{"entry_id": "glossary-cercor", "evidence_message_ids": []}],
    }
    assert (
        validate_semantic_extraction(
            grounded,
            expected_target_id="target",
            styled_target=target,
            context_bundle=bundle,
        )
        is grounded
    )

    unprotected = _semantic("target").model_copy(
        update={
            "slang_interpretations": [
                SlangInterpretation(
                    expression="ong",
                    interpretation="Unclear whether this is emphasis or a nickname.",
                    scope="uncertain",
                )
            ]
        }
    )
    with pytest.raises(SemanticFactConflictError, match="protected literal"):
        validate_semantic_extraction(
            unprotected,
            expected_target_id="target",
            styled_target=target,
        )

    absent = _semantic("target").model_copy(
        update={
            "slang_interpretations": [
                SlangInterpretation(
                    expression="fr fr",
                    interpretation="A sincerity emphasis.",
                    scope="widespread",
                )
            ]
        }
    )
    with pytest.raises(SemanticFactConflictError, match="absent from the target"):
        validate_semantic_extraction(
            absent,
            expected_target_id="target",
            styled_target=target,
        )


def test_partial_variant_resume_retries_only_missing_style_and_publishes_complete_group(
    tmp_path: Path,
    monkeypatch,
) -> None:
    splits = tmp_path / "splits"
    output = tmp_path / "output"
    _write_splits(splits, train=[_bart_record(1, target=STYLED_TARGET)])

    first_responses = FakeResponses(
        original_target=STYLED_TARGET,
        bad_first_variant_call=True,
    )
    _patch_client(monkeypatch, first_responses)
    first = _run(splits, output, tmp_path / "first-report.json")

    assert first["variants"]["accepted_total"] == 3
    assert first["variants"]["complete_groups"] == 0
    assert list(read_jsonl(output / "published.jsonl")) == []

    second_responses = FakeResponses(original_target=STYLED_TARGET)
    _patch_client(monkeypatch, second_responses)
    second = _run(splits, output, tmp_path / "second-report.json")

    assert second_responses.stage_counts["stage_a"] == 0
    assert second_responses.stage_counts["stage_b"] == 1
    assert second["usage"]["stage_a"]["total_tokens"] == 0
    assert second["variants"]["accepted_this_run"] == 1
    assert second["variants"]["accepted_total"] == 4
    assert second["variants"]["complete_groups"] == 1
    assert len(list(read_jsonl(output / "semantics.jsonl"))) == 1
    assert len(list(read_jsonl(output / "variants.jsonl"))) == 5
    assert len(list(read_jsonl(output / "published.jsonl"))) == 4


def test_resume_rejects_incompatible_generation_fingerprint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    splits = tmp_path / "splits"
    output = tmp_path / "output"
    _write_splits(splits, train=[_bart_record(1, target=STYLED_TARGET)])
    responses = FakeResponses(original_target=STYLED_TARGET)
    _patch_client(monkeypatch, responses)
    _run(splits, output, tmp_path / "first-report.json")

    with pytest.raises(ResumeCompatibilityError, match="fingerprint"):
        _run(
            splits,
            output,
            tmp_path / "second-report.json",
            model="different-model",
        )
    assert len(responses.calls) == 2


def test_target_selection_is_deterministic_and_split_allocated(tmp_path: Path) -> None:
    splits = tmp_path / "splits"
    _write_splits(
        splits,
        train=[_bart_record(index) for index in range(20)],
        valid=[_bart_record(100 + index) for index in range(8)],
        test=[_bart_record(200 + index) for index in range(8)],
    )
    allocation = {"train": 10, "valid": 3, "test": 3}

    first = select_convergence_targets(splits, allocation=allocation)
    second = select_convergence_targets(splits, allocation=allocation)

    assert first == second
    assert Counter(record["split"] for record in first) == allocation
    assert len({record["target_id"] for record in first}) == 16
    assert all(record["target_id"] != record["source_pair_id"] for record in first)


def test_target_selection_excludes_low_content_reactions(tmp_path: Path) -> None:
    splits = tmp_path / "splits"
    _write_splits(
        splits,
        train=[
            _bart_record(1, target="😂💀"),
            _bart_record(2, target="I am ordering dinner now"),
        ],
    )

    selected = select_convergence_targets(
        splits,
        allocation={"train": 1, "valid": 0, "test": 0},
    )

    assert len(selected) == 1
    assert selected[0]["source_pair_id"] == "pair-2"


def test_target_selection_accepts_context_bearing_unpaired_style_targets(tmp_path: Path) -> None:
    splits = tmp_path / "splits"
    target = {
        "pair_id": "target-1",
        "timestamp_ns": 1,
        "styled_text": "yeah i can come after dinner",
        "context": [{"role": "other", "text": "Can you come later?"}],
    }
    _write_splits(splits, train=[target], valid=[], test=[])

    selected = select_convergence_targets(
        splits,
        allocation={"train": 1, "valid": 0, "test": 0},
    )

    assert selected[0]["target_text"] == target["styled_text"]
    assert selected[0]["context"] == target["context"]
    assert selected[0]["selection_stratum"].startswith("unpaired|")


def test_context_ineligible_primary_is_backfilled_within_split(
    tmp_path: Path,
    monkeypatch,
) -> None:
    splits = tmp_path / "splits"
    output = tmp_path / "output"
    _write_splits(
        splits,
        train=[
            _bart_record(1, target="Can we talk about lunch?"),
            _bart_record(2, target="Can we talk about dinner?"),
            _bart_record(3, target="Can we talk about tomorrow?"),
        ],
    )
    responses = FakeResponses(
        original_target="not present in stage b",
        ineligible_semantic_calls={1},
    )
    _patch_client(monkeypatch, responses)

    report = _run(splits, output, tmp_path / "report.json")

    assert responses.stage_counts["stage_a"] == 2
    assert responses.stage_counts["stage_b"] == 1
    assert report["selection"]["context_ineligible"] == 1
    assert report["selection"]["backfilled"]["train"] == 1
    assert report["variants"]["complete_groups"] == 1


def test_generated_sources_must_preserve_opaque_literals() -> None:
    semantic = _semantic("target").model_copy(update={"protected_literals": ["grad"]})

    reasons = validate_generated_source(
        variant_kind="formal_professional",
        target_text="Are you speaking at grad",
        source_text="Are you speaking at the graduate level?",
        sibling_sources=[],
        max_characters=512,
        semantic=semantic,
    )

    assert "protected_literal_conflict" in reasons


def test_neutral_everyday_source_cannot_copy_the_styled_target() -> None:
    semantic = _semantic("target")

    reasons = validate_generated_source(
        variant_kind="neutral_everyday",
        target_text="yeah i can come after dinner",
        source_text="Yeah, I can come after dinner.",
        sibling_sources=[],
        max_characters=512,
        semantic=semantic,
    )

    assert "target_normalized_duplicate" in reasons


def test_context_resolved_facts_validate_against_semantics_not_terse_target() -> None:
    semantic = _semantic("target").model_copy(
        update={
            "numbers": ["7"],
            "atomic_propositions": ["The proposed time of 7 is acceptable."],
            "context_dependent": True,
        }
    )

    reasons = validate_generated_source(
        variant_kind="formal_professional",
        target_text="yeah that works",
        source_text="The proposed time of 7 works for me.",
        sibling_sources=[],
        max_characters=512,
        semantic=semantic,
    )

    assert "protected_fact_conflict" not in reasons


def test_semantic_extraction_must_keep_explicit_target_literals() -> None:
    semantic = _semantic("target")

    with pytest.raises(SemanticFactConflictError, match="target number"):
        validate_semantic_extraction(
            semantic,
            expected_target_id="target",
            styled_target="I can arrive at 7",
        )
