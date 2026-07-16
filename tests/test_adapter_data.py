import json
from pathlib import Path

import pytest

from imessage_mlx.adapter_worker import _semantic_reference
from imessage_mlx.data.adapters import (
    classify_pair_signal,
    prepare_adapter_datasets,
    prepare_convergence_adapter_datasets,
    protected_facts,
)
from imessage_mlx.utils import read_jsonl, write_json, write_jsonl


def _pair(pair_id: str, timestamp: int, neutral: str, styled: str) -> dict[str, object]:
    return {
        "pair_id": pair_id,
        "timestamp_ns": timestamp,
        "neutral_text": neutral,
        "styled_text": styled,
    }


def test_pair_signal_strata_and_protected_facts() -> None:
    assert classify_pair_signal("See you later", "See you later") == "exact"
    assert classify_pair_signal("See you later.", "see you later") == "surface_only"
    assert classify_pair_signal("I will see you tomorrow", "see u tmr") == "substantive"
    assert protected_facts("I can't arrive before 7:30 at <|url|>?") == {
        "numbers": ("7:30",),
        "placeholders": ("<|url|>",),
        "negated": True,
    }
    assert protected_facts("Don’t mess this up")["negated"] is True


def test_context_resolved_semantic_reference_is_used_when_available() -> None:
    row = {
        "target": "yeah that works",
        "semantic": {
            "speech_act": "agreement",
            "atomic_propositions": ["The proposed dinner time is acceptable."],
            "entities": [],
            "protected_literals": [],
            "time_references": ["the proposed dinner time"],
            "numbers": [],
            "modality_uncertainty": [],
            "question_intent": None,
            "emotion": "positive",
            "intensity": "normal",
        },
    }

    reference = _semantic_reference(row)

    assert "agreement" in reference
    assert "dinner time is acceptable" in reference
    assert _semantic_reference({"target": "fallback"}) == "fallback"

    row["semantic"]["resolved_paraphrase"] = "The proposed dinner time works for me."
    assert _semantic_reference(row) == "The proposed dinner time works for me."


def test_prepare_adapter_datasets_removes_cross_split_normalized_leakage(
    tmp_path: Path,
) -> None:
    splits = tmp_path / "splits"
    train = [
        _pair("train-1", 1, "Are you free later?", "u free later?"),
        _pair("train-2", 2, "That sounds good.", "sounds good"),
        _pair("train-3", 3, "No change", "No change"),
    ]
    validation = [
        _pair("valid-duplicate", 4, "Are you free later", "U FREE LATER"),
        _pair("valid-1", 5, "I will arrive soon.", "ill be there soon"),
        _pair(
            "valid-fuzzy",
            6,
            "That sounds goood.",
            "sounds goood",
        ),
        _pair("valid-fact-conflict", 7, "Meet me at 7.", "meet me at 8"),
    ]
    test = [
        _pair("test-duplicate", 8, "That sounds good", "sounds good!"),
        _pair("test-1", 9, "Please call me at 8.", "call me at 8"),
    ]
    write_jsonl(splits / "train.jsonl", train)
    write_jsonl(splits / "validation.jsonl", validation)
    write_jsonl(splits / "test.jsonl", test)

    report = prepare_adapter_datasets(
        splits,
        tmp_path / "adapter-data",
        tmp_path / "report.json",
        benchmark_train_size=2,
        benchmark_eval_size=1,
    )

    assert report["splits"]["train"]["output_pairs"] == 3
    assert report["splits"]["validation"]["removed_cross_split_duplicates"] == 2
    assert report["splits"]["validation"]["removed_fuzzy_duplicates"] == 1
    assert report["splits"]["validation"]["removed_fact_conflicts"] == 1
    assert report["splits"]["test"]["removed_cross_split_duplicates"] == 1
    assert report["train_strata"] == {
        "exact": 1,
        "high_overlap": 0,
        "substantive": 2,
        "surface_only": 0,
    }

    bart_valid = list(read_jsonl(tmp_path / "adapter-data/bart/valid.jsonl"))
    mlx_valid = list(read_jsonl(tmp_path / "adapter-data/mlx/valid.jsonl"))
    assert bart_valid == [
        {
            "pair_id": "valid-1",
            "source": "I will arrive soon.",
            "target": "ill be there soon",
        }
    ]
    assert mlx_valid[0]["prompt"].endswith("I will arrive soon.")
    assert mlx_valid[0]["completion"] == "ill be there soon"
    assert mlx_valid[0]["completion"] not in mlx_valid[0]["prompt"]
    assert len(list(read_jsonl(tmp_path / "adapter-data/benchmark/bart/train.jsonl"))) == 2
    assert len(list(read_jsonl(tmp_path / "adapter-data/benchmark/mlx/valid.jsonl"))) == 1
    assert json.loads((tmp_path / "report.json").read_text())["schema_version"] == 1


def test_prepare_convergence_data_uses_only_blind_generated_groups_by_default(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    for split in ("train", "valid", "test"):
        write_jsonl(
            base / f"{split}.jsonl",
            [
                {
                    "pair_id": f"{split}-base",
                    "source": f"Ordinary {split} source",
                    "target": f"my {split} target",
                }
            ],
        )

    styles = (
        "formal_professional",
        "neutral_everyday",
        "verbose_indirect",
        "terse_conversational",
    )
    generated = []
    for split in ("train", "valid", "test"):
        for index, style in enumerate(styles):
            generated.append(
                {
                    "pair_id": f"{split}-{index}",
                    "target_id": f"{split}-target",
                    "source_pair_id": f"{split}-base",
                    "split": split,
                    "variant_kind": style,
                    "source": f"{style} wording for the {split} message",
                    "target": f"my {split} target",
                    "semantic_similarity": 0.95,
                }
            )
    generated_path = tmp_path / "generated.jsonl"
    write_jsonl(generated_path, generated)

    report = prepare_convergence_adapter_datasets(
        base,
        generated_path,
        tmp_path / "prepared",
        tmp_path / "convergence-report.json",
    )

    challenge_valid = list(read_jsonl(tmp_path / "prepared/bart/challenge-valid.jsonl"))
    assert len(challenge_valid) == 4
    assert {row["target_id"] for row in challenge_valid} == {"valid-target"}
    assert {row["target"] for row in challenge_valid} == {"my valid target"}
    assert report["accepted_target_groups"] == {"train": 1, "valid": 1, "test": 1}
    train = list(read_jsonl(tmp_path / "prepared/bart/train.jsonl"))
    assert len(train) == 4
    assert {row["target_id"] for row in train} == {"train-target"}
    assert {row["variant_kind"] for row in train} == set(styles)
    assert report["legacy_base_included"] is False
    assert report["training_mode"] == "blind_generated_only"
    assert report["base_rows"] == {"train": 0, "valid": 0, "test": 0}


def test_context_grounded_adapter_data_requires_completed_review(tmp_path: Path) -> None:
    styles = (
        "formal_professional",
        "neutral_everyday",
        "verbose_indirect",
        "terse_conversational",
    )
    generated = [
        {
            "pair_id": f"{split}-{index}",
            "target_id": f"{split}-target",
            "source_pair_id": f"{split}-source",
            "split": split,
            "variant_kind": style,
            "source": f"{style} wording for {split}",
            "target": f"target wording for {split}",
            "semantic_similarity": 0.95,
            "generation_fingerprint": "fingerprint",
            "grounding_stratum": "historical",
            "context_bundle": {"retrieved_evidence": [{"message_id": "evidence"}]},
        }
        for split in ("train", "valid", "test")
        for index, style in enumerate(styles)
    ]
    pairs = tmp_path / "grounded.jsonl"
    write_jsonl(pairs, generated)

    with pytest.raises(ValueError, match="human review"):
        prepare_convergence_adapter_datasets(
            None,
            pairs,
            tmp_path / "rejected",
            tmp_path / "rejected-report.json",
        )

    summary = tmp_path / "review-summary.json"
    write_json(
        summary,
        {
            "human_approved": True,
            "reviewed_target_groups": 3,
            "semantic_or_fact_failures": 0,
            "generation_fingerprints": ["fingerprint"],
        },
    )
    report = prepare_convergence_adapter_datasets(
        None,
        pairs,
        tmp_path / "accepted",
        tmp_path / "accepted-report.json",
        review_summary_path=summary,
    )
    assert report["generation_fingerprint"] == "fingerprint"
    assert report["human_review_required"] is True


def test_prepare_convergence_data_rejects_cross_group_source_leakage(tmp_path: Path) -> None:
    base = tmp_path / "base"
    write_jsonl(
        base / "train.jsonl",
        [{"pair_id": "train-base", "source": "Shared training phrase", "target": "train target"}],
    )
    write_jsonl(
        base / "valid.jsonl",
        [
            {"pair_id": "valid-bad", "source": "valid source one", "target": "valid target one"},
            {"pair_id": "valid-good", "source": "valid source two", "target": "valid target two"},
        ],
    )
    write_jsonl(
        base / "test.jsonl",
        [{"pair_id": "test-base", "source": "test source", "target": "test target"}],
    )
    styles = (
        "formal_professional",
        "neutral_everyday",
        "verbose_indirect",
        "terse_conversational",
    )
    rows = []
    specs = (
        ("train", "train-group", "train-base", "train target"),
        ("valid", "valid-bad-group", "valid-bad", "valid target one"),
        ("valid", "valid-good-group", "valid-good", "valid target two"),
        ("test", "test-group", "test-base", "test target"),
    )
    for split, target_id, source_pair_id, target in specs:
        for index, style in enumerate(styles):
            source = f"{style} unique wording for {target_id}"
            if target_id == "valid-bad-group" and index == 0:
                source = "Shared training phrase"
            rows.append(
                {
                    "pair_id": f"{target_id}-{index}",
                    "target_id": target_id,
                    "source_pair_id": source_pair_id,
                    "split": split,
                    "variant_kind": style,
                    "source": source,
                    "target": target,
                    "semantic_similarity": 0.95,
                }
            )
    path = tmp_path / "generated.jsonl"
    write_jsonl(path, rows)

    report = prepare_convergence_adapter_datasets(
        base,
        path,
        tmp_path / "prepared",
        tmp_path / "report.json",
        include_legacy_base=True,
    )

    assert report["removed_target_groups"]["cross_split_normalized"] == 1
    assert report["accepted_target_groups"]["valid"] == 1
