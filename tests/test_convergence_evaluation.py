from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from imessage_mlx.convergence_evaluation import (
    STYLE_KINDS,
    compare_convergence_evaluations,
    create_convergence_comparison_review,
    create_convergence_pair_review,
    evaluate_convergence_predictions,
)
from imessage_mlx.utils import write_json, write_jsonl


def _sources() -> dict[str, str]:
    return {
        "formal_professional": "Would you please meet me at 7 near the station?",
        "neutral_everyday": "Can you meet me at 7 by the station?",
        "verbose_indirect": (
            "I was wondering whether you might be able to meet me near the station at 7."
        ),
        "terse_conversational": "meet at the station at 7?",
    }


def _rows(
    *,
    target_id: str = "target-1",
    target: str = "ill meet you by the station at 7",
    outputs: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    sources = _sources()
    generated = outputs or dict.fromkeys(STYLE_KINDS, target)
    return [
        {
            "pair_id": f"{target_id}-{style}",
            "target_id": target_id,
            "variant_kind": style,
            "source": sources[style],
            "target": target,
            "generated_text": generated[style],
        }
        for style in STYLE_KINDS
    ]


def _evaluate(
    tmp_path: Path,
    name: str,
    rows: list[dict[str, str]],
    *,
    train_targets: list[str] | None = None,
) -> dict:
    predictions = tmp_path / f"{name}-predictions.jsonl"
    train = tmp_path / f"{name}-train.jsonl"
    write_jsonl(predictions, rows)
    write_jsonl(
        train,
        [
            {"pair_id": f"train-{index}", "target": target}
            for index, target in enumerate(train_targets or ["unrelated training target"])
        ],
    )
    return evaluate_convergence_predictions(
        predictions,
        train,
        tmp_path / f"{name}-report.json",
    )


def test_convergence_evaluator_rejects_incomplete_and_changed_target_groups(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.jsonl"
    write_jsonl(train, [{"target": "unrelated"}])

    incomplete = tmp_path / "incomplete.jsonl"
    write_jsonl(incomplete, _rows()[:-1])
    with pytest.raises(ValueError, match="exactly four distinct variants"):
        evaluate_convergence_predictions(incomplete, train, tmp_path / "incomplete.json")

    changed = _rows()
    changed[1]["target"] = "ill meet you by the station at 8"
    changed_path = tmp_path / "changed.jsonl"
    write_jsonl(changed_path, changed)
    with pytest.raises(ValueError, match="inconsistent target text"):
        evaluate_convergence_predictions(changed_path, train, tmp_path / "changed.json")


def test_grouped_fact_preservation_fails_if_one_variant_changes_a_fact(
    tmp_path: Path,
) -> None:
    outputs = dict.fromkeys(STYLE_KINDS, "ill meet you by the station at 7")
    outputs["terse_conversational"] = "ill meet you by the station at 8"

    report = _evaluate(tmp_path, "bad-fact", _rows(outputs=outputs))

    assert report["content"]["protected_fact_preservation_rate"] == 0.0
    assert report["content"]["row_protected_fact_preservation_rate"] == 0.75
    assert report["content"]["protected_fact_failed_groups"] == 1
    assert report["content"]["protected_fact_failure_types"]["numbers"] == 1


def test_normalized_input_copy_is_reported_overall_and_per_style(tmp_path: Path) -> None:
    target = "ill meet you by the station at 7"
    outputs = dict.fromkeys(STYLE_KINDS, target)
    outputs["formal_professional"] = _sources()["formal_professional"]

    report = _evaluate(tmp_path, "copy", _rows(target=target, outputs=outputs))

    assert report["copying"]["normalized_input_copy_rate"] == 0.25
    assert report["copying"]["by_style"]["formal_professional"]["rate"] == 1.0
    assert report["copying"]["by_style"]["neutral_everyday"]["rate"] == 0.0
    assert report["per_style"]["formal_professional"]["normalized_input_copies"] == 1


def test_convergence_and_style_metrics_reward_register_invariant_outputs(
    tmp_path: Path,
) -> None:
    copied_report = _evaluate(tmp_path, "copied", _rows(outputs=_sources()))
    converged_report = _evaluate(tmp_path, "converged", _rows())

    copied_agreement = copied_report["convergence"]["mean_within_target_output_pairwise_agreement"]
    converged_agreement = converged_report["convergence"][
        "mean_within_target_output_pairwise_agreement"
    ]
    assert converged_agreement == 1.0
    assert converged_agreement > copied_agreement
    assert converged_report["style"]["output_marker_variance_across_styles"] == 0.0
    assert (
        converged_report["style"]["mean_proximity_to_target"]
        > copied_report["style"]["mean_proximity_to_target"]
    )
    assert set(converged_report["per_style"]) == set(STYLE_KINDS)
    assert converged_report["worst_register"]["variant_kind"] in STYLE_KINDS


def test_expansion_gate_passes_and_reports_each_explicit_gate(tmp_path: Path) -> None:
    baseline = _evaluate(tmp_path, "baseline", _rows(outputs=_sources()))
    augmented = _evaluate(tmp_path, "augmented", _rows())
    semantic = tmp_path / "semantic.json"
    write_json(
        semantic,
        {
            "baseline": {
                "mean_generated_source_similarity": 0.95,
                "generated_source_below_0_80": 1,
            },
            "augmented": {
                "mean_generated_source_similarity": 0.94,
                "generated_source_below_0_80": 1,
            },
        },
    )

    result = compare_convergence_evaluations(
        baseline,
        augmented,
        semantic_report=semantic,
    )

    assert result["expand_recommended"] is True
    assert result["reasons"] == ["all_expansion_gates_passed"]
    assert all(gate["passed"] for gate in result["gates"].values())
    assert result["weighted_score_used"] is False


def test_expansion_gate_fails_without_semantics_and_on_regressions(tmp_path: Path) -> None:
    better = _evaluate(tmp_path, "better", _rows())
    worse = _evaluate(tmp_path, "worse", _rows(outputs=_sources()))
    worse = copy.deepcopy(worse)
    worse["content"]["protected_fact_preservation_rate"] = 0.97

    result = compare_convergence_evaluations(better, worse)

    assert result["expand_recommended"] is False
    assert set(result["reasons"]) == {
        "protected_fact_preservation",
        "semantic_no_material_regression",
        "copying_lower_or_equal",
        "measurable_style_or_convergence_improvement",
    }


def test_training_target_memorization_is_grouped_and_counted(tmp_path: Path) -> None:
    target = "this deliberately long private training target is memorized exactly"
    report = _evaluate(
        tmp_path,
        "memorized",
        _rows(target=target, outputs=dict.fromkeys(STYLE_KINDS, target)),
        train_targets=[target],
    )

    assert report["memorization"]["exact_training_target_matches"] == 4
    assert report["memorization"]["exact_long_training_target_matches"] == 4
    assert report["memorization"]["target_groups_with_exact_training_target_match"] == 1
    assert report["memorization"]["training_text_persisted_in_report"] is False


def test_private_grouped_review_contains_four_sources_and_both_model_outputs(
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "baseline.jsonl"
    augmented_path = tmp_path / "augmented.jsonl"
    baseline_rows = _rows(outputs=_sources())
    augmented_rows = _rows()
    write_jsonl(baseline_path, baseline_rows)
    write_jsonl(augmented_path, augmented_rows)
    review_path = tmp_path / "review.md"
    summary_path = tmp_path / "review-summary.json"

    summary = create_convergence_comparison_review(
        {"current": baseline_path, "pilot": augmented_path},
        review_path,
        summary_path=summary_path,
        sample_size=1,
    )

    review = review_path.read_text(encoding="utf-8")
    for source in _sources().values():
        assert source in review
    assert _sources()["formal_professional"] in review
    assert "ill meet you by the station at 7" in review
    assert "**current output**" in review
    assert "**pilot output**" in review
    assert summary["sampled_target_groups"] == 1
    assert summary["human_approved"] is False
    assert summary["reviewed_target_groups"] == 0
    assert summary["changed_fact_failures"] is None
    assert summary["message_text_persisted_in_summary"] is False
    serialized_summary = json.dumps(json.loads(summary_path.read_text(encoding="utf-8")))
    assert "Would you please meet me" not in serialized_summary
    assert "ill meet you by the station" not in serialized_summary
    assert review_path.stat().st_mode & 0o777 == 0o600


def test_private_pair_review_requires_complete_groups_and_writes_safe_summary(
    tmp_path: Path,
) -> None:
    pairs = tmp_path / "pairs.jsonl"
    rows = _rows()
    for row in rows:
        row["semantic_similarity"] = 0.95
        row["source"] = row.pop("neutral_text", row["source"])
        row["target"] = row.pop("target_text", row["target"])
        row["grounding_stratum"] = "historical"
        row["generation_fingerprint"] = "fingerprint"
        row["context_bundle"] = {
            "exact_links": [],
            "recent_turns": [],
            "retrieved_evidence": [
                {
                    "message_id": "evidence",
                    "role": "me",
                    "text": "The station is beside the library.",
                    "relation": "historical_retrieval",
                }
            ],
            "glossary_entries": [],
        }
        row["semantic"] = {
            "resolved_paraphrase": "I plan to meet you near the station at 7.",
            "atomic_propositions": ["The author will meet the recipient at 7."],
            "resolved_entities": [],
            "slang_interpretations": [
                {
                    "expression": "ill",
                    "interpretation": "A casual spelling of 'I will'.",
                    "scope": "widespread",
                    "evidence_ids": [],
                }
            ],
            "remaining_ambiguities": [],
        }
    write_jsonl(pairs, rows)
    review_path = tmp_path / "pair-review.md"

    summary = create_convergence_pair_review(
        pairs,
        review_path,
        sample_size=1,
    )

    review = review_path.read_text(encoding="utf-8")
    assert "**Authored target**" in review
    assert "**Retrieved historical evidence**" in review
    assert "**Resolved meaning**" in review
    assert "**Extracted semantic propositions**" in review
    assert "**Slang and texting expressions**" in review
    assert "`widespread`" in review
    assert "local score 0.950" in review
    assert summary["human_approved"] is False
    assert summary["semantic_or_fact_failures"] is None
    serialized = json.dumps(
        json.loads(review_path.with_suffix(".summary.json").read_text(encoding="utf-8"))
    )
    assert "ill meet you by the station" not in serialized

    write_jsonl(tmp_path / "incomplete.jsonl", rows[:-1])
    with pytest.raises(ValueError, match="complete four-source"):
        create_convergence_pair_review(
            tmp_path / "incomplete.jsonl",
            tmp_path / "incomplete.md",
        )
