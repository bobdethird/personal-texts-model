from pathlib import Path

from imessage_mlx.rewrite_evaluation import (
    compare_rewrite_evaluations,
    create_rewrite_comparison_review,
    evaluate_rewrite_predictions,
)
from imessage_mlx.utils import write_json, write_jsonl


def test_rewrite_evaluator_passes_target_outputs_and_reports_style_lift(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    write_jsonl(
        train,
        [
            {
                "pair_id": "train-1",
                "neutral_text": "Are you available later?",
                "styled_text": "u free later?",
            }
        ],
    )
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(
        predictions,
        [
            {
                "pair_id": "test-1",
                "neutral_text": "I will arrive at 7.",
                "target_text": "ill be there at 7",
                "generated_text": "ill be there at 7",
            },
            {
                "pair_id": "test-2",
                "neutral_text": "Could you call me?",
                "target_text": "can u call me?",
                "generated_text": "can u call me?",
            },
        ],
    )

    semantic = tmp_path / "semantic.json"
    write_json(
        semantic,
        {
            "mean_generated_source_similarity": 0.99,
            "mean_target_source_similarity": 0.95,
            "generated_source_below_0_80": 0,
        },
    )
    report = evaluate_rewrite_predictions(
        predictions,
        train,
        tmp_path / "report.json",
        semantic_report_path=semantic,
    )

    assert report["examples"] == 2
    assert report["content"]["protected_fact_preservation_rate"] == 1.0
    assert report["content"]["mean_target_similarity"] == 1.0
    assert report["style"]["gap_closed"] == 1.0
    assert report["fluency"]["empty_outputs"] == 0
    assert report["ready_to_promote"] is True


def test_rewrite_evaluator_blocks_fact_loss_empty_and_memorized_outputs(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    write_jsonl(
        train,
        [
            {
                "pair_id": "train-1",
                "neutral_text": "Unrelated",
                "styled_text": "memorized private phrase",
            }
        ],
    )
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(
        predictions,
        [
            {
                "pair_id": "test-1",
                "neutral_text": "I cannot arrive before 8.",
                "target_text": "cant make it before 8",
                "generated_text": "memorized private phrase",
            },
            {
                "pair_id": "test-2",
                "neutral_text": "Are you free?",
                "target_text": "u free?",
                "generated_text": "",
            },
        ],
    )

    report = evaluate_rewrite_predictions(predictions, train, tmp_path / "report.json")

    assert report["content"]["protected_fact_preservation_rate"] == 0.0
    assert report["fluency"]["empty_outputs"] == 1
    assert report["memorization"]["exact_training_target_matches"] == 1
    assert report["ready_to_promote"] is False


def test_comparison_review_aligns_private_predictions(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    common = {
        "pair_id": "pair-1",
        "neutral_text": "I will arrive soon.",
        "target_text": "ill be there soon",
    }
    write_jsonl(first, [{**common, "generated_text": "be there soon"}])
    write_jsonl(second, [{**common, "generated_text": "I will arrive soon"}])

    report = create_rewrite_comparison_review(
        {"adapter": first, "legacy": second},
        tmp_path / "review.md",
        sample_size=1,
    )

    review = (tmp_path / "review.md").read_text()
    assert report["sampled_examples"] == 1
    assert "be there soon" in review
    assert "Adapter preferred over legacy" in review


def test_architecture_selection_requires_local_semantic_evidence(tmp_path: Path) -> None:
    common = {
        "content": {
            "protected_fact_preservation_rate": 1.0,
            "mean_target_similarity": 0.9,
            "identity_baseline_similarity": 0.8,
        },
        "style": {"gap_closed": 0.6, "lexical_change_gap_closed": 0.8},
        "fluency": {
            "empty_outputs": 0,
            "repeated_word_outputs": 0,
            "structural_token_outputs": 0,
        },
    }
    bart = tmp_path / "bart.json"
    qwen = tmp_path / "qwen.json"
    write_json(
        bart,
        {
            **common,
            "semantic": {
                "mean_generated_source_similarity": 0.95,
                "mean_target_source_similarity": 0.94,
            },
        },
    )
    write_json(qwen, {**common, "semantic": None})

    report = compare_rewrite_evaluations(
        {"bart": bart, "qwen": qwen},
        tmp_path / "selection.json",
    )

    assert report["selected"] == "bart"
    assert report["candidates"]["qwen"]["qualifies_for_full_training"] is False
