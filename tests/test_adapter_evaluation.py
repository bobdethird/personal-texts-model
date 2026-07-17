import json
from pathlib import Path

import pytest

from imessage_mlx.adapter_evaluation import (
    compare_prediction_files,
    evaluate_prediction_rows,
    normalized_numbers,
)


def _prediction(
    pair_id: str,
    *,
    source: str,
    target: str,
    generated: str,
) -> dict[str, str]:
    return {
        "pair_id": pair_id,
        "neutral_text": source,
        "target_text": target,
        "generated_text": generated,
    }


def test_evaluate_prediction_rows_reports_copying_and_semantic_risk() -> None:
    report = evaluate_prediction_rows(
        [
            _prediction(
                "one",
                source="I will not arrive at 7",
                target="wont be there at 7",
                generated="I will arrive at 7",
            ),
            _prediction(
                "two",
                source="Hello there",
                target="yo",
                generated="Hello there",
            ),
        ]
    )

    assert report["rows"] == 2
    assert report["exact_source_copy_rate"] == 0.5
    assert report["source_numeral_retention_rate"] == 1.0
    assert report["source_negation_agreement_rate"] == 0.5
    assert report["by_reference_transformation_strength"]["strong"]["rows"] == 2
    assert report["target_jaccard"] > 0


def test_normalized_numbers_equates_common_texting_formats() -> None:
    assert normalized_numbers("$3,000 at 10:00 a.m.") == normalized_numbers("3k at 10am")


def test_compare_prediction_files_requires_matching_pair_ids(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    output = tmp_path / "report.json"
    baseline.write_text(
        json.dumps(
            _prediction("one", source="hello", target="yo", generated="hello")
        )
        + "\n"
    )
    candidate.write_text(
        json.dumps(_prediction("two", source="hello", target="yo", generated="yo")) + "\n"
    )

    with pytest.raises(ValueError, match="identical pair IDs"):
        compare_prediction_files(baseline, candidate, output)
