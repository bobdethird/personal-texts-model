import pytest

from imessage_mlx.adapter_worker import (
    multiset_jaccard,
    normalized_words,
    transformation_sampling_weights,
)


def test_normalized_words_and_multiset_jaccard_ignore_case_and_punctuation() -> None:
    assert normalized_words("What’s good, BRO?") == ["what's", "good", "bro"]
    assert multiset_jaccard(["a", "a", "b"], ["a", "b", "b"]) == pytest.approx(0.5)


def test_transformation_sampling_weights_are_quality_gated_and_deterministic() -> None:
    rows = [
        {"source": "Where are you?", "target": "where are you"},
        {
            "source": "one two three four five six seven eight nine ten",
            "target": "one two three four five six seven eight nine",
        },
        {"source": "what do you mean right now", "target": "wdym rn fr"},
        {"source": "meet me at 7 outside", "target": "pull up at 8"},
    ]
    settings = {
        "enabled": True,
        "exact_copy_weight": 0.25,
        "near_copy_threshold": 0.9,
        "near_copy_weight": 0.5,
        "strong_threshold": 0.7,
        "strong_weight": 1.5,
        "strong_min_length_ratio": 0.5,
        "strong_max_length_ratio": 1.2,
        "require_source_numbers": True,
    }

    first = transformation_sampling_weights(rows, settings)
    second = transformation_sampling_weights(rows, settings)

    assert first == second
    weights, report = first
    assert weights == [0.25, 0.5, 1.5, 1.0]
    assert report["counts"] == {
        "exact_copy": 1,
        "near_copy": 1,
        "quality_gated_strong": 1,
        "standard": 1,
    }
    assert sum(report["expected_share"].values()) == pytest.approx(1.0, abs=1e-6)


def test_transformation_sampling_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="thresholds"):
        transformation_sampling_weights(
            [{"source": "hello", "target": "hi"}],
            {"enabled": True, "strong_threshold": 0.95, "near_copy_threshold": 0.9},
        )
