import pytest

from imessage_mlx.adapter_worker import (
    decoding_options,
    minimum_new_tokens,
    multiset_jaccard,
    normalized_words,
    select_rewrite_candidate,
    transformation_sampling_weights,
)

GUARDRAILS = decoding_options(
    {
        "decoding": {
            "best_of": 4,
            "min_new_tokens_ratio": 0.35,
            "length_ratio_min": 0.3,
            "length_ratio_max": 1.6,
        }
    }
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


def test_decoding_defaults_keep_legacy_greedy_behavior() -> None:
    options = decoding_options({})
    assert options["best_of"] == 1
    assert minimum_new_tokens("I will be there at seven.", options) == 0
    assert minimum_new_tokens("I will be there at seven.", GUARDRAILS) == 2


def test_select_rewrite_candidate_blocks_length_collapse() -> None:
    picked = select_rewrite_candidate(
        "I will be there at seven.",
        ["7", "I will be there at seven"],
        GUARDRAILS,
    )
    assert picked == "I will be there at seven"


def test_select_rewrite_candidate_escapes_copy_without_flipping_negation() -> None:
    picked = select_rewrite_candidate(
        "No, I do not think they are happening yet.",
        [
            "No, I do not think they are happening yet",
            "they are happening",
            "nah idt they're happening yet",
        ],
        GUARDRAILS,
    )
    assert picked == "nah idt they're happening yet"


def test_select_rewrite_candidate_trusts_transformed_greedy() -> None:
    picked = select_rewrite_candidate(
        "What is up with you today?",
        ["whats up w you today", "wyd"],
        GUARDRAILS,
    )
    assert picked == "whats up w you today"


def test_select_rewrite_candidate_escapes_copies_minimally() -> None:
    picked = select_rewrite_candidate(
        "I pushed the changes to the repository.",
        [
            "I pushed the changes to the repository",
            "pushed the changes to the repo",
            "pushed",
        ],
        GUARDRAILS,
    )
    assert picked == "pushed the changes to the repo"


def test_select_rewrite_candidate_allows_initialisms_for_short_drafts() -> None:
    picked = select_rewrite_candidate(
        "What do you mean?",
        ["What do you mean", "wdym"],
        GUARDRAILS,
    )
    assert picked == "wdym"


def test_select_rewrite_candidate_requires_source_numerals() -> None:
    picked = select_rewrite_candidate(
        "I have $3,000 to spend before 10:00 a.m.",
        ["I have money to spend before the morning", "i got 3k to spend before 10am"],
        GUARDRAILS,
    )
    assert picked == "i got 3k to spend before 10am"


def test_transformation_sampling_downweights_severe_deletion() -> None:
    rows = [
        {
            "source": "one two three four five six seven eight nine ten",
            "target": "totally different words",
        },
    ]
    weights, report = transformation_sampling_weights(
        rows,
        {"enabled": True, "severe_deletion_weight": 0.35},
    )
    assert weights == [0.35]
    assert report["counts"] == {"severe_deletion": 1}


def test_transformation_sampling_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="thresholds"):
        transformation_sampling_weights(
            [{"source": "hello", "target": "hi"}],
            {"enabled": True, "strong_threshold": 0.95, "near_copy_threshold": 0.9},
        )
