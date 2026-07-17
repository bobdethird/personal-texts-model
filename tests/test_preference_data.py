import pytest

from imessage_mlx.preference_data import (
    build_preference_pairs,
    prediction_failure_reasons,
)


def test_prediction_failure_reasons_catch_known_modes() -> None:
    assert prediction_failure_reasons(
        "I will be there at seven.", "I will be there at seven"
    ) == ["copy"]
    assert prediction_failure_reasons(
        "I have $3,000 to spend.", "I got money to spend"
    ) == ["numeral_loss"]
    assert prediction_failure_reasons(
        "No, I am not coming tonight.", "I am coming tonight"
    ) == ["negation_flip"]
    assert "length_collapse" in prediction_failure_reasons(
        "I will be there at seven with the whole crew.", "7"
    )
    assert prediction_failure_reasons("What do you mean?", "wdym") == []


def test_build_preference_pairs_keeps_only_failures_up_to_cap() -> None:
    train = [
        {"pair_id": "a", "source": "Where are you?", "target": "wya"},
        {"pair_id": "b", "source": "What do you mean?", "target": "wdym"},
        {"pair_id": "c", "source": "No, it is fine.", "target": "nah it's fine"},
    ]
    predictions = [
        {"pair_id": "a", "generated_text": "Where are you"},
        {"pair_id": "b", "generated_text": "wdym"},
        {"pair_id": "c", "generated_text": "it is fine"},
    ]

    pairs, report = build_preference_pairs(train, predictions, cap=10)

    assert [pair["pair_id"] for pair in pairs] == ["a", "c"]
    assert pairs[0]["chosen"] == "wya"
    assert pairs[0]["rejected"] == "Where are you"
    assert report["failure_reasons"] == {"copy": 1, "negation_flip": 1}

    capped, _ = build_preference_pairs(train, predictions, cap=1)
    assert len(capped) == 1
    with pytest.raises(ValueError, match="positive"):
        build_preference_pairs(train, predictions, cap=0)
