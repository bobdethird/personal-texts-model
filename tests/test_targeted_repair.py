import json
from pathlib import Path

from imessage_mlx.data.repair import repair_low_signal_pairs
from imessage_mlx.utils import read_jsonl, write_jsonl


def _pair(pair_id: str, neutral: str, styled: str) -> dict[str, object]:
    return {
        "pair_id": pair_id,
        "timestamp_ns": 1,
        "neutral_text": neutral,
        "styled_text": styled,
    }


def test_targeted_repair_blinds_reconstruction_and_keeps_failed_original(
    tmp_path: Path,
) -> None:
    pairs = tmp_path / "pairs.jsonl"
    write_jsonl(
        pairs,
        [
            _pair("accepted", "Meet at 7", "Meet at 7"),
            _pair("rejected", "Do not call before 8", "Do not call before 8"),
            _pair("substantive", "I will arrive soon.", "ill be there soon"),
        ],
    )
    stage_b_payloads: list[str] = []

    def generate(stage: str, payload: str) -> str:
        if stage == "extract":
            number = "7" if "7" in payload else "8"
            return json.dumps(
                {
                    "intent": "inform",
                    "propositions": ["meeting time"] if number == "7" else ["call restriction"],
                    "entities": [],
                    "times": [number],
                    "numbers": [number],
                    "placeholders": [],
                    "negated": number == "8",
                    "uncertainty": [],
                    "emotion": "neutral",
                    "intensity": "normal",
                }
            )
        stage_b_payloads.append(payload)
        assert "Meet at" not in payload
        assert "Do not call" not in payload
        return "meeting is at 7" if '"7"' in payload else "dont call before 9"

    report = repair_low_signal_pairs(
        pairs,
        tmp_path / "repaired.jsonl",
        tmp_path / "report.json",
        generate=generate,
        limit=2,
    )

    repaired = list(read_jsonl(tmp_path / "repaired.jsonl"))
    assert repaired[0]["neutral_text"] == "meeting is at 7"
    assert repaired[1]["neutral_text"] == "Do not call before 8"
    assert repaired[2]["neutral_text"] == "I will arrive soon."
    assert report["selected"] == 2
    assert report["accepted"] == 1
    assert report["rejected"] == 1
    assert len(stage_b_payloads) == 2
