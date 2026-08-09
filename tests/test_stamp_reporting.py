from pathlib import Path

from imessage_mlx.reporting import render_stamp_report, render_stamp_report_file


def _results() -> dict[str, object]:
    return {
        "run_name": "stamp-test",
        "methods": ["base", "sft", "cpo-1"],
        "classifier_metrics": {"accuracy": 0.9, "roc_auc": 0.95},
        "metrics": {
            "base": {"style_probability": 0.2, "semantic_similarity": 0.9},
            "sft": {"style_probability": 0.7, "semantic_similarity": 0.86},
            "cpo-1": {
                "style_probability": 0.82,
                "semantic_similarity": 0.88,
                "fluency": 0.75,
                "reward": 0.71,
            },
        },
        "examples": [
            {
                "neutral": "Would you like to have dinner?",
                "target": "wanna get food",
                "outputs": {
                    "base": "Would you like dinner?",
                    "sft": "want to get food?",
                    "cpo-1": "wanna get food?",
                },
                "metrics": {
                    "cpo-1": {
                        "style_probability": 0.9,
                        "semantic_similarity": 0.95,
                    }
                },
            }
        ],
    }


def test_stamp_report_escapes_private_text_and_lists_methods() -> None:
    data = _results()
    data["examples"][0]["neutral"] = "<script>alert(1)</script>"  # type: ignore[index]

    report = render_stamp_report(data)

    assert "<script>alert(1)</script>" not in report
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in report
    assert "cpo-1" in report
    assert "not proof of authorship" in report


def test_stamp_report_file_is_private(tmp_path: Path) -> None:
    import json

    source = tmp_path / "results.json"
    destination = tmp_path / "report.html"
    source.write_text(json.dumps(_results()), encoding="utf-8")

    result = render_stamp_report_file(source, destination)

    assert result["examples"] == 1
    assert destination.is_file()
    assert destination.stat().st_mode & 0o777 == 0o600
