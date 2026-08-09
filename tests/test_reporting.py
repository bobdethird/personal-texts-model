from pathlib import Path

from imessage_mlx.reporting import (
    render_personalization_report,
    render_personalization_report_file,
)


def _results() -> dict[str, object]:
    return {
        "model_name": "test-model",
        "embedding_model": "test-encoder",
        "examples": [
            {
                "context": [
                    {"role": "them", "content": "<script>alert('no')</script>"},
                    {"role": "you", "content": "previous reply"},
                ],
                "query": "incoming text",
                "gold": "actual reply",
                "base": "base reply",
                "retrieval": "retrieved reply",
                "retrieval_style": "styled reply",
                "content_cosine": {
                    "base": 0.1,
                    "retrieval": 0.7,
                    "retrieval_style": 0.5,
                },
                "retrieved": [
                    {
                        "query": "similar text",
                        "reply": "historical reply",
                        "score": 0.8,
                    }
                ],
            }
        ],
    }


def test_report_explains_and_escapes_personalization_results() -> None:
    report = render_personalization_report(_results())

    assert "Personal texting experiment" in report
    assert "What you actually sent" in report
    assert "Retrieval had the" in report
    assert "retrieved reply" in report
    assert "historical reply" in report
    assert "&lt;script&gt;alert" in report
    assert "<script>alert" not in report


def test_report_file_is_private_html(tmp_path: Path) -> None:
    import json

    source = tmp_path / "samples.json"
    output = tmp_path / "report.html"
    source.write_text(json.dumps(_results()), encoding="utf-8")

    result = render_personalization_report_file(source, output)

    assert result["examples"] == 1
    assert output.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert output.stat().st_mode & 0o777 == 0o600
