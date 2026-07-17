from pathlib import Path

from typer.testing import CliRunner

from imessage_mlx.cli import app


def test_build_llm_dataset_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("imessage_mlx.cli.load_dotenv", lambda: None)

    result = CliRunner().invoke(app, ["build-llm-dataset"])

    assert result.exit_code != 0
    assert "OPENAI_API_KEY" in result.output


def test_censor_llm_dataset_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("imessage_mlx.cli.load_dotenv", lambda: None)

    result = CliRunner().invoke(app, ["censor-llm-dataset"])

    assert result.exit_code != 0
    assert "OPENAI_API_KEY" in result.output


def test_train_requires_data_derived_model_selection(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "train",
            "--config",
            "configs/model-1m.yaml",
            "--selection-report",
            str(tmp_path / "missing-selection.json"),
        ],
    )

    assert result.exit_code != 0
    assert "corpus-stats" in result.output
