from pathlib import Path

from typer.testing import CliRunner

from imessage_mlx.cli import app
from imessage_mlx.utils import write_json


def test_rewrite_command_forwards_one_shot_options(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def fake_generate(model, draft, **options):
        captured.update({"model": model, "draft": draft, **options})
        return "casual result"

    monkeypatch.setattr("imessage_mlx.cli.generate_rewrite", fake_generate)
    result = CliRunner().invoke(
        app,
        [
            "rewrite",
            "Neutral draft",
            "--model",
            str(tmp_path),
            "--temperature",
            "0.2",
            "--max-new-tokens",
            "12",
        ],
    )

    assert result.exit_code == 0
    assert result.stdout == "casual result\n"
    assert captured["model"] == tmp_path.resolve()
    assert captured["draft"] == "Neutral draft"
    assert captured["temperature"] == 0.2
    assert captured["max_new_tokens"] == 12


def test_pair_generation_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("imessage_mlx.cli.load_dotenv", lambda: None)

    result = CliRunner().invoke(app, ["generate-rewrite-pairs"])

    assert result.exit_code != 0
    assert "OPENAI_API_KEY" in result.output


def test_rewrite_train_uses_task_specific_selection_report(tmp_path: Path, monkeypatch) -> None:
    tokenizer_dir = tmp_path / "tokenizer"
    write_json(
        tokenizer_dir / "training-report.json",
        {"requested_vocab_size": 2048},
    )
    selection = {
        "schema_version": 1,
        "task": "rewrite",
        "selection_scope": "training_split",
        "eligible_to_train": True,
        "selected": "model-rewrite-190k",
        "vocab_size": 2048,
    }
    selection_path = tmp_path / "selection.json"
    write_json(selection_path, selection)

    class FakeTokenizer:
        def get_vocab_size(self):
            return 2048

    captured = {}

    def fake_train(*args, **options):
        captured["args"] = args
        captured["options"] = options
        return {"trained": True}

    monkeypatch.setattr("imessage_mlx.cli.load_tokenizer", lambda _path: FakeTokenizer())
    monkeypatch.setattr("imessage_mlx.cli.train_model", fake_train)
    result = CliRunner().invoke(
        app,
        [
            "train",
            "--config",
            "configs/model-rewrite-190k.yaml",
            "--data",
            str(tmp_path / "tokens"),
            "--tokenizer",
            str(tokenizer_dir),
            "--output",
            str(tmp_path / "run"),
            "--selection-report",
            str(selection_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"][0]["name"] == "model-rewrite-190k"

    selection["eligible_to_train"] = False
    selection["selected"] = None
    write_json(selection_path, selection)
    rejected = CliRunner().invoke(
        app,
        [
            "train",
            "--config",
            "configs/model-rewrite-190k.yaml",
            "--tokenizer",
            str(tokenizer_dir),
            "--selection-report",
            str(selection_path),
        ],
    )
    assert rejected.exit_code != 0
    assert "does not satisfy" in rejected.output
