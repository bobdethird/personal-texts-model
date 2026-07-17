import subprocess
from pathlib import Path

import pytest

from imessage_mlx.adapter_runtime import (
    environment_python,
    setup_adapter_environment,
    train_adapter,
)


def test_environment_python_is_isolated() -> None:
    assert environment_python("work/envs/bart") == Path("work/envs/bart/bin/python")


def test_setup_rejects_unknown_architecture(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="bart.*qwen"):
        setup_adapter_environment("unknown", tmp_path / "env")


def test_qwen_training_masks_prompt_in_isolated_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = tmp_path / "qwen.yaml"
    config.write_text(
        "\n".join(
            (
                "architecture: qwen",
                "base_model: local/qwen",
                "revision: pinned",
                "iterations: 10",
                "gradient_accumulation_steps: 2",
            )
        )
    )
    python = tmp_path / "env/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    captured: list[list[str]] = []

    def fake_run(command, **_options):
        captured.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("imessage_mlx.adapter_runtime._run", fake_run)
    monkeypatch.setattr(
        "imessage_mlx.adapter_runtime._pinned_huggingface_snapshot",
        lambda *_args: tmp_path / "model",
    )

    train_adapter(
        config,
        tmp_path / "data",
        tmp_path / "run",
        tmp_path / "env",
        project_root=tmp_path,
    )

    assert "--mask-prompt" in captured[-1]
    assert captured[-1][captured[-1].index("--data") + 1].endswith("/data/mlx")


@pytest.mark.parametrize(
    ("architecture", "expected_command"),
    (("bart", "train-bart"), ("flan_t5", "train-seq2seq"), ("marian", "train-seq2seq")),
)
def test_seq2seq_training_uses_shared_data_and_worker(
    tmp_path: Path,
    monkeypatch,
    architecture: str,
    expected_command: str,
) -> None:
    config = tmp_path / f"{architecture}.yaml"
    config.write_text(
        "\n".join(
            (
                f"architecture: {architecture}",
                "base_model: local/model",
                "revision: pinned",
            )
        )
    )
    python = tmp_path / "env/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    captured: list[list[str]] = []

    def fake_run(command, **_options):
        captured.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("imessage_mlx.adapter_runtime._run", fake_run)

    train_adapter(
        config,
        tmp_path / "data",
        tmp_path / "run",
        tmp_path / "env",
        project_root=tmp_path,
    )

    assert expected_command in captured[-1]
    assert captured[-1][captured[-1].index("--data") + 1].endswith("/data/bart")
