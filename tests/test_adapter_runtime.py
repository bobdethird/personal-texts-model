import subprocess
from pathlib import Path

import pytest

from imessage_mlx.adapter_runtime import (
    environment_python,
    promote_adapter,
    rewrite_with_adapter,
    setup_adapter_environment,
    train_adapter,
)
from imessage_mlx.utils import write_json


def test_environment_python_is_isolated() -> None:
    assert environment_python("work/envs/bart") == Path("work/envs/bart/bin/python")


def test_setup_rejects_unknown_architecture(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="bart.*qwen"):
        setup_adapter_environment("unknown", tmp_path / "env")


def test_promotion_requires_passing_evaluation_and_hashes_artifact(tmp_path: Path) -> None:
    adapter = tmp_path / "run"
    adapter.mkdir()
    (adapter / "adapter.safetensors").write_bytes(b"private-adapter")
    write_json(
        adapter / "training-report.json",
        {
            "architecture": "flan_t5",
            "base_model": "google/flan-t5-small",
            "base_revision": "pinned-revision",
            "base_model_license": "apache-2.0",
            "source_prefix": "rewrite: ",
            "lora_target_modules": ["q", "v"],
        },
    )
    evaluation = tmp_path / "evaluation.json"
    write_json(evaluation, {"ready_to_promote": False})
    review = tmp_path / "review.json"
    write_json(
        review,
        {
            "human_approved": True,
            "reviewed_target_groups": 20,
            "changed_fact_failures": 0,
        },
    )

    with pytest.raises(ValueError, match="did not pass"):
        promote_adapter(
            adapter,
            evaluation,
            tmp_path / "final",
            review_summary_path=review,
        )

    write_json(
        evaluation,
        {
            "ready_to_promote": True,
            "semantic": {"mean_generated_source_similarity": 0.99},
        },
    )
    with pytest.raises(ValueError, match="human review"):
        promote_adapter(adapter, evaluation, tmp_path / "final")

    manifest = promote_adapter(
        adapter,
        evaluation,
        tmp_path / "final",
        review_summary_path=review,
    )

    assert manifest["evaluation_ready_to_promote"] is True
    assert "adapter/adapter.safetensors" in manifest["adapter_files"]
    assert (tmp_path / "final/evaluation.json").exists()
    assert manifest["pretrained_weights_used"] is True
    assert manifest["adapter_fused"] is False
    assert manifest["base_revision"] == "pinned-revision"
    assert manifest["base_model_license"] == "apache-2.0"
    assert manifest["architecture"] == "flan_t5"
    assert manifest["source_prefix"] == "rewrite: "
    assert manifest["lora_target_modules"] == ["q", "v"]
    assert manifest["human_review_approved"] is True
    assert (tmp_path / "final/human-review-summary.json").exists()

    (tmp_path / "final/old-marker").write_text("rollback")
    second = promote_adapter(
        adapter,
        evaluation,
        tmp_path / "final",
        review_summary_path=review,
    )
    rollback = Path(second["rollback_artifact"])
    assert (rollback / "old-marker").read_text() == "rollback"


def test_rewrite_rejects_artifact_without_rewrite_capability(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    write_json(artifact / "data-manifest.json", {"capabilities": ["reply"]})
    config = tmp_path / "adapter.yaml"
    config.write_text("architecture: bart\n")

    with pytest.raises(ValueError, match="not promoted for rewrite"):
        rewrite_with_adapter(
            "draft",
            config,
            artifact,
            tmp_path / "env",
            project_root=tmp_path,
        )


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
