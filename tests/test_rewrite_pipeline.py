from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from imessage_mlx.data.rewrite import prepare_rewrite_dataset
from imessage_mlx.dataset import encode_all_rewrite_splits
from imessage_mlx.evaluate import evaluate_checkpoint
from imessage_mlx.export import export_model
from imessage_mlx.tokenizer.train import train_tokenizer
from imessage_mlx.train import train_model
from imessage_mlx.utils import write_json


def test_synthetic_rewrite_pipeline_exports_and_loads_in_fresh_process(
    tmp_path: Path,
) -> None:
    source = Path(__file__).parent / "fixtures/synthetic_rewrites.jsonl"
    work = tmp_path / "work"
    outputs = tmp_path / "outputs"
    prepare_rewrite_dataset(
        source,
        work / "processed.jsonl",
        work / "splits",
        work / "preparation.json",
        work / "split-report.json",
        guard_days=0,
    )
    train_tokenizer(
        work / "splits/train.jsonl",
        outputs / "tokenizer",
        vocab_size=256,
        minimum_frequency=1,
    )
    encode_all_rewrite_splits(
        work / "splits",
        outputs / "tokenizer",
        work / "tokens",
        context_length=96,
    )
    training_config = {
        "name": "smoke",
        "task": "rewrite",
        "objective": "target_only",
        "vocab_size": 256,
        "hidden_size": 16,
        "num_layers": 1,
        "num_heads": 4,
        "intermediate_size": 32,
        "max_sequence_length": 96,
        "dropout": 0.0,
        "tie_embeddings": True,
        "batch_size": 2,
        "epochs": 1,
        "learning_rate": 0.01,
        "minimum_learning_rate": 0.001,
        "weight_decay": 0.0,
        "gradient_clip": 1.0,
        "warmup_steps": 1,
        "evaluation_interval": 1000,
        "checkpoint_interval": 1000,
        "early_stopping_patience": 5,
        "seed": 42,
    }
    summary = train_model(
        training_config,
        work / "tokens",
        outputs / "tokenizer",
        outputs / "run",
        compile_step=True,
    )
    assert summary["objective"] == "target_only"
    assert summary["supervised_tokens_trained"] > 0
    wrong_context_config = {**training_config, "max_sequence_length": 48}
    with pytest.raises(ValueError, match="encoded for context 96"):
        train_model(
            wrong_context_config,
            work / "tokens",
            outputs / "tokenizer",
            outputs / "wrong-context-run",
            compile_step=False,
        )
    with pytest.raises(ValueError, match="supports reply checkpoints only"):
        evaluate_checkpoint(
            outputs / "run/best",
            work / "tokens",
            outputs / "invalid-evaluation.json",
        )

    manifest = export_model(
        outputs / "run/best",
        outputs / "final",
        split_report_path=work / "split-report.json",
        split_dir=work / "splits",
    )
    assert manifest["capabilities"] == ["rewrite"]
    assert json.loads((outputs / "final/generation-config.json").read_text())["task"] == "rewrite"
    write_json(outputs / "reply-metrics.json", {"task": "reply"})
    with pytest.raises(ValueError, match="Cannot attach"):
        export_model(
            outputs / "run/best",
            outputs / "invalid-final",
            metrics_path=outputs / "reply-metrics.json",
        )

    code = (
        "from imessage_mlx.generate import generate_rewrite; "
        f"print(repr(generate_rewrite({str(outputs / 'final')!r}, "
        "'I will arrive later.', max_new_tokens=3)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    reply_code = (
        "from imessage_mlx.generate import generate_reply; "
        f"generate_reply({str(outputs / 'final')!r}, 'hello', max_new_tokens=3)"
    )
    reply_result = subprocess.run(
        [sys.executable, "-c", reply_code],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert reply_result.returncode != 0
    assert "not trained for reply" in reply_result.stderr

    train_tokenizer(
        work / "processed.jsonl",
        outputs / "different-tokenizer",
        vocab_size=512,
        minimum_frequency=1,
    )
    with pytest.raises(ValueError, match="differs from the checkpoint"):
        train_model(
            training_config,
            work / "tokens",
            outputs / "different-tokenizer",
            outputs / "resume-attempt",
            resume_from=outputs / "run/last",
            compile_step=False,
        )
    wrong_task_config = {**training_config, "task": "reply", "objective": "causal"}
    with pytest.raises(ValueError, match="Cannot change task or objective"):
        train_model(
            wrong_task_config,
            work / "tokens",
            outputs / "tokenizer",
            outputs / "wrong-task-resume",
            resume_from=outputs / "run/last",
            compile_step=False,
        )
