from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

from imessage_mlx.data.extract import extract_messages
from imessage_mlx.data.sessions import build_sessions
from imessage_mlx.data.split import split_sessions
from imessage_mlx.dataset import encode_all_splits
from imessage_mlx.evaluate import evaluate_checkpoint
from imessage_mlx.export import export_model
from imessage_mlx.tokenizer.train import train_tokenizer
from imessage_mlx.train import train_model


def _add_conversations(database: Path, count: int = 40) -> None:
    with sqlite3.connect(database) as connection:
        for index in range(count):
            chat_id = 100 + index
            handle_id = 100 + index
            incoming_id = 1000 + index * 2
            outgoing_id = incoming_id + 1
            timestamp = 710_000_000 + index * 86_400
            connection.execute("INSERT INTO handle VALUES (?, ?)", (handle_id, f"person-{index}"))
            connection.execute(
                "INSERT INTO chat VALUES (?, ?)", (chat_id, f"synthetic-chat-{index}")
            )
            connection.execute("INSERT INTO chat_handle_join VALUES (?, ?)", (chat_id, handle_id))
            connection.execute(
                "INSERT INTO message VALUES (?, ?, ?, NULL, 0, ?, ?, 'iMessage', 0, 0, NULL, 0, 0)",
                (
                    incoming_id,
                    f"synthetic-in-{index}",
                    f"synthetic question number {index}",
                    handle_id,
                    timestamp,
                ),
            )
            connection.execute(
                "INSERT INTO message VALUES (?, ?, ?, NULL, 1, NULL, ?, 'iMessage', "
                "0, 0, NULL, 0, 0)",
                (
                    outgoing_id,
                    f"synthetic-out-{index}",
                    f"synthetic answer number {index}",
                    timestamp + 60,
                ),
            )
            connection.execute(
                "INSERT INTO chat_message_join VALUES (?, ?)", (chat_id, incoming_id)
            )
            connection.execute(
                "INSERT INTO chat_message_join VALUES (?, ?)", (chat_id, outgoing_id)
            )
        connection.commit()


def test_synthetic_end_to_end_pipeline_in_fresh_process(synthetic_db: Path, tmp_path: Path) -> None:
    _add_conversations(synthetic_db)
    work = tmp_path / "work"
    outputs = tmp_path / "outputs"
    extract_messages(
        synthetic_db,
        work / "messages.jsonl",
        work / "extraction.json",
        work / "key",
        minimum_body_recovery_rate=0.9,
    )
    build_sessions(
        work / "messages.jsonl",
        work / "sessions.jsonl",
        work / "sessions-report.json",
    )
    split_sessions(
        work / "sessions.jsonl",
        work / "splits",
        work / "split-report.json",
        guard_days=0,
    )
    train_tokenizer(
        work / "splits/train.jsonl",
        outputs / "tokenizer",
        vocab_size=256,
        minimum_frequency=1,
    )
    encode_all_splits(work / "splits", outputs / "tokenizer", work / "tokens")
    training_config = {
        "name": "integration",
        "vocab_size": 256,
        "hidden_size": 16,
        "num_layers": 1,
        "num_heads": 4,
        "intermediate_size": 32,
        "max_sequence_length": 16,
        "dropout": 0.0,
        "tie_embeddings": True,
        "batch_size": 4,
        "epochs": 1,
        "learning_rate": 0.01,
        "minimum_learning_rate": 0.001,
        "weight_decay": 0.0,
        "gradient_clip": 1.0,
        "warmup_steps": 2,
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
    assert summary["global_step"] > 0
    metrics = evaluate_checkpoint(
        outputs / "run/best",
        work / "tokens",
        outputs / "evaluation.json",
        batch_size=4,
        generation_samples=2,
    )
    assert metrics["test_loss"] > 0
    export_model(
        outputs / "run/best",
        outputs / "final",
        metrics_path=outputs / "evaluation.json",
        split_report_path=work / "split-report.json",
        split_dir=work / "splits",
    )

    code = (
        "from imessage_mlx.generate import generate_reply; "
        f"print(repr(generate_reply({str(outputs / 'final')!r}, 'hello', max_new_tokens=3)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert (outputs / "final/model.safetensors").exists()
    assert (outputs / "final/tokenizer/tokenizer.json").exists()
