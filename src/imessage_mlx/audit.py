from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from imessage_mlx.data.redact import contains_obvious_pii
from imessage_mlx.data.snapshot import can_open_readonly
from imessage_mlx.tokenizer.train import load_tokenizer
from imessage_mlx.utils import sha256_file, write_json


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _command_ok(
    command: list[str], root: Path, *, input_text: str | None = None
) -> tuple[bool, str]:
    result = subprocess.run(
        command,
        cwd=root,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    return result.returncode == 0, result.stdout


def completion_audit(
    project_root: str | Path,
    *,
    run_dir: str | Path = "outputs/runs/model-1m-v2-clean",
    final_dir: str | Path = "outputs/final",
    output_path: str | Path = "work/reports/completion-audit.json",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    run = root / run_dir
    final = root / final_dir
    snapshot = _load(root / "work/snapshot/manifest.json")
    extraction = _load(root / "work/reports/extraction.json")
    sessions = _load(root / "work/reports/sessions.json")
    split = _load(root / "work/reports/split-report.json")
    privacy = _load(root / "work/reports/privacy-audit.json")
    tokenizer_report = _load(root / "outputs/tokenizer/training-report.json")
    selection = _load(root / "work/reports/model-selection.json")
    training = _load(run / "training-summary.json")
    best_state = _load(run / "best/trainer-state.json")
    evaluation = _load(root / "outputs/evaluation.json")
    manifest = _load(final / "data-manifest.json")
    model_config = _load(final / "model-config.json")
    tokenizer = load_tokenizer(final / "tokenizer")

    source_readable, _ = can_open_readonly(Path.home() / "Library/Messages/chat.db")
    lint_ok, _ = _command_ok(["ruff", "check", "."], root)
    format_ok, _ = _command_ok(["ruff", "format", "--check", "."], root)
    tests_ok, test_stdout = _command_ok([sys.executable, "-m", "pytest", "-q"], root)
    diff_ok, _ = _command_ok(["git", "diff", "--check"], root)
    tracked_ok, tracked_stdout = _command_ok(["git", "ls-files", "work", "outputs"], root)
    no_private_files_tracked = tracked_ok and not tracked_stdout.strip()

    cli_ok, cli_stdout = _command_ok(
        [
            sys.executable,
            "-m",
            "imessage_mlx",
            "chat",
            "--model",
            str(final),
            "--max-new-tokens",
            "16",
            "--seed",
            "20260708",
        ],
        root,
        input_text="Synthetic final acceptance prompt.\n/quit\n",
    )

    required_final = {
        "model.safetensors",
        "model-config.json",
        "generation-config.json",
        "metrics.json",
        "data-manifest.json",
        "README.md",
        "tokenizer",
    }
    actual_final = {path.name for path in final.iterdir()}
    private_permissions_ok = all(
        (path.stat().st_mode & 0o777) == (0o700 if path.is_dir() else 0o600)
        for base in (root / "work", root / "outputs")
        for path in (base, *base.rglob("*"))
    )

    gate_a = {
        "source_database_readable_read_only": source_readable,
        "snapshot_quick_check": snapshot["quick_check"] == "ok",
        "snapshot_hash_matches_manifest": sha256_file(root / "work/snapshot/chat.db")
        == snapshot["snapshot_sha256"],
        "source_open_mode_read_only": snapshot["source_open_mode"] == "read-only",
    }
    gate_b = {
        "all_message_rows_accounted_for": extraction["all_rows_accounted_for"],
        "body_recovery_at_least_90_percent": extraction["body_recovery_rate"] >= 0.90,
        "privacy_audit_passed": privacy["passed"],
        "sessions_created": sessions["session_count"] > 0,
        "all_splits_nonempty": all(split["counts"].values()),
        "zero_cross_split_duplicate_hashes": split["cross_split_duplicate_hashes"] == 0,
        "tokenizer_used_train_only": tokenizer_report["training_source"] == "train split only",
        "model_and_tokenizer_vocab_match": model_config["vocab_size"] == tokenizer.get_vocab_size(),
    }
    gate_c = {
        "model_selected_from_real_token_count": selection["selected"] == "model-1m",
        "random_initialization_recorded": not training["initialized_from_checkpoint"],
        "no_pretrained_weights_recorded": not training["pretrained_weights_used"],
        "compiled_mlx_step_recorded": best_state["compiled_step"],
        "training_completed": training["global_step"] > 0 and training["epoch"] == 5,
        "best_and_last_checkpoints_present": all(
            (run / name).is_dir() for name in ("best", "last")
        ),
        "checkpoint_yaml_present": all(
            (run / name / "training-config.yaml").is_file() for name in ("best", "last")
        ),
        "lint_passed": lint_ok,
        "format_check_passed": format_ok,
        "tests_passed": tests_ok,
        "checkpoint_resume_test_present": "test_checkpoint"
        in " ".join(path.name for path in (root / "tests").glob("test_*.py")),
    }
    gate_d = {
        "held_out_evaluation_present": evaluation["test_tokens"] > 0,
        "beats_unigram_baseline": evaluation["beats_unigram_baseline"],
        "memorization_probe_present": evaluation["memorization"]["generation_samples"] > 0,
        "final_artifacts_complete": required_final <= actual_final,
        "final_model_hash_matches": sha256_file(final / "model.safetensors")
        == manifest["model_sha256"],
        "optimizer_excluded_from_final": "optimizer.safetensors" not in actual_final,
        "fresh_process_chat_cli_passed": cli_ok,
        "fresh_process_output_has_no_obvious_pii": not contains_obvious_pii(cli_stdout),
        "readme_present": (root / "README.md").is_file(),
        "git_diff_check_passed": diff_ok,
        "no_private_files_tracked": no_private_files_tracked,
        "private_permissions_are_0600_or_0700": private_permissions_ok,
    }
    ready = all((*gate_a.values(), *gate_b.values(), *gate_c.values(), *gate_d.values()))
    report = {
        "ready": ready,
        "gate_a": gate_a,
        "gate_b": gate_b,
        "gate_c": gate_c,
        "gate_d": gate_d,
        "aggregate_evidence": {
            "retained_messages": extraction["counts"]["retained_rows"],
            "body_recovery_rate": extraction["body_recovery_rate"],
            "session_count": sessions["session_count"],
            "split_counts": split["counts"],
            "training_tokens": selection["train_tokens"],
            "parameter_count": training["parameter_count"],
            "global_steps": training["global_step"],
            "validation_perplexity": evaluation["validation_perplexity"],
            "test_perplexity": evaluation["test_perplexity"],
            "me_turn_test_perplexity": evaluation["me_turn_test_perplexity"],
            "memorization_prone_experiment": selection["memorization_prone_experiment"],
            "test_command_summary": test_stdout.strip().splitlines()[-1] if test_stdout else "",
            "fresh_process_cli_stdout_characters": len(cli_stdout),
        },
    }
    write_json(root / output_path, report)
    return report
