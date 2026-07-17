from pathlib import Path

import pytest

from imessage_mlx.config import load_yaml
from imessage_mlx.seq2seq_profile import format_seq2seq_source, resolve_seq2seq_profile

ROOT = Path(__file__).resolve().parents[1]


def test_bart_defaults_preserve_existing_behavior() -> None:
    profile = resolve_seq2seq_profile(load_yaml(ROOT / "configs/adapter-bart-base.yaml"))

    assert profile.architecture == "bart"
    assert profile.source_prefix == ""
    assert profile.lora_target_modules == ("q_proj", "v_proj")
    assert format_seq2seq_source("hello", profile) == "hello"


@pytest.mark.parametrize(
    ("config_name", "architecture", "prefix", "targets"),
    (
        (
            "adapter-flan-t5-small.yaml",
            "flan_t5",
            "rewrite in the learned personal texting style while preserving meaning: ",
            ("q", "v"),
        ),
        (
            "adapter-opus-mt-gem-gem.yaml",
            "marian",
            ">>eng<< ",
            ("q_proj", "v_proj"),
        ),
    ),
)
def test_competitor_configs_resolve(
    config_name: str,
    architecture: str,
    prefix: str,
    targets: tuple[str, ...],
) -> None:
    profile = resolve_seq2seq_profile(load_yaml(ROOT / f"configs/{config_name}"))

    assert profile.architecture == architecture
    assert profile.source_prefix == prefix
    assert profile.lora_target_modules == targets
    assert format_seq2seq_source("hello", profile) == f"{prefix}hello"


def test_profile_rejects_invalid_target_modules() -> None:
    with pytest.raises(ValueError, match="non-empty list"):
        resolve_seq2seq_profile(
            {
                "architecture": "flan_t5",
                "lora_target_modules": [],
            }
        )


def test_profile_rejects_non_seq2seq_architecture() -> None:
    with pytest.raises(ValueError, match="bart.*flan_t5.*marian"):
        resolve_seq2seq_profile({"architecture": "qwen"})
