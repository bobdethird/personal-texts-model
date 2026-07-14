from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SEQ2SEQ_ARCHITECTURES = frozenset({"bart", "flan_t5", "marian"})


@dataclass(frozen=True)
class Seq2SeqProfile:
    architecture: str
    source_prefix: str
    lora_target_modules: tuple[str, ...]
    base_model_license: str


_DEFAULTS = {
    "bart": Seq2SeqProfile(
        architecture="bart",
        source_prefix="",
        lora_target_modules=("q_proj", "v_proj"),
        base_model_license="not_declared_in_hugging_face_model_card",
    ),
    "flan_t5": Seq2SeqProfile(
        architecture="flan_t5",
        source_prefix="rewrite in the learned personal texting style while preserving meaning: ",
        lora_target_modules=("q", "v"),
        base_model_license="apache-2.0",
    ),
    "marian": Seq2SeqProfile(
        architecture="marian",
        source_prefix="",
        lora_target_modules=("q_proj", "v_proj"),
        base_model_license="license_provenance_requires_review",
    ),
}


def resolve_seq2seq_profile(config: dict[str, Any]) -> Seq2SeqProfile:
    architecture = str(config.get("architecture", ""))
    if architecture not in SEQ2SEQ_ARCHITECTURES:
        supported = ", ".join(sorted(SEQ2SEQ_ARCHITECTURES))
        raise ValueError(f"Seq2seq architecture must be one of: {supported}")
    default = _DEFAULTS[architecture]

    source_prefix = config.get("source_prefix", default.source_prefix)
    if not isinstance(source_prefix, str):
        raise ValueError("source_prefix must be a string")

    targets = config.get("lora_target_modules", list(default.lora_target_modules))
    if (
        not isinstance(targets, list)
        or not targets
        or any(not isinstance(target, str) or not target for target in targets)
    ):
        raise ValueError("lora_target_modules must be a non-empty list of strings")

    license_name = config.get("base_model_license", default.base_model_license)
    if not isinstance(license_name, str) or not license_name:
        raise ValueError("base_model_license must be a non-empty string")

    return Seq2SeqProfile(
        architecture=architecture,
        source_prefix=source_prefix,
        lora_target_modules=tuple(targets),
        base_model_license=license_name,
    )


def format_seq2seq_source(text: str, profile: Seq2SeqProfile) -> str:
    return f"{profile.source_prefix}{text}"
