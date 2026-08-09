"""Adapted STAMP pipeline for private text-message style transfer.

Public objects are loaded on first access. Importing :mod:`imessage_mlx.stamp`
therefore never imports Torch, Transformers, Datasets, PEFT, or TRL.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

STAMP_FORMAT_VERSION = 1

_EXPORTS = {
    "CPO_PAIR_FORMAT": "preference",
    "EVALUATION_FORMAT": "evaluate",
    "NEUTRAL_PAIR_FORMAT": "neutralize",
    "CandidateReward": "rewards",
    "ClassifierTrainingConfig": "classifier",
    "CPOTrainingConfig": "preference",
    "HopeFearSelection": "rewards",
    "NeutralValidationConfig": "neutralize",
    "NeutralValidationResult": "neutralize",
    "NeutralizationPromptConfig": "neutralize",
    "RewardExponents": "rewards",
    "SFTTrainingConfig": "sft",
    "binary_classification_metrics": "classifier",
    "build_classifier_examples": "classifier",
    "build_neutralization_messages": "neutralize",
    "build_rewrite_messages": "sft",
    "build_sft_example": "sft",
    "build_sft_examples": "sft",
    "candidate_reward_from_scores": "rewards",
    "cpo_pair_loss": "preference",
    "dynamic_reward_exponents": "rewards",
    "evaluate_candidate_rows": "evaluate",
    "evaluate_model_candidates": "evaluate",
    "format_rewrite_prompt": "sft",
    "format_cpo_prompt": "preference",
    "make_cpo_pair": "preference",
    "make_cpo_pair_from_selection": "preference",
    "make_neutral_pair": "neutralize",
    "normalized_base_model_likelihood": "rewards",
    "parse_neutral_output": "neutralize",
    "parse_rewrite_output": "sft",
    "run_neutralization": "neutralize",
    "select_hope_and_fear": "rewards",
    "train_cpo": "preference",
    "train_rewrite_sft": "sft",
    "train_style_classifier": "classifier",
    "validate_neutral": "neutralize",
    "write_cpo_pairs": "preference",
}

__all__ = ["STAMP_FORMAT_VERSION", *_EXPORTS]


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
