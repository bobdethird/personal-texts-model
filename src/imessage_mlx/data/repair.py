from __future__ import annotations

import json
from collections.abc import Callable
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from imessage_mlx.data.adapters import (
    classify_pair_signal,
    protected_facts,
    protected_facts_match,
)
from imessage_mlx.data.rewrite import validate_rewrite_pair
from imessage_mlx.utils import read_jsonl, write_json, write_jsonl

SEMANTIC_EXTRACTION_INSTRUCTION = """
Extract only the meaning of this text message as one JSON object with these keys:
intent, propositions, entities, times, numbers, placeholders, negated, uncertainty,
emotion, intensity. Preserve every fact, entity, time, number, negation, modality,
question intent, emotional meaning, and intensity. Do not quote or imitate the wording.
Apply these corpus-specific, case-insensitive meanings: "sm" means "something"; "ts"
means either "this" or "type shit". Resolve "ts" only when context makes one reading
clear; otherwise record the ambiguity instead of guessing.

Message:
""".strip()

BLIND_RECONSTRUCTION_INSTRUCTION = """
Write one clear, casually conversational text-message draft using only the semantic JSON below.
Preserve all listed facts, negation, uncertainty, emotion, and intensity. Do not add details.
Do not use idiosyncratic slang. Return only the draft.

Semantic JSON:
""".strip()

REQUIRED_SEMANTIC_KEYS = {
    "intent",
    "propositions",
    "entities",
    "times",
    "numbers",
    "placeholders",
    "negated",
    "uncertainty",
    "emotion",
    "intensity",
}


def _parse_semantics(value: str, styled_text: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict) or set(parsed) != REQUIRED_SEMANTIC_KEYS:
        raise ValueError("Semantic extraction returned an unexpected schema")
    facts = protected_facts(styled_text)
    extracted_numbers = tuple(str(item) for item in parsed["numbers"])
    extracted_placeholders = tuple(str(item) for item in parsed["placeholders"])
    if extracted_numbers != facts["numbers"]:
        raise ValueError("Semantic extraction changed numeric facts")
    if extracted_placeholders != facts["placeholders"]:
        raise ValueError("Semantic extraction changed placeholders")
    if bool(parsed["negated"]) != facts["negated"]:
        raise ValueError("Semantic extraction changed negation")
    return parsed


def repair_low_signal_pairs(
    pairs_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    *,
    generate: Callable[[str, str], str],
    limit: int = 500,
) -> dict[str, Any]:
    if limit <= 0:
        raise ValueError("Repair pilot limit must be positive")
    pairs = [validate_rewrite_pair(record) for record in read_jsonl(pairs_path)]
    all_eligible = [
        index
        for index, pair in enumerate(pairs)
        if classify_pair_signal(str(pair["neutral_text"]), str(pair["styled_text"]))
        in {"exact", "surface_only", "high_overlap"}
    ]
    eligible = all_eligible[:limit]
    accepted_ids: list[str] = []
    rejection_counts: dict[str, int] = {}

    for index in eligible:
        pair = pairs[index]
        styled = str(pair["styled_text"])
        try:
            extracted = generate(
                "extract",
                f"{SEMANTIC_EXTRACTION_INSTRUCTION}\n{styled}",
            )
            semantics = _parse_semantics(extracted, styled)
            semantic_payload = json.dumps(semantics, sort_keys=True, ensure_ascii=False)
            reconstructed = generate(
                "reconstruct",
                f"{BLIND_RECONSTRUCTION_INSTRUCTION}\n{semantic_payload}",
            ).strip()
            if not reconstructed:
                raise ValueError("Reconstruction was empty")
            if not protected_facts_match(styled, reconstructed):
                raise ValueError("Reconstruction changed protected facts")
            if SequenceMatcher(None, styled.casefold(), reconstructed.casefold()).ratio() < 0.45:
                raise ValueError("Reconstruction failed the local content floor")
            if classify_pair_signal(reconstructed, styled) != "substantive":
                raise ValueError("Reconstruction did not repair the low-signal pair")
            pair["neutral_text"] = reconstructed
            accepted_ids.append(str(pair["pair_id"]))
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            name = type(error).__name__
            rejection_counts[name] = rejection_counts.get(name, 0) + 1

    write_jsonl(output_path, pairs)
    report = {
        "schema_version": 1,
        "task": "targeted_rewrite_pair_repair",
        "input_pairs": len(pairs),
        "eligible_low_signal_pairs": len(all_eligible),
        "selected": len(eligible),
        "accepted": len(accepted_ids),
        "rejected": len(eligible) - len(accepted_ids),
        "accepted_pair_ids": accepted_ids,
        "rejection_counts": rejection_counts,
        "failed_checks_retained_existing_pair": True,
        "original_or_generated_text_persisted_in_report": False,
        "private_local_artifact": True,
    }
    write_json(report_path, report)
    return report
