"""Content-preserving neutralization data preparation.

The functions in this module intentionally depend only on the standard library.
Model loading is delegated to an injected generator so neutralization can run
locally, remotely, or in tests without importing an inference framework.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from imessage_mlx.utils import read_jsonl, sha256_text, write_json, write_jsonl

NEUTRAL_PAIR_FORMAT = "stamp-neutral-pair-v1"

DEFAULT_SYSTEM_PROMPT = """\
You are preparing parallel data for authorship-style transfer. Rewrite only the
entire TARGET burst into one plain, conventional English draft while preserving every fact,
intent, level of certainty, negation, number, named entity, placeholder, and
meaningful detail. Remove idiosyncratic texting style, slang, abbreviations,
emoji, expressive punctuation, and stylistic bubble boundaries. Do not answer the
message or add information."""

DEFAULT_USER_TEMPLATE = """\
Neutralize the complete thought represented by the {bubble_count} TARGET bubble(s).
Combine the burst into one neutral draft. CONTEXT is provided only to disambiguate
the target and must not be rewritten.

CONTEXT:
{context_json}

TARGET:
{target_json}

Return JSON only in this exact shape:
{{"neutral_text": "one content-preserving neutral draft"}}"""

_PLACEHOLDER_RE = re.compile(
    r"<\|[A-Za-z0-9_-]+\|>|"
    r"\{\{[A-Za-z0-9_.:-]+\}\}|"
    r"\[(?:[A-Z][A-Z0-9_-]{1,})\]|"
    r"<(?:[A-Z][A-Z0-9_-]{1,})>"
)
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])([+-]?\d[\d,]*(?::\d{2})?(?:\.\d+)?)([kmb])?"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
_NEGATIONS = frozenset(
    {
        "ain't",
        "aint",
        "cannot",
        "can't",
        "cant",
        "couldn't",
        "couldnt",
        "didn't",
        "didnt",
        "doesn't",
        "doesnt",
        "don't",
        "dont",
        "hardly",
        "isn't",
        "isnt",
        "neither",
        "never",
        "no",
        "nobody",
        "none",
        "nope",
        "nor",
        "not",
        "nothing",
        "nowhere",
        "rarely",
        "shouldn't",
        "shouldnt",
        "wasn't",
        "wasnt",
        "weren't",
        "werent",
        "without",
        "won't",
        "wont",
        "wouldn't",
        "wouldnt",
    }
)


@dataclass(frozen=True, slots=True)
class NeutralizationPromptConfig:
    """Prompts used for a Qwen-compatible chat generation call."""

    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_template: str = DEFAULT_USER_TEMPLATE


@dataclass(frozen=True, slots=True)
class NeutralValidationConfig:
    """Hard content checks applied before a generated neutral is accepted."""

    min_length_ratio: float = 0.35
    max_length_ratio: float = 2.5
    max_characters: int | None = None
    min_semantic_similarity: float | None = None

    def __post_init__(self) -> None:
        if self.min_length_ratio <= 0:
            raise ValueError("min_length_ratio must be positive")
        if self.max_length_ratio < self.min_length_ratio:
            raise ValueError("max_length_ratio must be at least min_length_ratio")
        if self.max_characters is not None and self.max_characters < 1:
            raise ValueError("max_characters must be positive")
        if self.min_semantic_similarity is not None and not (
            0 <= self.min_semantic_similarity <= 1
        ):
            raise ValueError("min_semantic_similarity must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class NeutralValidationResult:
    valid: bool
    errors: tuple[str, ...]
    length_ratio: float
    semantic_similarity: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "length_ratio": self.length_ratio,
            "semantic_similarity": self.semantic_similarity,
        }


def _coerce_bubbles(value: str | Sequence[str], *, field: str) -> list[str]:
    if isinstance(value, str):
        bubbles = [value]
    elif isinstance(value, Sequence):
        bubbles = [str(item) for item in value]
    else:
        raise TypeError(f"{field} must be a string or sequence of strings")
    if not bubbles:
        raise ValueError(f"{field} must contain at least one bubble")
    return bubbles


def build_neutralization_messages(
    original_bubbles: str | Sequence[str],
    *,
    context_bubbles: str | Sequence[str] = (),
    config: NeutralizationPromptConfig | None = None,
) -> list[dict[str, str]]:
    """Build configurable chat messages suitable for ``apply_chat_template``."""

    prompt = config or NeutralizationPromptConfig()
    target = _coerce_bubbles(original_bubbles, field="original_bubbles")
    context = (
        [context_bubbles]
        if isinstance(context_bubbles, str)
        else [str(item) for item in context_bubbles]
    )
    replacements = {
        "{bubble_count}": str(len(target)),
        "{context_json}": json.dumps(context, ensure_ascii=False),
        "{target_json}": json.dumps(target, ensure_ascii=False),
    }
    user_content = prompt.user_template
    for marker, replacement in replacements.items():
        user_content = user_content.replace(marker, replacement)
    return [
        {"role": "system", "content": prompt.system_prompt.strip()},
        {"role": "user", "content": user_content.strip()},
    ]


def parse_neutral_output(
    output: str | Mapping[str, Any] | Sequence[str],
    *,
    expected_bubbles: int | None = None,
) -> list[str]:
    """Parse the strict JSON response while tolerating a Markdown JSON fence."""

    value: Any = output
    if isinstance(output, str):
        text = output.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1)
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError("Neutralizer output must be valid JSON") from error
    if isinstance(value, Mapping):
        value = value.get("neutral_text", value.get("neutral_bubbles"))
    if isinstance(value, str):
        value = [value]
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError("Neutralizer output must contain neutral_text or neutral_bubbles")
    bubbles = [str(item) for item in value]
    if expected_bubbles is not None and len(bubbles) != expected_bubbles:
        raise ValueError(
            f"Expected {expected_bubbles} neutral bubbles, received {len(bubbles)}"
        )
    return bubbles


def normalized_numbers(text: str) -> Counter[str]:
    """Return value-normalized numbers, preserving multiplicity and clock times."""

    numbers: Counter[str] = Counter()
    multipliers = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
    for match in _NUMBER_RE.finditer(text):
        raw, suffix = match.groups()
        if ":" in raw:
            sign = "-" if raw.startswith("-") else ""
            unsigned = raw.lstrip("+-")
            hours, minutes = unsigned.replace(",", "").split(":", maxsplit=1)
            canonical = (
                f"{sign}{int(hours)}"
                if minutes == "00"
                else f"{sign}{int(hours)}:{minutes}"
            )
        else:
            try:
                number = Decimal(raw.replace(",", ""))
            except InvalidOperation:
                continue
            if suffix:
                number *= multipliers[suffix.lower()]
            canonical = format(number.normalize(), "f")
        numbers[canonical] += 1
    return numbers


def placeholder_counts(text: str) -> Counter[str]:
    return Counter(_PLACEHOLDER_RE.findall(text))


def negation_count(text: str) -> int:
    """Count explicit negative operators, allowing contraction expansion."""

    normalized = text.lower().replace("’", "'")
    return sum(token in _NEGATIONS for token in _WORD_RE.findall(normalized))


def validate_neutral(
    original_bubbles: str | Sequence[str],
    neutral_bubbles: str | Sequence[str],
    *,
    config: NeutralValidationConfig | None = None,
    semantic_scorer: Callable[[str, str], float] | None = None,
) -> NeutralValidationResult:
    """Validate content invariants for one original/neutral multi-bubble pair."""

    rules = config or NeutralValidationConfig()
    original = _coerce_bubbles(original_bubbles, field="original_bubbles")
    neutral = _coerce_bubbles(neutral_bubbles, field="neutral_bubbles")
    errors: list[str] = []
    if any(not bubble.strip() for bubble in neutral):
        errors.append("empty_bubble")

    original_text = "\n".join(original)
    neutral_text = "\n".join(neutral)
    if normalized_numbers(original_text) != normalized_numbers(neutral_text):
        errors.append("numbers_changed")
    if negation_count(original_text) != negation_count(neutral_text):
        errors.append("negation_changed")
    if placeholder_counts(original_text) != placeholder_counts(neutral_text):
        errors.append("placeholders_changed")

    source_length = max(1, len(original_text.strip()))
    length_ratio = len(neutral_text.strip()) / source_length
    if not rules.min_length_ratio <= length_ratio <= rules.max_length_ratio:
        errors.append("length_out_of_bounds")
    if rules.max_characters is not None and len(neutral_text) > rules.max_characters:
        errors.append("length_out_of_bounds")

    semantic_similarity = None
    if rules.min_semantic_similarity is not None:
        if semantic_scorer is None:
            errors.append("semantic_score_missing")
        else:
            semantic_similarity = float(semantic_scorer(original_text, neutral_text))
            if not 0 <= semantic_similarity <= 1:
                errors.append("semantic_score_out_of_range")
            elif semantic_similarity < rules.min_semantic_similarity:
                errors.append("semantic_similarity_too_low")

    return NeutralValidationResult(
        valid=not errors,
        errors=tuple(dict.fromkeys(errors)),
        length_ratio=length_ratio,
        semantic_similarity=semantic_similarity,
    )


def make_neutral_pair(
    *,
    pair_id: str,
    original_bubbles: str | Sequence[str],
    neutral_bubbles: str | Sequence[str],
    split: str | None = None,
    source_id: str | None = None,
    context_bubbles: str | Sequence[str] = (),
    validation: NeutralValidationResult | None = None,
) -> dict[str, Any]:
    """Create the versioned on-disk original/neutral pair record."""

    original = _coerce_bubbles(original_bubbles, field="original_bubbles")
    neutral = _coerce_bubbles(neutral_bubbles, field="neutral_bubbles")
    context = (
        [context_bubbles]
        if isinstance(context_bubbles, str)
        else [str(item) for item in context_bubbles]
    )
    record: dict[str, Any] = {
        "format": NEUTRAL_PAIR_FORMAT,
        "pair_id": str(pair_id),
        "original": original,
        "neutral": neutral,
        "context": context,
    }
    if split is not None:
        record["split"] = str(split)
    if source_id is not None:
        record["source_id"] = str(source_id)
    if validation is not None:
        record["validation"] = validation.as_dict()
    return record


def _record_bubbles(record: Mapping[str, Any]) -> list[str]:
    for key in ("original", "target_bubbles", "reply"):
        if key in record:
            return _coerce_bubbles(record[key], field=key)
    raise ValueError("Neutralization source record requires original, target_bubbles, or reply")


def _record_context(record: Mapping[str, Any]) -> list[str]:
    if "context_bubbles" in record:
        return _coerce_bubbles(record["context_bubbles"], field="context_bubbles")
    messages = record.get("context_messages", record.get("messages", []))
    if isinstance(messages, Sequence) and not isinstance(messages, str):
        return [
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, Mapping) and str(message.get("content", "")).strip()
        ]
    return []


def run_neutralization(
    records: Iterable[Mapping[str, Any]],
    output_path: str | Path,
    generator: Callable[[list[dict[str, str]]], str | Mapping[str, Any] | Sequence[str]],
    *,
    prompt_config: NeutralizationPromptConfig | None = None,
    validation_config: NeutralValidationConfig | None = None,
    semantic_scorer: Callable[[str, str], float] | None = None,
    max_attempts: int = 2,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run an injected generator, checkpointing each accepted pair atomically.

    Existing valid pair IDs in ``output_path`` are skipped, so an interrupted run
    can safely restart with the same source records.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    destination = Path(output_path)
    accepted = list(read_jsonl(destination)) if destination.is_file() else []
    completed: set[str] = set()
    for pair in accepted:
        if pair.get("format") != NEUTRAL_PAIR_FORMAT:
            raise ValueError(f"Unsupported neutral pair format {pair.get('format')!r}")
        pair_id = str(pair.get("pair_id", ""))
        if not pair_id or pair_id in completed:
            raise ValueError("Existing neutral pairs must have unique nonempty pair IDs")
        completed.add(pair_id)

    skipped = generated = 0
    failures: list[dict[str, Any]] = []
    for record in records:
        original = _record_bubbles(record)
        source_id = str(record.get("pair_id") or record.get("source_id") or "")
        pair_id = source_id or sha256_text(json.dumps(original, ensure_ascii=False))[:24]
        if pair_id in completed:
            skipped += 1
            continue
        context = _record_context(record)
        messages = build_neutralization_messages(
            original,
            context_bubbles=context,
            config=prompt_config,
        )
        attempt_errors: list[str] = []
        for _attempt in range(max_attempts):
            try:
                raw_output = generator(messages)
                neutral = parse_neutral_output(raw_output)
                validation = validate_neutral(
                    original,
                    neutral,
                    config=validation_config,
                    semantic_scorer=semantic_scorer,
                )
                if not validation.valid:
                    attempt_errors.extend(validation.errors)
                    continue
                pair = make_neutral_pair(
                    pair_id=pair_id,
                    source_id=source_id or None,
                    split=str(record["split"]) if "split" in record else None,
                    original_bubbles=original,
                    neutral_bubbles=neutral,
                    context_bubbles=context,
                    validation=validation,
                )
                accepted.append(pair)
                completed.add(pair_id)
                write_jsonl(destination, accepted)
                generated += 1
                break
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                attempt_errors.append(str(error))
        else:
            failures.append(
                {
                    "pair_id": pair_id,
                    "errors": list(dict.fromkeys(attempt_errors)) or ["generation_failed"],
                }
            )

    summary = {
        "format": NEUTRAL_PAIR_FORMAT,
        "output_path": str(destination),
        "pairs": len(accepted),
        "generated": generated,
        "skipped_existing": skipped,
        "failed": len(failures),
        "failures": failures,
    }
    if report_path is not None:
        write_json(report_path, summary)
    return summary
