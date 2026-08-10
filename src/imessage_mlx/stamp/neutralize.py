"""Content-preserving neutralization data preparation.

The functions in this module intentionally depend only on the standard library.
Model loading is delegated to an injected generator so neutralization can run
locally, remotely, or in tests without importing an inference framework.
"""

from __future__ import annotations

import inspect
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from imessage_mlx.utils import extend_jsonl, read_jsonl, sha256_text, write_json

NEUTRAL_PAIR_FORMAT = "stamp-neutral-pair-v1"

DEFAULT_SYSTEM_PROMPT = """\
You are preparing parallel data for authorship-style transfer. Rewrite only the
entire TARGET burst into one plain, conventional English draft while preserving every fact,
intent, level of certainty, negation, number, named entity, placeholder, and
meaningful detail. Remove idiosyncratic texting style, slang, abbreviations,
emoji, expressive punctuation, and stylistic bubble boundaries. Do not answer the
message or add information.

Write the draft as the sender speaking, in the same grammatical person as the
TARGET. Keep first-person pronouns first person and keep questions as questions
addressed to the reader. Never describe the message from the outside: do not
write "the sender asks", "they state that", or any other reported speech.

The draft must be a genuine rewrite. If the TARGET is already plain English,
still produce a neutral phrasing of it rather than copying it back verbatim."""

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
_FIRST_PERSON = frozenset(
    {"i", "id", "ill", "im", "ive", "me", "mine", "my", "myself", "our", "ours", "us", "we"}
)
_REPORTED_SPEECH_RE = re.compile(
    r"\b(?:"
    r"the (?:sender|speaker|author|writer|message|user|individual|person)|"
    r"this (?:message|response|text)|"
    r"(?:asks|asking|states|stating|says|saying|mentions|mentioning|indicates|"
    r"indicating|explains|explaining|notes|noting|requests|requesting)"
    r"\s+(?:if|whether|that|for)"
    r")\b",
    re.IGNORECASE,
)
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
    require_person_preserved: bool = True
    # An unchanged draft is reported rather than rejected: some messages are
    # already plain English, and a few such pairs teach the model restraint.
    # Callers bound how many they keep via ``max_unchanged_ratio``.
    reject_unchanged: bool = False

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
    unchanged: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "length_ratio": self.length_ratio,
            "semantic_similarity": self.semantic_similarity,
            "unchanged": self.unchanged,
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


def has_first_person(text: str) -> bool:
    """Report whether the text speaks in first person."""

    normalized = text.lower().replace("’", "")
    return any(
        token.replace("'", "") in _FIRST_PERSON for token in _WORD_RE.findall(normalized)
    )


def is_reported_speech(text: str) -> bool:
    """Report whether the text describes the message from the outside."""

    return bool(_REPORTED_SPEECH_RE.search(text))


def _comparable_text(text: str) -> str:
    """Collapse text to letters, digits, and single spaces for copy detection."""

    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


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
    if rules.require_person_preserved:
        # Dropping the sender's voice, or narrating the message from outside,
        # breaks the assumption that the neutral draft is the same utterance.
        if has_first_person(original_text) and not has_first_person(neutral_text):
            errors.append("person_changed")
        if is_reported_speech(neutral_text) and not is_reported_speech(original_text):
            errors.append("reported_speech")
    unchanged = _comparable_text(original_text) == _comparable_text(neutral_text)
    if unchanged and rules.reject_unchanged:
        errors.append("unchanged_text")

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
        unchanged=unchanged,
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


@dataclass
class _PendingUnit:
    """One record awaiting an accepted neutralization."""

    pair_id: str
    source_id: str
    split: str | None
    original: list[str]
    context: list[str]
    messages: list[dict[str, str]]
    errors: list[str]
    rejected: list[str]


def token_budget_slices(lengths: Sequence[int], budget: int) -> list[tuple[int, int]]:
    """Group consecutive items into ``[start, end)`` spans that fit a token budget.

    Padding makes every sequence in a batch as long as the longest one, so the
    cost of a span is ``len(span) * max(span)`` rather than its sum. Order is
    preserved so callers can zip results back onto their inputs.
    """

    if budget < 1:
        raise ValueError("budget must be positive")
    spans: list[tuple[int, int]] = []
    start = 0
    total = len(lengths)
    while start < total:
        longest = 0
        end = start
        while end < total:
            candidate = max(longest, lengths[end])
            if end > start and candidate * (end - start + 1) > budget:
                break
            longest = candidate
            end += 1
        spans.append((start, end))
        start = end
    return spans


def _unchanged_has_room(unchanged: int, accepted: int, ratio: float) -> bool:
    """Report whether one more unchanged pair stays within the allowed share."""

    if ratio >= 1:
        return True
    if ratio <= 0:
        return False
    # The share is meaningless before anything is accepted, so always allow one
    # through; otherwise a corpus of already-neutral text could never start.
    return (unchanged + 1) <= max(1.0, ratio * (accepted + 1))


def _accepts_attempt(function: Callable[..., Any]) -> bool:
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False
    return "attempt" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _chunked(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    chunk: list[Any] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def run_neutralization(
    records: Iterable[Mapping[str, Any]],
    output_path: str | Path,
    generator: Callable[[list[dict[str, str]]], str | Mapping[str, Any] | Sequence[str]]
    | None = None,
    *,
    batch_generator: Callable[[list[list[dict[str, str]]]], Sequence[Any]] | None = None,
    batch_size: int = 1,
    prompt_config: NeutralizationPromptConfig | None = None,
    validation_config: NeutralValidationConfig | None = None,
    semantic_scorer: Callable[[str, str], float] | None = None,
    max_attempts: int = 2,
    max_unchanged_ratio: float = 0.05,
    report_path: str | Path | None = None,
    on_progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run an injected generator, checkpointing accepted pairs as they are produced.

    Existing valid pair IDs in ``output_path`` are skipped, so an interrupted run
    can safely restart with the same source records. Supplying ``batch_generator``
    lets the caller neutralize ``batch_size`` records per model call.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if not 0 <= max_unchanged_ratio <= 1:
        raise ValueError("max_unchanged_ratio must be in [0, 1]")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if generator is None and batch_generator is None:
        raise ValueError("Provide either generator or batch_generator")

    destination = Path(output_path)
    existing = list(read_jsonl(destination)) if destination.is_file() else []
    completed: set[str] = set()
    for pair in existing:
        if pair.get("format") != NEUTRAL_PAIR_FORMAT:
            raise ValueError(f"Unsupported neutral pair format {pair.get('format')!r}")
        pair_id = str(pair.get("pair_id", ""))
        if not pair_id or pair_id in completed:
            raise ValueError("Existing neutral pairs must have unique nonempty pair IDs")
        completed.add(pair_id)

    batch_takes_attempt = batch_generator is not None and _accepts_attempt(batch_generator)

    def generate_many(prompts: list[list[dict[str, str]]], attempt: int) -> list[Any]:
        if batch_generator is not None:
            # A retry must sample differently, or it reproduces the rejected draft.
            outputs = list(
                batch_generator(prompts, attempt=attempt)
                if batch_takes_attempt
                else batch_generator(prompts)
            )
            if len(outputs) != len(prompts):
                raise ValueError("batch_generator returned the wrong number of outputs")
            return outputs
        assert generator is not None
        return [generator(prompt) for prompt in prompts]

    pair_count = len(existing)
    skipped = generated = 0
    unchanged_count = sum(
        1 for record in existing if record.get("validation", {}).get("unchanged")
    )
    failures: list[dict[str, Any]] = []

    def pending_units() -> Iterable[_PendingUnit]:
        nonlocal skipped
        for record in records:
            original = _record_bubbles(record)
            source_id = str(record.get("pair_id") or record.get("source_id") or "")
            pair_id = source_id or sha256_text(json.dumps(original, ensure_ascii=False))[:24]
            if pair_id in completed:
                skipped += 1
                continue
            context = _record_context(record)
            yield _PendingUnit(
                pair_id=pair_id,
                source_id=source_id,
                split=str(record["split"]) if "split" in record else None,
                original=original,
                context=context,
                messages=build_neutralization_messages(
                    original,
                    context_bubbles=context,
                    config=prompt_config,
                ),
                errors=[],
                rejected=[],
            )

    for chunk in _chunked(pending_units(), batch_size):
        remaining = list(chunk)
        chunk_pairs: list[dict[str, Any]] = []
        for attempt in range(max_attempts):
            if not remaining:
                break
            try:
                outputs = generate_many([unit.messages for unit in remaining], attempt)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                for unit in remaining:
                    unit.errors.append(str(error))
                continue
            retry: list[_PendingUnit] = []
            for unit, raw_output in zip(remaining, outputs, strict=True):
                try:
                    neutral = parse_neutral_output(raw_output)
                    validation = validate_neutral(
                        unit.original,
                        neutral,
                        config=validation_config,
                        semantic_scorer=semantic_scorer,
                    )
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    unit.errors.append(str(error))
                    retry.append(unit)
                    continue
                if not validation.valid:
                    unit.errors.extend(validation.errors)
                    # Keeping the rejected draft makes it possible to tell a
                    # prompt-adherence failure from an unusable input.
                    unit.rejected.append(" / ".join(neutral))
                    retry.append(unit)
                    continue
                if validation.unchanged and not _unchanged_has_room(
                    unchanged_count, pair_count + len(chunk_pairs), max_unchanged_ratio
                ):
                    # A later attempt samples more widely and often finds a real
                    # rewrite, so spend the remaining attempts before giving up.
                    unit.errors.append("unchanged_over_cap")
                    unit.rejected.append(" / ".join(neutral))
                    retry.append(unit)
                    continue
                if validation.unchanged:
                    unchanged_count += 1
                chunk_pairs.append(
                    make_neutral_pair(
                        pair_id=unit.pair_id,
                        source_id=unit.source_id or None,
                        split=unit.split,
                        original_bubbles=unit.original,
                        neutral_bubbles=neutral,
                        context_bubbles=unit.context,
                        validation=validation,
                    )
                )
                completed.add(unit.pair_id)
            remaining = retry

        failures.extend(
            {
                "pair_id": unit.pair_id,
                "errors": list(dict.fromkeys(unit.errors)) or ["generation_failed"],
                "original": list(unit.original),
                "rejected": list(unit.rejected),
            }
            for unit in remaining
        )
        if chunk_pairs:
            # Appending keeps checkpoint cost flat as the output file grows.
            extend_jsonl(destination, chunk_pairs)
            generated += len(chunk_pairs)
            pair_count += len(chunk_pairs)
        if on_progress is not None:
            on_progress(
                {
                    "pairs": pair_count,
                    "generated": generated,
                    "skipped_existing": skipped,
                    "failed": len(failures),
                }
            )

    summary = {
        "format": NEUTRAL_PAIR_FORMAT,
        "output_path": str(destination),
        "pairs": pair_count,
        "generated": generated,
        "skipped_existing": skipped,
        "failed": len(failures),
        "unchanged_pairs": unchanged_count,
        "failures": failures,
    }
    if report_path is not None:
        write_json(report_path, summary)
    return summary
