# ruff: noqa: E501

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from imessage_mlx.data.adapters import (
    classify_pair_signal,
    normalized_fingerprint,
    protected_facts,
)
from imessage_mlx.data.rewrite import STRUCTURAL_TOKEN_RE
from imessage_mlx.utils import (
    ensure_private_dir,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)

CONVERGENCE_SCHEMA_VERSION = 2
SEMANTIC_OUTPUT_SCHEMA_VERSION = "convergence-semantics-v4-slang"
VARIANT_OUTPUT_SCHEMA_VERSION = "convergence-variants-v1"
SEMANTIC_PROMPT_VERSION = "semantic-extraction-v11-widespread-calibration"
VARIANT_PROMPT_VERSION = "blind-style-generation-v6-slang"
SOURCE_VALIDATION_VERSION = "semantic-protected-facts-v4-grounded"

STYLE_KINDS = (
    "formal_professional",
    "neutral_everyday",
    "verbose_indirect",
    "terse_conversational",
)
DEFAULT_SPLIT_ALLOCATION = {"train": 400, "valid": 50, "test": 50}
SPLIT_ORDER = ("train", "valid", "test")

SEMANTIC_EXTRACTION_INSTRUCTIONS = """
Extract a canonical semantic representation of the supplied target message. Earlier conversation
turns may be supplied only to resolve what the target communicates. Do not include facts from the
context unless the target affirms, rejects, requests, or otherwise communicates them. Do not
rewrite, quote, paraphrase, or imitate the target wording. Record only meaning needed to create
semantically equivalent messages: speech act, atomic propositions, entities and protected literals,
time references, numbers, negation, modality or uncertainty, question intent, emotion, and
intensity.

Also emit resolved_paraphrase: one complete standard-English sentence (two only if unavoidable)
that expresses everything the target communicates, written from the author's own perspective as a
direct message. Use the context bundle to resolve references the evidence supports; keep genuinely
unresolved pronouns unresolved. It must carry the same speech act, question intent, negation,
numbers, modality, and emotion as the target while using noticeably different wording and sentence
structure than the target. Never copy the target text into resolved_paraphrase.

The context_bundle separates exact reply links, recent turns, locally retrieved historical evidence,
and human-approved glossary entries. Prefer approved glossary definitions and exact links, then
retrieved evidence, then chronological proximity. Historical evidence is a clue, not permission to
add unrelated facts.

For every project, coined term, person, or domain entity whose meaning you resolve beyond its literal
surface form, emit a resolved_entities item. Cite only entry_id or message_id values present in the
context bundle. Never invent evidence. Put anything that remains materially uncertain into
remaining_ambiguities and preserve that uncertainty in the propositions.

Texting fillers, abbreviations, slang, and idioms need explicit interpretation. For every expression
you read beyond its literal dictionary words, add a slang_interpretations item containing its exact
target wording, its meaning in this message, and its scope. Use scope "widespread" for slang shared
across current English texting culture, including short abbreviations such as "rn", "omw", "ts"
meaning this stuff, "sm" meaning so much, "ong", and "prolly"; idioms such as "imma keep it a stack";
and generic friendly address terms such as "brodie", "bro", and "cuz". Record the ordinary meaning
rather than the wording. Use scope "in_group" only for readings particular to the author's circle;
these are usually proper nouns or coined names for the author's projects, tools, people, places, or
running personal topics, or an address term when evidence shows it names one specific person rather
than addressing a friend generically. in_group readings must cite glossary entry_id or context
message_id evidence, and without such evidence you must not guess. Use scope "uncertain" when you
cannot commit to one reading; then copy the expression verbatim into protected_literals, record the
open question in remaining_ambiguities, and set eligible=false when that expression carries the core
meaning. Pure tone markers such as "lol" or "tho" usually contribute emotion or intensity rather
than propositions; record them there instead of inventing propositions.

Preserve unresolved pronouns and references without resolving them. Preserve grammatical perspective:
use I/me/my for the author, you/your for the addressee, we/our for the author plus others, and retain
third-person pronouns, tense, modality, and ellipsis. Do not replace people with labels such as
"speaker", "sender", "addressee", "recipient", "entity", or "unspecified actor" in propositions.

Put genuinely ambiguous coined terms, codes, product labels, and names into protected_literals exactly
as written. Never expand or guess their meaning. A protected literal is valid only when the surrounding
message still has a clear communicative act that can be paraphrased while leaving that literal in place.
Set eligible=false when ambiguity is itself central to the speech act, or when the message is only a
reaction token, emoji, laughter, acknowledgment, or context fragment with no independently restatable
proposition. A message may otherwise be context_dependent and eligible when unresolved pronouns can be
preserved naturally. Preserve target_id verbatim. Return only the requested structured output.
""".strip()

BLIND_VARIANT_INSTRUCTIONS = """
Generate four independently worded English source messages from the supplied semantic JSON.
The original message is deliberately unavailable. Treat resolved_paraphrase as the primary
statement of the complete meaning and the remaining fields as constraints on it. Express exactly
that meaning without adding facts, resolving uncertainty, changing negation, or changing question
intent. Do not copy resolved_paraphrase verbatim; each source must reword it.

Write each source as the direct message itself, from the same grammatical perspective. Convert
semantic role descriptions back into natural pronouns: the author is I/me/my, the addressee is
you/your, and the author plus others is we/our. Never output analytical or meta-language such as
"the speaker", "the sender", "the addressee", "the referent", "the entity", "unspecified actor",
"opaque term", "unresolved", "the message says", or "the question is whether". Preserve unresolved
pronouns such as it, this, that, they, and one instead of explaining that their referents are unknown.

Return exactly one source for each of these labels:
- formal_professional: complete professional wording and syntax
- neutral_everyday: ordinary standard English with different syntax or vocabulary from the semantic
  propositions wherever meaning permits
- verbose_indirect: a longer but precise formulation without new hedging or facts
- terse_conversational: a concise standard-English formulation that retains every proposition

Make the sources substantively different in wording and register, not punctuation-only variants.
Express every slang_interpretations meaning in plain standard English suited to each register; never
reproduce a slang surface form unless it also appears in protected_literals.
Copy every protected_literal exactly as supplied in every source. Do not expand, translate, quote,
define, normalize, or reinterpret protected literals. Verbose and formal variants may add connective
wording but must not add politeness, urgency, certainty, timing, or other meaning. Terse variants must
retain every proposition.
Preserve the supplied opaque target_id on the batch and every item. Return only the requested
structured output.
""".strip()

_WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)


class ResolvedEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    term: str
    interpretation: str
    evidence_ids: list[str]
    confidence: Literal["low", "medium", "high"]


class SlangInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expression: str
    interpretation: str
    scope: Literal["widespread", "in_group", "uncertain"]
    evidence_ids: list[str] = Field(default_factory=list)


class SemanticExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["convergence-semantics-v4-slang"] = SEMANTIC_OUTPUT_SCHEMA_VERSION
    target_id: str
    speech_act: str
    resolved_paraphrase: str
    atomic_propositions: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    protected_literals: list[str] = Field(default_factory=list)
    time_references: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)
    negation: bool
    modality_uncertainty: list[str] = Field(default_factory=list)
    question_intent: str | None = None
    emotion: str | None = None
    intensity: str
    resolved_entities: list[ResolvedEntity] = Field(default_factory=list)
    slang_interpretations: list[SlangInterpretation] = Field(default_factory=list)
    remaining_ambiguities: list[str] = Field(default_factory=list)
    eligible: bool
    context_dependent: bool
    ineligibility_reason: str | None = None


class GeneratedVariant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_id: str
    variant_kind: Literal[
        "formal_professional",
        "neutral_everyday",
        "verbose_indirect",
        "terse_conversational",
    ]
    source_text: str


class GeneratedVariantBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["convergence-variants-v1"] = VARIANT_OUTPUT_SCHEMA_VERSION
    target_id: str
    variants: list[GeneratedVariant]


class StructuredOutputMissingError(RuntimeError):
    pass


class TargetIdentifierMismatchError(RuntimeError):
    pass


class StyleIdentifierMismatchError(RuntimeError):
    pass


class SemanticWordingLeakageError(RuntimeError):
    pass


class SemanticFactConflictError(RuntimeError):
    pass


class ResumeCompatibilityError(RuntimeError):
    pass


SemanticValidator = Callable[
    [str, str, SemanticExtraction],
    bool | str | Iterable[str] | tuple[bool, str | None],
]


def deterministic_pair_id(target_id: str, variant_kind: str) -> str:
    if variant_kind not in STYLE_KINDS:
        raise ValueError(f"Unknown convergence variant kind {variant_kind!r}")
    return sha256_text(f"convergence-pair:{CONVERGENCE_SCHEMA_VERSION}:{target_id}:{variant_kind}")[
        :24
    ]


def _target_id(split: str, source_pair_id: str) -> str:
    return sha256_text(f"convergence-target:{CONVERGENCE_SCHEMA_VERSION}:{split}:{source_pair_id}")[
        :24
    ]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _usage(response: Any) -> dict[str, int]:
    value = getattr(response, "usage", None)
    return {
        "input_tokens": int(getattr(value, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(value, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(value, "total_tokens", 0) or 0),
    }


def _parsed_model(response: Any, expected_type: type[BaseModel]) -> BaseModel:
    for output in getattr(response, "output", ()):
        if getattr(output, "type", None) != "message":
            continue
        for content in getattr(output, "content", ()):
            parsed = getattr(content, "parsed", None)
            if isinstance(parsed, expected_type):
                return parsed
    raise StructuredOutputMissingError(
        "OpenAI response did not contain the requested structured output"
    )


def _context_evidence_ids(context_bundle: Mapping[str, Any] | None) -> set[str]:
    if not context_bundle:
        return set()
    values: set[str] = set()
    for field in ("exact_links", "recent_turns", "retrieved_evidence"):
        records = context_bundle.get(field, [])
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, Mapping):
                continue
            message_id = record.get("message_id")
            if isinstance(message_id, str):
                values.add(message_id)
            message_ids = record.get("message_ids", [])
            if isinstance(message_ids, list):
                values.update(str(value) for value in message_ids)
    glossary = context_bundle.get("glossary_entries", [])
    if isinstance(glossary, list):
        for entry in glossary:
            if not isinstance(entry, Mapping):
                continue
            entry_id = entry.get("entry_id")
            if isinstance(entry_id, str):
                values.add(entry_id)
            evidence_ids = entry.get("evidence_message_ids", [])
            if isinstance(evidence_ids, list):
                values.update(str(value) for value in evidence_ids)
    return values


def validate_semantic_extraction(
    semantic: SemanticExtraction,
    *,
    expected_target_id: str,
    styled_target: str | None = None,
    context_bundle: Mapping[str, Any] | None = None,
) -> SemanticExtraction:
    if semantic.target_id != expected_target_id:
        raise TargetIdentifierMismatchError("Stage A returned an unexpected target identifier")
    if semantic.schema_version != SEMANTIC_OUTPUT_SCHEMA_VERSION:
        raise ResumeCompatibilityError("Stage A returned an incompatible semantic schema version")
    if not semantic.resolved_paraphrase.strip():
        raise StructuredOutputMissingError("Stage A omitted the resolved paraphrase")
    if styled_target is not None:
        target_facts = protected_facts(styled_target)
        semantic_numbers = tuple(str(value) for value in semantic.numbers)
        if semantic_numbers != target_facts["numbers"]:
            raise SemanticFactConflictError("Stage A changed explicit target numbers")
        if target_facts["negated"] and not semantic.negation:
            raise SemanticFactConflictError("Stage A omitted explicit target negation")
    allowed_evidence = _context_evidence_ids(context_bundle)
    for entity in semantic.resolved_entities:
        if not entity.evidence_ids:
            raise SemanticFactConflictError("Resolved entity omitted grounding evidence")
        if any(evidence_id not in allowed_evidence for evidence_id in entity.evidence_ids):
            raise SemanticFactConflictError("Resolved entity cited unknown grounding evidence")
    protected_casefold = {literal.strip().casefold() for literal in semantic.protected_literals}
    target_casefold = styled_target.casefold() if styled_target is not None else None
    for slang in semantic.slang_interpretations:
        expression = slang.expression.strip()
        if not expression or not slang.interpretation.strip():
            raise SemanticFactConflictError("Slang interpretation is missing wording or meaning")
        if target_casefold is not None and expression.casefold() not in target_casefold:
            raise SemanticFactConflictError(
                "Slang interpretation references wording absent from the target"
            )
        if slang.scope == "in_group":
            if not slang.evidence_ids:
                raise SemanticFactConflictError("In-group slang requires grounding evidence")
            if any(evidence_id not in allowed_evidence for evidence_id in slang.evidence_ids):
                raise SemanticFactConflictError("In-group slang cited unknown grounding evidence")
        if slang.scope == "uncertain" and expression.casefold() not in protected_casefold:
            raise SemanticFactConflictError(
                "Uncertain slang must be preserved verbatim as a protected literal"
            )
    return semantic


def semantic_repeats_original_wording(
    semantic: SemanticExtraction,
    styled_target: str,
) -> bool:
    original = styled_target.strip().casefold()
    if not original:
        return False
    if _near_duplicate(styled_target, semantic.resolved_paraphrase):
        return True
    narrative_values: list[str] = [
        semantic.speech_act,
        semantic.resolved_paraphrase,
        *semantic.atomic_propositions,
        semantic.question_intent or "",
        semantic.emotion or "",
        semantic.intensity,
        semantic.ineligibility_reason or "",
        *(entity.interpretation for entity in semantic.resolved_entities),
        *(slang.interpretation for slang in semantic.slang_interpretations),
        *semantic.remaining_ambiguities,
    ]
    return any(original in value.strip().casefold() for value in narrative_values if value.strip())


def validate_generated_variant_batch(
    batch: GeneratedVariantBatch,
    *,
    expected_target_id: str,
) -> GeneratedVariantBatch:
    if batch.target_id != expected_target_id:
        raise TargetIdentifierMismatchError("Stage B returned an unexpected target identifier")
    if batch.schema_version != VARIANT_OUTPUT_SCHEMA_VERSION:
        raise ResumeCompatibilityError("Stage B returned an incompatible variant schema version")

    returned: dict[str, GeneratedVariant] = {}
    for variant in batch.variants:
        if variant.target_id != expected_target_id:
            raise TargetIdentifierMismatchError(
                "Stage B returned a variant with an unexpected target identifier"
            )
        if variant.variant_kind in returned:
            raise StyleIdentifierMismatchError("Stage B returned a duplicate style identifier")
        returned[variant.variant_kind] = variant
    if set(returned) != set(STYLE_KINDS):
        raise StyleIdentifierMismatchError("Stage B must return exactly the configured style set")
    return batch


async def extract_target_semantics(
    client: AsyncOpenAI,
    *,
    model: str,
    target_id: str,
    styled_target: str,
    context_bundle: Mapping[str, Any] | None = None,
    context: list[dict[str, Any]] | None = None,
) -> tuple[SemanticExtraction, dict[str, int]]:
    if context_bundle is None and context is not None:
        context_bundle = {
            "schema_version": 1,
            "exact_links": [
                turn for turn in context if turn.get("relation") in {"reply_to", "thread_root"}
            ],
            "recent_turns": [
                turn for turn in context if turn.get("relation") not in {"reply_to", "thread_root"}
            ],
            "retrieved_evidence": [],
            "glossary_entries": [],
        }
    payload = {
        "target_id": target_id,
        "context_bundle": dict(context_bundle or {}),
        "styled_target": styled_target,
    }
    response = await client.responses.parse(
        model=model,
        instructions=SEMANTIC_EXTRACTION_INSTRUCTIONS,
        input=json.dumps(payload, sort_keys=True, ensure_ascii=False),
        text_format=SemanticExtraction,
        store=False,
    )
    parsed = _parsed_model(response, SemanticExtraction)
    assert isinstance(parsed, SemanticExtraction)
    return (
        validate_semantic_extraction(
            parsed,
            expected_target_id=target_id,
            styled_target=styled_target,
            context_bundle=context_bundle,
        ),
        _usage(response),
    )


async def generate_blind_variants(
    client: AsyncOpenAI,
    *,
    model: str,
    target_id: str,
    semantic: SemanticExtraction,
) -> tuple[GeneratedVariantBatch, dict[str, int]]:
    if semantic.target_id != target_id:
        raise TargetIdentifierMismatchError(
            "Stage B semantic input has the wrong target identifier"
        )
    payload = {
        "target_id": target_id,
        "semantic": semantic.model_dump(mode="json"),
    }
    response = await client.responses.parse(
        model=model,
        instructions=BLIND_VARIANT_INSTRUCTIONS,
        input=json.dumps(payload, sort_keys=True, ensure_ascii=False),
        text_format=GeneratedVariantBatch,
        store=False,
    )
    parsed = _parsed_model(response, GeneratedVariantBatch)
    assert isinstance(parsed, GeneratedVariantBatch)
    return validate_generated_variant_batch(parsed, expected_target_id=target_id), _usage(response)


def _allocation(value: Mapping[str, int] | None) -> dict[str, int]:
    allocation = dict(DEFAULT_SPLIT_ALLOCATION if value is None else value)
    if set(allocation) != set(SPLIT_ORDER):
        raise ValueError("Convergence allocation must contain exactly train, valid, and test")
    for split, count in allocation.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"Convergence allocation for {split!r} must be nonnegative")
    return {split: allocation[split] for split in SPLIT_ORDER}


def _length_band(text: str) -> str:
    length = len(text)
    if length <= 40:
        return "short"
    if length <= 100:
        return "medium"
    if length <= 200:
        return "long"
    return "very_long"


def _load_split_candidates(
    path: Path,
    split: str,
    *,
    max_characters: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidates: list[dict[str, Any]] = []
    counts = Counter()
    seen_pair_ids: set[str] = set()
    for chronology_index, record in enumerate(read_jsonl(path)):
        counts["input_records"] += 1
        pair_id = record.get("pair_id")
        source = record.get("source")
        target = record.get("target", record.get("styled_text"))
        context = record.get("context", [])
        context_bundle = record.get("context_bundle")
        if context_bundle is None:
            context_bundle = {
                "schema_version": 1,
                "exact_links": [
                    turn
                    for turn in context
                    if isinstance(turn, dict)
                    and turn.get("relation") in {"reply_to", "thread_root"}
                ],
                "recent_turns": [
                    turn
                    for turn in context
                    if isinstance(turn, dict)
                    and turn.get("relation") not in {"reply_to", "thread_root"}
                ],
                "retrieved_evidence": [],
                "glossary_entries": [],
                "retriever": {"kind": "legacy_recent_context"},
            }
        bundle_lists_valid = isinstance(context_bundle, dict) and all(
            isinstance(context_bundle.get(field, []), list)
            for field in (
                "exact_links",
                "recent_turns",
                "retrieved_evidence",
                "glossary_entries",
            )
        )
        temporal_context_valid = True
        if isinstance(context_bundle, dict) and context_bundle.get("schema_version") == 2:
            target_timestamp = record.get("timestamp_ns")
            temporal_context_valid = isinstance(target_timestamp, int)
            if temporal_context_valid:
                for field in ("exact_links", "recent_turns", "retrieved_evidence"):
                    for evidence in context_bundle.get(field, []):
                        evidence_timestamp = (
                            evidence.get("timestamp_ns") if isinstance(evidence, dict) else None
                        )
                        if (
                            isinstance(evidence_timestamp, int)
                            and evidence_timestamp >= target_timestamp
                        ):
                            temporal_context_valid = False
                for entry in context_bundle.get("glossary_entries", []):
                    valid_from = (
                        entry.get("valid_from_timestamp_ns") if isinstance(entry, dict) else None
                    )
                    if isinstance(valid_from, int) and valid_from >= target_timestamp:
                        temporal_context_valid = False
        if (
            not isinstance(pair_id, str)
            or not pair_id.strip()
            or not isinstance(target, str)
            or not target.strip()
            or not isinstance(context, list)
            or any(
                not isinstance(turn, dict)
                or turn.get("role") not in {"me", "other"}
                or not isinstance(turn.get("text"), str)
                or not str(turn["text"]).strip()
                or turn.get("relation")
                not in {None, "reply_to", "thread_root", "historical_retrieval"}
                for turn in context
            )
            or not bundle_lists_valid
            or not temporal_context_valid
        ):
            counts["excluded_invalid"] += 1
            continue
        pair_id = pair_id.strip()
        target = target.strip()
        source = source.strip() if isinstance(source, str) else ""
        if pair_id in seen_pair_ids:
            raise ValueError(f"Duplicate accepted BART pair_id {pair_id!r} in {split}")
        seen_pair_ids.add(pair_id)
        if len(target) > max_characters:
            counts["excluded_too_long"] += 1
            continue
        if STRUCTURAL_TOKEN_RE.search(target):
            counts["excluded_structural_token"] += 1
            continue
        words = _WORD_RE.findall(target)
        if len(target) < 10 or len(words) < 2:
            counts["excluded_low_content"] += 1
            continue
        signal = classify_pair_signal(source, target) if source else "unpaired"
        if context_bundle["glossary_entries"]:
            grounding = "glossary"
        elif context_bundle["retrieved_evidence"]:
            grounding = "historical"
        elif context_bundle["exact_links"]:
            grounding = "reply"
        else:
            grounding = "recent_or_none"
        candidates.append(
            {
                "target_id": _target_id(split, pair_id),
                "source_pair_id": pair_id,
                "split": split,
                "target_text": target,
                "context": context,
                "context_bundle": context_bundle,
                "chronology_index": chronology_index,
                "selection_stratum": "|".join(
                    (
                        signal,
                        "question" if "?" in target else "statement",
                        _length_band(target),
                        grounding,
                    )
                ),
            }
        )
        counts["eligible_records"] += 1
    return candidates, dict(counts)


def _evenly_spaced_indices(length: int, count: int) -> list[int]:
    if count <= 0:
        return []
    if count >= length:
        return list(range(length))
    return [min(length - 1, math.floor((index + 0.5) * length / count)) for index in range(count)]


def _stratified_sample(
    records: list[dict[str, Any]],
    count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if count < 0 or count > len(records):
        raise ValueError("Stratified sample size exceeds the available records")
    if count == 0:
        return [], list(records)

    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_stratum[str(record["selection_stratum"])].append(record)

    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for stratum, values in by_stratum.items():
        exact = count * len(values) / len(records)
        quotas[stratum] = math.floor(exact)
        remainders.append((exact - quotas[stratum], stratum))
    unallocated = count - sum(quotas.values())
    for _remainder, stratum in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if unallocated == 0:
            break
        if quotas[stratum] < len(by_stratum[stratum]):
            quotas[stratum] += 1
            unallocated -= 1

    selected_ids: set[str] = set()
    for stratum in sorted(by_stratum):
        values = by_stratum[stratum]
        for index in _evenly_spaced_indices(len(values), quotas[stratum]):
            selected_ids.add(str(values[index]["target_id"]))

    selected = [record for record in records if str(record["target_id"]) in selected_ids]
    remainder = [record for record in records if str(record["target_id"]) not in selected_ids]
    if len(selected) != count:
        raise RuntimeError("Deterministic stratified selection produced the wrong sample size")
    return selected, remainder


def select_convergence_targets(
    splits_dir: str | Path,
    *,
    allocation: Mapping[str, int] | None = None,
    max_characters: int = 512,
) -> list[dict[str, Any]]:
    if max_characters <= 0:
        raise ValueError("Maximum target length must be positive")
    requested = _allocation(allocation)
    root = Path(splits_dir)
    selected: list[dict[str, Any]] = []
    for split in SPLIT_ORDER:
        candidates, _counts = _load_split_candidates(
            root / f"{split}.jsonl",
            split,
            max_characters=max_characters,
        )
        if len(candidates) < requested[split]:
            raise ValueError(
                f"Accepted BART {split} split has only {len(candidates)} eligible targets; "
                f"{requested[split]} requested"
            )
        split_selected, _remainder = _stratified_sample(candidates, requested[split])
        selected.extend(split_selected)
    return selected


def _reserve_counts(
    allocation: Mapping[str, int],
    value: int | Mapping[str, int] | None,
) -> dict[str, int]:
    if value is None:
        return {
            split: (max(20, math.ceil(count * 0.50)) if count else 0)
            for split, count in allocation.items()
        }
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0:
            raise ValueError("Reserve target count must be nonnegative")
        return {split: (value if allocation[split] else 0) for split in SPLIT_ORDER}
    if not isinstance(value, Mapping) or set(value) != set(SPLIT_ORDER):
        raise ValueError("Reserve allocation must contain exactly train, valid, and test")
    result = {split: value[split] for split in SPLIT_ORDER}
    invalid = (
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in result.values()
    )
    if any(invalid):
        raise ValueError("Reserve target counts must be nonnegative integers")
    return result


def create_convergence_manifest(
    splits_dir: str | Path,
    *,
    model: str,
    allocation: Mapping[str, int] | None = None,
    max_characters: int = 512,
    reserve_per_split: int | Mapping[str, int] | None = None,
) -> dict[str, Any]:
    if not model.strip():
        raise ValueError("OpenAI model identifier is required")
    if max_characters <= 0:
        raise ValueError("Maximum target length must be positive")
    requested = _allocation(allocation)
    reserves = _reserve_counts(requested, reserve_per_split)
    root = Path(splits_dir)
    input_hashes = {split: sha256_file(root / f"{split}.jsonl") for split in SPLIT_ORDER}
    input_fingerprint = sha256_text(_canonical_json(input_hashes))

    targets: list[dict[str, Any]] = []
    selection_counts: dict[str, dict[str, int]] = {}
    for split in SPLIT_ORDER:
        candidates, counts = _load_split_candidates(
            root / f"{split}.jsonl",
            split,
            max_characters=max_characters,
        )
        count = requested[split]
        if len(candidates) < count:
            raise ValueError(
                f"Accepted BART {split} split has only {len(candidates)} eligible targets; "
                f"{count} requested"
            )
        primary, remaining = _stratified_sample(candidates, count)
        reserve_count = min(reserves[split], len(remaining))
        reserve, _unused = _stratified_sample(remaining, reserve_count)
        pool = primary + reserve
        primary_ids = {str(record["target_id"]) for record in primary}
        for rank, record in enumerate(pool):
            targets.append(
                {
                    **record,
                    "selection_rank": rank,
                    "primary": str(record["target_id"]) in primary_ids,
                }
            )
        selection_counts[split] = {
            **counts,
            "requested_primary": count,
            "selected_primary": len(primary),
            "selected_reserve": len(reserve),
        }

    selection_fingerprint = sha256_text(_canonical_json(targets))
    schema_hashes = {
        "semantic": sha256_text(_canonical_json(SemanticExtraction.model_json_schema())),
        "variant": sha256_text(_canonical_json(GeneratedVariantBatch.model_json_schema())),
    }
    descriptor = {
        "artifact_schema_version": CONVERGENCE_SCHEMA_VERSION,
        "model": model,
        "prompt_versions": {
            "stage_a": SEMANTIC_PROMPT_VERSION,
            "stage_b": VARIANT_PROMPT_VERSION,
        },
        "source_validation_version": SOURCE_VALIDATION_VERSION,
        "prompt_hashes": {
            "stage_a": sha256_text(SEMANTIC_EXTRACTION_INSTRUCTIONS),
            "stage_b": sha256_text(BLIND_VARIANT_INSTRUCTIONS),
        },
        "output_schema_versions": {
            "stage_a": SEMANTIC_OUTPUT_SCHEMA_VERSION,
            "stage_b": VARIANT_OUTPUT_SCHEMA_VERSION,
        },
        "schema_hashes": schema_hashes,
        "styles": list(STYLE_KINDS),
        "allocation": requested,
        "reserve_allocation": reserves,
        "maximum_characters": max_characters,
        "input_hashes": input_hashes,
        "input_fingerprint": input_fingerprint,
        "selection_fingerprint": selection_fingerprint,
    }
    generation_fingerprint = sha256_text(_canonical_json(descriptor))
    return {
        "kind": "openai_convergence_generation_manifest",
        **descriptor,
        "generation_fingerprint": generation_fingerprint,
        "selection_counts": selection_counts,
        "targets": targets,
    }


def _write_immutable_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ResumeCompatibilityError("Existing convergence manifest is unreadable") from error
        if not isinstance(existing, dict):
            raise ResumeCompatibilityError("Existing convergence manifest is not an object")
        if existing.get("generation_fingerprint") != manifest["generation_fingerprint"]:
            raise ResumeCompatibilityError(
                "Existing convergence manifest has an incompatible generation fingerprint"
            )
        if existing != manifest:
            raise ResumeCompatibilityError(
                "Existing convergence manifest metadata or selected inputs are incompatible"
            )
        path.chmod(0o600)
        return
    write_json(path, manifest)


def _stage_metadata(
    manifest: Mapping[str, Any],
    stage: Literal["stage_a", "stage_b"],
) -> dict[str, Any]:
    return {
        "artifact_schema_version": CONVERGENCE_SCHEMA_VERSION,
        "generation_fingerprint": manifest["generation_fingerprint"],
        "input_fingerprint": manifest["input_fingerprint"],
        "model": manifest["model"],
        "prompt_version": manifest["prompt_versions"][stage],
        "prompt_hash": manifest["prompt_hashes"][stage],
        "output_schema_version": manifest["output_schema_versions"][stage],
        "schema_hash": manifest["schema_hashes"]["semantic" if stage == "stage_a" else "variant"],
    }


def _validate_resume_metadata(
    record: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    artifact_name: str,
) -> None:
    for field, expected_value in expected.items():
        if record.get(field) != expected_value:
            raise ResumeCompatibilityError(
                f"{artifact_name} resume has incompatible {field.replace('_', ' ')}"
            )


def _append_private_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    ensure_private_dir(path.parent)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o600)
    return count


def _ensure_private_jsonl(path: Path) -> None:
    if not path.exists():
        write_jsonl(path, [])
    else:
        path.chmod(0o600)


def _candidate_index(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    targets = manifest.get("targets")
    if not isinstance(targets, list):
        raise ResumeCompatibilityError("Convergence manifest targets are invalid")
    result: dict[str, dict[str, Any]] = {}
    for value in targets:
        if not isinstance(value, dict):
            raise ResumeCompatibilityError("Convergence manifest target is invalid")
        target_id = str(value.get("target_id", ""))
        if not target_id or target_id in result:
            raise ResumeCompatibilityError("Convergence manifest has duplicate or empty target IDs")
        result[target_id] = value
    return result


def _load_semantics(
    path: Path,
    manifest: Mapping[str, Any],
    candidates: Mapping[str, Mapping[str, Any]],
) -> dict[str, SemanticExtraction]:
    if not path.exists():
        return {}
    expected = _stage_metadata(manifest, "stage_a")
    loaded: dict[str, SemanticExtraction] = {}
    for record in read_jsonl(path):
        _validate_resume_metadata(record, expected, artifact_name="Semantic")
        target_id = str(record.get("target_id", ""))
        if target_id not in candidates:
            raise ResumeCompatibilityError("Semantic resume references an unknown target")
        if record.get("split") != candidates[target_id]["split"]:
            raise ResumeCompatibilityError("Semantic resume has an incompatible split")
        if target_id in loaded:
            raise ResumeCompatibilityError("Semantic resume has a duplicate target")
        semantic = SemanticExtraction.model_validate(record.get("semantic"))
        loaded[target_id] = validate_semantic_extraction(
            semantic,
            expected_target_id=target_id,
            styled_target=str(candidates[target_id]["target_text"]),
            context_bundle=candidates[target_id].get("context_bundle"),
        )
    return loaded


def _load_variants(
    path: Path,
    manifest: Mapping[str, Any],
    candidates: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], Counter[tuple[str, str]], Counter[str]]:
    accepted: dict[tuple[str, str], dict[str, Any]] = {}
    attempts: Counter[tuple[str, str]] = Counter()
    rejections: Counter[str] = Counter()
    if not path.exists():
        return accepted, attempts, rejections
    expected = _stage_metadata(manifest, "stage_b")
    for record in read_jsonl(path):
        _validate_resume_metadata(record, expected, artifact_name="Variant")
        target_id = str(record.get("target_id", ""))
        variant_kind = str(record.get("variant_kind", ""))
        if target_id not in candidates or variant_kind not in STYLE_KINDS:
            raise ResumeCompatibilityError("Variant resume references an unknown target or style")
        if record.get("split") != candidates[target_id]["split"]:
            raise ResumeCompatibilityError("Variant resume has an incompatible split")
        if record.get("pair_id") != deterministic_pair_id(target_id, variant_kind):
            raise ResumeCompatibilityError("Variant resume has an incompatible pair identifier")
        source = record.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ResumeCompatibilityError("Variant resume contains empty generated text")
        key = (target_id, variant_kind)
        attempts[key] += 1
        reasons = record.get("rejection_reasons", [])
        if not isinstance(reasons, list) or not all(isinstance(reason, str) for reason in reasons):
            raise ResumeCompatibilityError("Variant resume rejection metadata is invalid")
        rejections.update(reasons)
        if record.get("accepted") is True:
            if key in accepted:
                raise ResumeCompatibilityError(
                    "Variant resume has duplicate accepted target/style IDs"
                )
            accepted[key] = record
        elif record.get("accepted") is not False:
            raise ResumeCompatibilityError("Variant resume acceptance metadata is invalid")
    return accepted, attempts, rejections


def _load_usage(path: Path, manifest: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    totals = {
        "stage_a": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "stage_b": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }
    if not path.exists():
        return totals
    for record in read_jsonl(path):
        stage = record.get("stage")
        if stage not in totals:
            raise ResumeCompatibilityError("Usage resume has an invalid stage")
        _validate_resume_metadata(
            record,
            _stage_metadata(manifest, stage),
            artifact_name="Usage",
        )
        usage = record.get("usage")
        if not isinstance(usage, dict):
            raise ResumeCompatibilityError("Usage resume token metadata is invalid")
        for name in totals[stage]:
            value = usage.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ResumeCompatibilityError("Usage resume token count is invalid")
            totals[stage][name] += value
    return totals


def _near_duplicate(first: str, second: str) -> str | None:
    left = normalized_fingerprint(first)
    right = normalized_fingerprint(second)
    if not left or not right:
        return None
    if left == right:
        return "normalized_duplicate"
    if min(len(left), len(right)) >= 12 and SequenceMatcher(None, left, right).ratio() >= 0.96:
        return "fuzzy_duplicate"
    return None


def _semantic_validator_reasons(
    validator: SemanticValidator,
    target_text: str,
    source_text: str,
    semantic: SemanticExtraction,
) -> list[str]:
    result = validator(target_text, source_text, semantic)
    if result is True or result is None:
        return []
    if result is False:
        return ["semantic_evaluator"]
    if isinstance(result, str):
        return [result]
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], bool):
        passed, reason = result
        return [] if passed else [reason or "semantic_evaluator"]
    return [str(reason) for reason in result]


def validate_generated_source(
    *,
    variant_kind: str,
    target_text: str,
    source_text: str,
    sibling_sources: Iterable[str],
    max_characters: int,
    semantic: SemanticExtraction,
    semantic_validator: SemanticValidator | None = None,
) -> list[str]:
    candidate = source_text.strip()
    reasons: list[str] = []
    if not candidate:
        reasons.append("empty")
        return reasons
    if len(candidate) > max_characters:
        reasons.append("too_long")
    if STRUCTURAL_TOKEN_RE.search(candidate):
        reasons.append("structural_token")
    candidate_facts = protected_facts(candidate)
    semantic_numbers = tuple(str(value) for value in semantic.numbers)
    if candidate_facts["numbers"] != semantic_numbers:
        reasons.append("protected_fact_conflict")
    missing_literals = [
        literal
        for literal in semantic.protected_literals
        if re.search(
            rf"(?<!\w){re.escape(literal)}(?!\w)",
            candidate,
            flags=re.IGNORECASE,
        )
        is None
    ]
    if missing_literals:
        reasons.append("protected_literal_conflict")

    target_match = _near_duplicate(target_text, candidate)
    if target_match:
        reasons.append(f"target_{target_match}")
    else:
        target_tokens = set(_WORD_RE.findall(target_text.casefold()))
        candidate_tokens = set(_WORD_RE.findall(candidate.casefold()))
        union = target_tokens | candidate_tokens
        if len(target_tokens) >= 4 and union:
            lexical_overlap = len(target_tokens & candidate_tokens) / len(union)
            if lexical_overlap >= 0.90:
                reasons.append("insufficient_target_lexical_diversity")

    for sibling in sibling_sources:
        match = _near_duplicate(candidate, sibling)
        if match:
            reasons.append(f"sibling_{match}")
            break
    if semantic_validator is not None:
        reasons.extend(
            _semantic_validator_reasons(
                semantic_validator,
                target_text,
                candidate,
                semantic,
            )
        )
    return sorted(set(reasons))


def _usage_record(
    manifest: Mapping[str, Any],
    *,
    stage: Literal["stage_a", "stage_b"],
    target_id: str,
    usage: Mapping[str, int],
) -> dict[str, Any]:
    return {
        **_stage_metadata(manifest, stage),
        "record_type": "usage",
        "stage": stage,
        "target_id": target_id,
        "usage": dict(usage),
    }


def _add_usage(total: dict[str, int], value: Mapping[str, int]) -> None:
    for name in total:
        total[name] += int(value[name])


def _active_targets(
    manifest: Mapping[str, Any],
    semantics: Mapping[str, SemanticExtraction],
    *,
    excluded_target_ids: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    allocation = manifest["allocation"]
    excluded = excluded_target_ids or set()
    active: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLIT_ORDER}
    for split in SPLIT_ORDER:
        pool = sorted(
            (target for target in manifest["targets"] if target["split"] == split),
            key=lambda target: int(target["selection_rank"]),
        )
        for target in pool:
            target_id = str(target["target_id"])
            if target_id in excluded:
                continue
            semantic = semantics.get(target_id)
            if (
                semantic is None
                or not semantic.eligible
                or (semantic.context_dependent and not target.get("context"))
            ):
                continue
            active[split].append(target)
            if len(active[split]) == int(allocation[split]):
                break
    return active


async def generate_openai_convergence_data(
    splits_dir: str | Path,
    output_dir: str | Path,
    report_path: str | Path,
    *,
    api_key: str,
    model: str,
    allocation: Mapping[str, int] | None = None,
    reserve_per_split: int | Mapping[str, int] | None = None,
    max_characters: int = 512,
    concurrency: int = 4,
    max_variant_attempts: int = 2,
    semantic_validator: SemanticValidator | None = None,
) -> dict[str, Any]:
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required")
    if concurrency <= 0 or max_variant_attempts <= 0:
        raise ValueError("Concurrency and variant attempts must be positive")

    destination = ensure_private_dir(output_dir)
    manifest_path = destination / "manifest.json"
    semantics_path = destination / "semantics.jsonl"
    variants_path = destination / "variants.jsonl"
    usage_path = destination / "usage.jsonl"
    published_path = destination / "published.jsonl"
    evaluator_path = destination / "semantic-evaluation-inputs.jsonl"

    manifest = create_convergence_manifest(
        splits_dir,
        model=model,
        allocation=allocation,
        max_characters=max_characters,
        reserve_per_split=reserve_per_split,
    )
    _write_immutable_manifest(manifest_path, manifest)
    candidates = _candidate_index(manifest)
    semantics = _load_semantics(semantics_path, manifest, candidates)
    accepted, variant_attempt_counts, cumulative_rejections = _load_variants(
        variants_path,
        manifest,
        candidates,
    )
    cumulative_usage = _load_usage(usage_path, manifest)
    for path in (semantics_path, variants_path, usage_path):
        _ensure_private_jsonl(path)

    run_usage = {
        "stage_a": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "stage_b": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }
    errors: Counter[str] = Counter()
    run_rejections: Counter[str] = Counter()
    generated_semantics = 0
    generated_variant_attempts = 0
    accepted_variants = 0
    semaphore = asyncio.Semaphore(concurrency)
    client = AsyncOpenAI(api_key=api_key, timeout=120.0, max_retries=5)

    async def request_semantic(
        target: Mapping[str, Any],
    ) -> tuple[SemanticExtraction, dict[str, int]]:
        async with semaphore:
            return await extract_target_semantics(
                client,
                model=model,
                target_id=str(target["target_id"]),
                styled_target=str(target["target_text"]),
                context_bundle=target.get("context_bundle"),
            )

    async def request_variants(
        target: Mapping[str, Any],
        semantic: SemanticExtraction,
    ) -> tuple[GeneratedVariantBatch, dict[str, int]]:
        async with semaphore:
            return await generate_blind_variants(
                client,
                model=model,
                target_id=str(target["target_id"]),
                semantic=semantic,
            )

    attempted_semantics: set[str] = set()

    async def run_semantic_wave(targets: list[dict[str, Any]]) -> None:
        nonlocal generated_semantics
        pending = [
            target
            for target in targets
            if str(target["target_id"]) not in semantics
            and str(target["target_id"]) not in attempted_semantics
        ]
        if not pending:
            return
        attempted_semantics.update(str(target["target_id"]) for target in pending)
        results = await asyncio.gather(
            *(request_semantic(target) for target in pending),
            return_exceptions=True,
        )
        semantic_records: list[dict[str, Any]] = []
        usage_records: list[dict[str, Any]] = []
        for target, result in zip(pending, results, strict=True):
            if isinstance(result, BaseException):
                errors[f"stage_a:{type(result).__name__}"] += 1
                continue
            semantic, usage = result
            target_id = str(target["target_id"])
            usage_records.append(
                _usage_record(
                    manifest,
                    stage="stage_a",
                    target_id=target_id,
                    usage=usage,
                )
            )
            _add_usage(run_usage["stage_a"], usage)
            _add_usage(cumulative_usage["stage_a"], usage)
            if semantic_repeats_original_wording(semantic, str(target["target_text"])):
                errors[f"stage_a:{SemanticWordingLeakageError.__name__}"] += 1
                continue
            semantics[target_id] = semantic
            semantic_records.append(
                {
                    **_stage_metadata(manifest, "stage_a"),
                    "record_type": "semantic",
                    "target_id": target_id,
                    "split": target["split"],
                    "semantic": semantic.model_dump(mode="json"),
                }
            )
            generated_semantics += 1
        _append_private_jsonl(semantics_path, semantic_records)
        _append_private_jsonl(usage_path, usage_records)

    async def run_variant_attempts(targets: list[dict[str, Any]]) -> None:
        nonlocal generated_variant_attempts, accepted_variants
        for _round in range(max_variant_attempts):
            pending_targets = [
                target
                for target in targets
                if any((str(target["target_id"]), style) not in accepted for style in STYLE_KINDS)
            ]
            if not pending_targets:
                break
            results = await asyncio.gather(
                *(
                    request_variants(
                        target,
                        semantics[str(target["target_id"])],
                    )
                    for target in pending_targets
                ),
                return_exceptions=True,
            )
            variant_records: list[dict[str, Any]] = []
            usage_records: list[dict[str, Any]] = []
            for target, result in zip(pending_targets, results, strict=True):
                target_id = str(target["target_id"])
                if isinstance(result, BaseException):
                    errors[f"stage_b:{type(result).__name__}"] += 1
                    continue
                batch, usage = result
                usage_records.append(
                    _usage_record(
                        manifest,
                        stage="stage_b",
                        target_id=target_id,
                        usage=usage,
                    )
                )
                _add_usage(run_usage["stage_b"], usage)
                _add_usage(cumulative_usage["stage_b"], usage)
                by_style = {variant.variant_kind: variant for variant in batch.variants}
                missing_styles = [
                    style for style in STYLE_KINDS if (target_id, style) not in accepted
                ]
                new_sources = {
                    style: by_style[style].source_text.strip() for style in missing_styles
                }
                existing_sources = [
                    str(record["source"])
                    for (existing_target_id, _style), record in accepted.items()
                    if existing_target_id == target_id
                ]
                for style in missing_styles:
                    key = (target_id, style)
                    source = new_sources[style]
                    sibling_sources = existing_sources + [
                        sibling_source
                        for sibling_style, sibling_source in new_sources.items()
                        if sibling_style != style
                    ]
                    reasons = validate_generated_source(
                        variant_kind=style,
                        target_text=str(target["target_text"]),
                        source_text=source,
                        sibling_sources=sibling_sources,
                        max_characters=max_characters,
                        semantic=semantics[target_id],
                        semantic_validator=semantic_validator,
                    )
                    variant_attempt_counts[key] += 1
                    record = {
                        **_stage_metadata(manifest, "stage_b"),
                        "record_type": "variant_attempt",
                        "target_id": target_id,
                        "split": target["split"],
                        "variant_kind": style,
                        "pair_id": deterministic_pair_id(target_id, style),
                        "attempt": variant_attempt_counts[key],
                        "source": source,
                        "accepted": not reasons,
                        "rejection_reasons": reasons,
                    }
                    variant_records.append(record)
                    generated_variant_attempts += 1
                    if reasons:
                        run_rejections.update(reasons)
                        cumulative_rejections.update(reasons)
                    else:
                        accepted[key] = record
                        accepted_variants += 1
            _append_private_jsonl(variants_path, variant_records)
            _append_private_jsonl(usage_path, usage_records)

    excluded_incomplete: set[str] = set()
    try:
        primary_targets = [target for target in manifest["targets"] if target["primary"]]
        await run_semantic_wave(primary_targets)

        while True:
            while True:
                active = _active_targets(
                    manifest,
                    semantics,
                    excluded_target_ids=excluded_incomplete,
                )
                deficits = {
                    split: int(manifest["allocation"][split]) - len(active[split])
                    for split in SPLIT_ORDER
                }
                if not any(deficits.values()):
                    break
                wave: list[dict[str, Any]] = []
                for split in SPLIT_ORDER:
                    if deficits[split] <= 0:
                        continue
                    unknown = [
                        target
                        for target in sorted(
                            (
                                value
                                for value in manifest["targets"]
                                if value["split"] == split
                                and not value["primary"]
                                and str(value["target_id"]) not in excluded_incomplete
                            ),
                            key=lambda value: int(value["selection_rank"]),
                        )
                        if str(target["target_id"]) not in semantics
                        and str(target["target_id"]) not in attempted_semantics
                    ]
                    wave.extend(unknown[: deficits[split]])
                if not wave:
                    break
                await run_semantic_wave(wave)

            active = _active_targets(
                manifest,
                semantics,
                excluded_target_ids=excluded_incomplete,
            )
            active_targets = [target for split in SPLIT_ORDER for target in active[split]]
            await run_variant_attempts(active_targets)
            incomplete = {
                str(target["target_id"])
                for target in active_targets
                if any((str(target["target_id"]), style) not in accepted for style in STYLE_KINDS)
            }
            if not incomplete:
                break
            previous_excluded_count = len(excluded_incomplete)
            excluded_incomplete.update(incomplete)
            if len(excluded_incomplete) == previous_excluded_count:
                break
    finally:
        await client.close()

    active = _active_targets(
        manifest,
        semantics,
        excluded_target_ids=excluded_incomplete,
    )
    active_targets = [target for split in SPLIT_ORDER for target in active[split]]
    published: list[dict[str, Any]] = []
    evaluator_inputs: list[dict[str, Any]] = []
    complete_groups = 0
    for target in active_targets:
        target_id = str(target["target_id"])
        if not all((target_id, style) in accepted for style in STYLE_KINDS):
            continue
        complete_groups += 1
        semantic = semantics[target_id]
        for style in STYLE_KINDS:
            variant = accepted[(target_id, style)]
            published_record = {
                "pair_id": deterministic_pair_id(target_id, style),
                "target_id": target_id,
                "source_pair_id": target["source_pair_id"],
                "split": target["split"],
                "variant_kind": style,
                "source": variant["source"],
                "target": target["target_text"],
                "generation_fingerprint": manifest["generation_fingerprint"],
            }
            published.append(published_record)
            evaluator_inputs.append(
                {
                    **published_record,
                    "semantic": semantic.model_dump(mode="json"),
                    "context_bundle": target.get("context_bundle", {}),
                    "grounding_stratum": str(target["selection_stratum"]).rsplit("|", 1)[-1],
                    "local_validation": "passed",
                    "semantic_evaluator_status": (
                        "passed" if semantic_validator is not None else "pending"
                    ),
                }
            )
    write_jsonl(published_path, published)
    write_jsonl(evaluator_path, evaluator_inputs)

    active_counts = {split: len(active[split]) for split in SPLIT_ORDER}
    primary_ids = {str(target["target_id"]) for target in manifest["targets"] if target["primary"]}
    backfilled = {
        split: sum(str(target["target_id"]) not in primary_ids for target in active[split])
        for split in SPLIT_ORDER
    }
    ineligible = sum(not semantic.eligible for semantic in semantics.values())
    context_dependent = sum(semantic.context_dependent for semantic in semantics.values())
    artifacts = {
        name: sha256_file(path)
        for name, path in (
            ("manifest", manifest_path),
            ("semantics", semantics_path),
            ("variants", variants_path),
            ("usage", usage_path),
            ("published", published_path),
            ("semantic_evaluation_inputs", evaluator_path),
        )
    }
    report = {
        "artifact_schema_version": CONVERGENCE_SCHEMA_VERSION,
        "provider": "openai",
        "model": model,
        "generation_fingerprint": manifest["generation_fingerprint"],
        "input_fingerprint": manifest["input_fingerprint"],
        "styles": list(STYLE_KINDS),
        "selection": {
            "requested": dict(manifest["allocation"]),
            "active": active_counts,
            "backfilled": backfilled,
            "context_ineligible": ineligible,
            "context_dependent": context_dependent,
            "variant_incomplete_replaced": len(excluded_incomplete),
            "candidate_pool": len(manifest["targets"]),
        },
        "semantics": {
            "generated_this_run": generated_semantics,
            "available_total": len(semantics),
        },
        "variants": {
            "attempts_this_run": generated_variant_attempts,
            "accepted_this_run": accepted_variants,
            "accepted_total": len(accepted),
            "complete_groups": complete_groups,
            "published_rows": len(published),
            "incomplete_active_groups": len(active_targets) - complete_groups,
            "allowed_neutral_identity_rows": sum(
                record["variant_kind"] == "neutral_everyday"
                and _near_duplicate(str(record["source"]), str(record["target"])) is not None
                for record in published
            ),
            "target_identity_rows_allowed": False,
        },
        "usage": run_usage,
        "cumulative_usage": cumulative_usage,
        "rejections_this_run": dict(sorted(run_rejections.items())),
        "cumulative_rejections": dict(sorted(cumulative_rejections.items())),
        "errors": dict(sorted(errors.items())),
        "semantic_evaluator": {
            "hook_configured": semantic_validator is not None,
            "artifact_rows": len(evaluator_inputs),
        },
        "artifacts": artifacts,
        "message_text_persisted_in_report": False,
    }
    write_json(report_path, report)
    return report


generate_openai_convergence_pilot = generate_openai_convergence_data
