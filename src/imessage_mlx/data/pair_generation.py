from __future__ import annotations

import asyncio
import html
import json
import os
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

from imessage_mlx.data.rewrite import STRUCTURAL_TOKEN_RE, validate_rewrite_pair
from imessage_mlx.utils import atomic_write_text, ensure_private_dir, read_jsonl, write_json

NEUTRALIZATION_INSTRUCTIONS = """
Convert each message into natural, casual conversational wording while preserving its exact
meaning. The result is a style-neutral text-message draft, not formal or business writing.

Rules:
- Rewrite the message itself. Do not reply to it.
- Preserve factual details, intent, uncertainty, questions, and emotional meaning.
- Preserve ordinary casual vocabulary and contractions such as "I'm", "don't", and "that's".
- Remove only distinctly personal slang, unclear abbreviations, stylized casing, and idiosyncratic
  punctuation.
- Apply these corpus-specific, case-insensitive meanings: "sm" means "something"; "ts" means either
  "this" or "type shit". Expand "ts" only when surrounding grammar and context clearly determine
  which meaning applies; otherwise preserve "ts" rather than guessing.
- Do not replace simple words with more formal synonyms or add politeness that was not present.
- Keep profanity, humor, emotional intensity, and directness when they affect meaning.
- Make the smallest wording change needed to produce a clear, casually conversational draft.
- Do not add information, explanations, quotation marks, or commentary.
- If wording is already clear and casually neutral, return it unchanged.
- Return exactly one item for every supplied pair_id and preserve each pair_id verbatim.
""".strip()


class NeutralizedItem(BaseModel):
    pair_id: str
    neutral_text: str


class NeutralizedBatch(BaseModel):
    items: list[NeutralizedItem]


class StructuredOutputMissingError(RuntimeError):
    pass


class DuplicatePairIdentifierError(RuntimeError):
    pass


class PairIdentifierMismatchError(RuntimeError):
    pass


def create_rewrite_review(
    pairs_path: str | Path,
    output_path: str | Path,
    *,
    sample_size: int = 50,
) -> dict[str, Any]:
    if sample_size <= 0:
        raise ValueError("Review sample size must be positive")
    pairs = sorted(
        read_jsonl(pairs_path),
        key=lambda value: (int(value["timestamp_ns"]), str(value["pair_id"])),
    )
    if not pairs:
        raise ValueError("Rewrite pair input contains no records")
    selected_count = min(sample_size, len(pairs))
    if selected_count == 1:
        indices = [len(pairs) - 1]
    else:
        indices = [
            round(index * (len(pairs) - 1) / (selected_count - 1))
            for index in range(selected_count)
        ]
    selected = [pairs[index] for index in indices]
    exact_matches = sum(str(value["neutral_text"]) == str(value["styled_text"]) for value in pairs)
    ratios = [
        len(str(value["neutral_text"])) / max(1, len(str(value["styled_text"]))) for value in pairs
    ]
    lines = [
        "# Rewrite Pair Pilot Review",
        "",
        "This private sample is evenly distributed across the pilot timeline.",
        "For each pair, confirm that the neutral version preserves meaning while removing",
        "personal texting style. Mark problems directly in this ignored file.",
        "",
        f"- Total pairs: {len(pairs)}",
        f"- Sampled pairs: {len(selected)}",
        f"- Exact neutral/original matches: {exact_matches}",
        f"- Average neutral/original character ratio: {sum(ratios) / len(ratios):.2f}",
        "",
    ]
    for index, value in enumerate(selected, start=1):
        neutral = html.escape(str(value["neutral_text"]))
        styled = html.escape(str(value["styled_text"]))
        lines.extend(
            [
                f"## Pair {index}",
                "",
                "**Neutral input**",
                f"<pre>{neutral}</pre>",
                "",
                "**Original styled target**",
                f"<pre>{styled}</pre>",
                "",
                "- [ ] Meaning changed or information lost",
                "- [ ] Input is still too stylistically personal",
                "- [ ] Other problem",
                "",
            ]
        )
    atomic_write_text(output_path, "\n".join(lines) + "\n")
    return {
        "total_pairs": len(pairs),
        "sampled_pairs": len(selected),
        "exact_matches": exact_matches,
        "average_character_ratio": sum(ratios) / len(ratios),
        "output": str(Path(output_path)),
        "private_local_artifact": True,
    }


def select_outgoing_messages(
    messages_path: str | Path,
    *,
    limit: int,
    max_characters: int = 1_000,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if limit <= 0:
        raise ValueError("Pair generation limit must be positive")
    candidates: list[dict[str, Any]] = []
    counts = {
        "input_records": 0,
        "outgoing_records": 0,
        "skipped_empty": 0,
        "skipped_structural_tokens": 0,
        "skipped_too_long": 0,
        "skipped_duplicate_text": 0,
    }
    for record in read_jsonl(messages_path):
        counts["input_records"] += 1
        if record.get("sender_role") != "me":
            continue
        counts["outgoing_records"] += 1
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            counts["skipped_empty"] += 1
            continue
        if STRUCTURAL_TOKEN_RE.search(text):
            counts["skipped_structural_tokens"] += 1
            continue
        if len(text) > max_characters:
            counts["skipped_too_long"] += 1
            continue
        candidates.append(
            {
                "pair_id": str(record["message_id"]),
                "timestamp_ns": int(record["timestamp_ns"]),
                "styled_text": text,
            }
        )

    selected: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    for candidate in sorted(
        candidates,
        key=lambda value: (int(value["timestamp_ns"]), str(value["pair_id"])),
        reverse=True,
    ):
        styled_text = str(candidate["styled_text"])
        if styled_text in seen_text:
            counts["skipped_duplicate_text"] += 1
            continue
        seen_text.add(styled_text)
        selected.append(candidate)
        if len(selected) >= limit:
            break
    selected.reverse()
    counts["selected_records"] = len(selected)
    return selected, counts


def _parsed_batch(response: Any) -> NeutralizedBatch:
    for output in response.output:
        if getattr(output, "type", None) != "message":
            continue
        for content in output.content:
            parsed = getattr(content, "parsed", None)
            if isinstance(parsed, NeutralizedBatch):
                return parsed
    raise StructuredOutputMissingError(
        "OpenAI response did not contain the requested structured output"
    )


def _validated_results(
    batch: list[dict[str, Any]],
    neutralized: NeutralizedBatch,
) -> list[dict[str, Any]]:
    source_by_id = {str(value["pair_id"]): value for value in batch}
    returned_by_id: dict[str, NeutralizedItem] = {}
    for item in neutralized.items:
        if item.pair_id in returned_by_id:
            raise DuplicatePairIdentifierError("OpenAI returned a duplicate pair identifier")
        returned_by_id[item.pair_id] = item
    if set(returned_by_id) != set(source_by_id):
        raise PairIdentifierMismatchError(
            "OpenAI returned an incomplete or unexpected pair identifier set"
        )

    results = []
    for value in batch:
        pair_id = str(value["pair_id"])
        results.append(
            validate_rewrite_pair(
                {
                    "pair_id": pair_id,
                    "timestamp_ns": int(value["timestamp_ns"]),
                    "neutral_text": returned_by_id[pair_id].neutral_text,
                    "styled_text": str(value["styled_text"]),
                }
            )
        )
    return results


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


async def generate_openai_rewrite_pairs(
    messages_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    *,
    api_key: str,
    model: str,
    limit: int = 500,
    batch_size: int = 20,
    concurrency: int = 4,
    max_characters: int = 1_000,
) -> dict[str, Any]:
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required")
    if batch_size <= 0 or concurrency <= 0:
        raise ValueError("Batch size and concurrency must be positive")

    selected, selection_counts = select_outgoing_messages(
        messages_path,
        limit=limit,
        max_characters=max_characters,
    )
    destination = Path(output_path)
    existing_ids = (
        {str(record["pair_id"]) for record in read_jsonl(destination)}
        if destination.exists()
        else set()
    )
    pending = [value for value in selected if str(value["pair_id"]) not in existing_ids]
    batches = [pending[start : start + batch_size] for start in range(0, len(pending), batch_size)]
    semaphore = asyncio.Semaphore(concurrency)
    client = AsyncOpenAI(api_key=api_key, timeout=120.0, max_retries=5)

    async def neutralize(batch: list[dict[str, Any]]):
        request_payload = [
            {"pair_id": value["pair_id"], "text": value["styled_text"]} for value in batch
        ]
        async with semaphore:
            response = await client.responses.parse(
                model=model,
                instructions=NEUTRALIZATION_INSTRUCTIONS,
                input=json.dumps(request_payload, ensure_ascii=False),
                text_format=NeutralizedBatch,
                store=False,
            )
        usage = getattr(response, "usage", None)
        return (
            _validated_results(batch, _parsed_batch(response)),
            {
                "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            },
        )

    generated = 0
    failed_batches = 0
    error_types: Counter[str] = Counter()
    usage_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    tasks = [asyncio.create_task(neutralize(batch)) for batch in batches]
    try:
        for task in asyncio.as_completed(tasks):
            try:
                records, usage = await task
            except Exception as error:
                failed_batches += 1
                error_types[type(error).__name__] += 1
                continue
            generated += _append_private_jsonl(destination, records)
            for name, value in usage.items():
                usage_totals[name] += value
    finally:
        await client.close()

    report = {
        "provider": "openai",
        "model": model,
        "requested_limit": limit,
        "selection": selection_counts,
        "already_present": len(existing_ids & {str(value["pair_id"]) for value in selected}),
        "pending_pairs": len(pending),
        "generated_pairs": generated,
        "successful_batches": len(batches) - failed_batches,
        "failed_batches": failed_batches,
        "error_types": dict(sorted(error_types.items())),
        "usage": usage_totals,
        "message_text_persisted_in_report": False,
    }
    write_json(report_path, report)
    return report
