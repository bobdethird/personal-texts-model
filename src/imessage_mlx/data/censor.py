# ruff: noqa: E501
"""LLM censor stage: the only publisher of ready-to-train splits.

Every judge-accepted pair must pass a hosted screening call before publication. The censor
model reads the entire training row — the assistant-style draft, the owner's original
messages, and the reviewer note — and excludes any pair containing adult content (vulgar,
violent, or sexual material) or secrets and super-personal information (passwords, API keys,
credentials, financial numbers, government identifiers, and similar). Excluded pairs are
dropped entirely, never redacted. Pairs whose screening permanently fails are also excluded:
publication is fail-closed, so nothing unscreened can reach the training splits.

Consistent with the LLM-only mandate, local code performs no content heuristics; it batches
pairs, verifies responses mechanically (echoed identifiers, one verdict per pair), checkpoints
screenings for resume, and serializes the surviving pairs chronologically.
"""

from __future__ import annotations

import asyncio
import html
import json
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from imessage_mlx.data.llm_dataset import (
    REWRITE_INSTRUCTION,
    SPLIT_NAMES,
    load_accepted_examples,
)
from imessage_mlx.utils import (
    atomic_write_text,
    ensure_private_dir,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_json,
    write_jsonl,
)

CENSOR_SCHEMA_VERSION = 1
CENSOR_PROMPT_VERSION = "llm-dataset-censor-v1"
EXCLUSION_CATEGORIES = ("adult_content", "sensitive_secret")

CENSOR_INSTRUCTIONS = """
You are the final content censor for a private machine-learning dataset built from the device
owner's real text messages. You receive a batch of candidate training rows. Each row pairs an
assistant-style draft with the owner's original messages, plus a short reviewer note. A row you
allow will be used verbatim as training data, so screen every field of every row.

Exclude a row when any field contains:
- adult_content: vulgar, violent, or sexual material of any kind, including crude sexual slang,
  graphic descriptions, threats, or glorified violence. Ordinary mild profanity used as casual
  emphasis is not by itself adult content; exclude when the content itself is vulgar, violent,
  or sexual.
- sensitive_secret: anything that must never leak — passwords, passcodes, API keys, secret keys,
  access tokens, credentials, one-time or verification codes, private URLs that grant access,
  bank/card/account numbers, government identifiers, and comparably sensitive personal data.
  Placeholder tokens such as <|url|>, <|email|>, and <|phone|> are prior redactions, not
  secrets. A term merely naming a topic (for example "my password") without revealing the value
  is not itself a secret; exclude when a value or usable detail is present.

Rows are excluded entirely, never edited, so do not propose rewordings. When a row genuinely
falls in both categories, report the category that makes the leak most severe. When you are
genuinely uncertain whether content crosses either line, exclude it: dropping a safe row costs
little, publishing an unsafe one is unacceptable.

Return exactly one verdict per row, echoing its pair_id: allowed true or false, category
"adult_content" or "sensitive_secret" when excluded and "none" when allowed, and a short reason.
If previous_attempt_error is present, your prior response was rejected for exactly that reason;
return a corrected response. Copy batch_id verbatim. Return only the requested structured
output.
""".strip()


class PairScreening(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pair_id: str
    allowed: bool
    category: Literal["adult_content", "sensitive_secret", "none"] = Field(
        description="Exclusion category; 'none' if and only if the row is allowed."
    )
    reason: str = Field(description="Short justification for the verdict.")


class CensorBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_id: str
    verdicts: list[PairScreening]


class CensorResponseError(RuntimeError):
    """The censor response was mechanically malformed for this batch."""


class CensorResumeError(RuntimeError):
    """Existing censor artifacts are incompatible with the requested run."""


def _usage(response: Any) -> dict[str, int]:
    value = getattr(response, "usage", None)
    return {
        "input_tokens": int(getattr(value, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(value, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(value, "total_tokens", 0) or 0),
    }


def _parsed_batch(response: Any) -> CensorBatch:
    for output in getattr(response, "output", ()):
        if getattr(output, "type", None) != "message":
            continue
        for content in getattr(output, "content", ()):
            parsed = getattr(content, "parsed", None)
            if isinstance(parsed, CensorBatch):
                return parsed
    raise CensorResponseError("Response did not contain the requested structured output")


def _local_time(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000).strftime("%Y-%m-%d %H:%M")


def batch_identifier(pair_ids: Sequence[str]) -> str:
    return sha256_text(f"censor-batch:{CENSOR_SCHEMA_VERSION}:" + ",".join(pair_ids))[:24]


def validate_censor_batch(
    batch: CensorBatch,
    *,
    expected_batch_id: str,
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Mechanical response-integrity checks only; content judgment stays with the model."""
    if batch.batch_id != expected_batch_id:
        raise CensorResponseError("Censor response echoed the wrong batch identifier")
    expected = {str(pair["pair_id"]) for pair in pairs}
    verdicts: dict[str, dict[str, Any]] = {}
    for verdict in batch.verdicts:
        if verdict.pair_id not in expected:
            raise CensorResponseError("Censor response references an unknown pair identifier")
        if verdict.pair_id in verdicts:
            raise CensorResponseError("Censor response repeats a pair identifier")
        if verdict.allowed and verdict.category != "none":
            raise CensorResponseError("Allowed rows must use category 'none'")
        if not verdict.allowed and verdict.category == "none":
            raise CensorResponseError("Excluded rows require an exclusion category")
        verdicts[verdict.pair_id] = {
            "allowed": bool(verdict.allowed),
            "category": verdict.category,
            "reason": verdict.reason.strip(),
        }
    if set(verdicts) != expected:
        raise CensorResponseError("Censor response must return exactly one verdict per row")
    return verdicts


async def screen_pair_batch(
    client: AsyncOpenAI,
    *,
    model: str,
    pairs: Sequence[Mapping[str, Any]],
    previous_attempt_error: str | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    batch_id = batch_identifier([str(pair["pair_id"]) for pair in pairs])
    payload: dict[str, Any] = {
        "batch_id": batch_id,
        "rows": [
            {
                "pair_id": str(pair["pair_id"]),
                "draft": str(pair["source"]),
                "original": str(pair["target"]),
                "note": str(pair.get("note", "")),
            }
            for pair in pairs
        ],
    }
    if previous_attempt_error:
        payload["previous_attempt_error"] = previous_attempt_error
    response = await client.responses.parse(
        model=model,
        instructions=CENSOR_INSTRUCTIONS,
        input=json.dumps(payload, sort_keys=True, ensure_ascii=False),
        text_format=CensorBatch,
        store=False,
    )
    parsed = _parsed_batch(response)
    return (
        validate_censor_batch(parsed, expected_batch_id=batch_id, pairs=pairs),
        _usage(response),
    )


def _censor_metadata(model: str) -> dict[str, Any]:
    return {
        "censor_schema_version": CENSOR_SCHEMA_VERSION,
        "censor_prompt_version": CENSOR_PROMPT_VERSION,
        "censor_model": model,
    }


def _load_screenings(path: Path, metadata: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Load prior screenings; only completed verdicts count as resolved.

    Fail-closed records (`screening == "failed"`) are ignored here so a transient failure
    does not permanently exclude a pair; the next run re-screens it.
    """
    if not path.exists():
        return {}
    screenings: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        for field, expected in metadata.items():
            if record.get(field) != expected:
                raise CensorResumeError(
                    "Existing censor artifacts are incompatible with the requested "
                    f"{field.replace('_', ' ')}; use a fresh output directory"
                )
        pair_id = record.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise CensorResumeError("Existing censor artifacts lack pair identifiers")
        if record.get("screening") == "completed":
            screenings[pair_id] = record
    return screenings


def _publish_datasets(
    dataset_dir: Path,
    allowed: list[dict[str, Any]],
    *,
    valid_fraction: float,
    test_fraction: float,
) -> dict[str, int]:
    if valid_fraction < 0 or test_fraction < 0 or valid_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be nonnegative and sum below 1")
    ordered = sorted(allowed, key=lambda entry: (int(entry["timestamp_ns"]), entry["pair_id"]))
    total = len(ordered)
    test_count = int(total * test_fraction)
    valid_count = int(total * valid_fraction)
    train_count = total - valid_count - test_count
    splits = {
        "train": ordered[:train_count],
        "valid": ordered[train_count : train_count + valid_count],
        "test": ordered[train_count + valid_count :],
    }
    ensure_private_dir(dataset_dir)
    for name in SPLIT_NAMES:
        write_jsonl(dataset_dir / f"{name}.jsonl", splits[name])
        write_jsonl(
            dataset_dir / "bart" / f"{name}.jsonl",
            (
                {
                    "pair_id": entry["pair_id"],
                    "source": entry["source"],
                    "target": entry["target"],
                }
                for entry in splits[name]
            ),
        )
        write_jsonl(
            dataset_dir / "mlx" / f"{name}.jsonl",
            (
                {
                    "pair_id": entry["pair_id"],
                    "prompt": f"{REWRITE_INSTRUCTION}{entry['source']}",
                    "completion": entry["target"],
                }
                for entry in splits[name]
            ),
        )
    return {name: len(splits[name]) for name in SPLIT_NAMES}


def create_censor_review(
    censor_path: str | Path,
    output_path: str | Path,
    *,
    max_rows: int = 200,
) -> dict[str, Any]:
    """Render every excluded pair (up to a cap) so the user can spot-check the drops."""
    if max_rows <= 0:
        raise ValueError("The review row cap must be positive")
    excluded: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    total = 0
    for record in read_jsonl(censor_path):
        total += 1
        if record.get("allowed"):
            continue
        counts[str(record.get("category", "unknown"))] += 1
        excluded.append(record)
    excluded.sort(key=lambda entry: (int(entry.get("timestamp_ns", 0)), str(entry["pair_id"])))
    shown = excluded[:max_rows]
    lines = [
        "# Censor Exclusion Review",
        "",
        "Private record of every training row the censor dropped. Excluded rows never reach the",
        "published splits. This file itself contains the sensitive text; keep it local.",
        "",
        f"- Rows screened: {total}",
        f"- Rows excluded: {len(excluded)}",
        *(f"- Excluded as {name}: {count}" for name, count in sorted(counts.items())),
        f"- Shown below: {len(shown)}",
        "",
    ]
    for number, entry in enumerate(shown, start=1):
        timestamp = _local_time(int(entry.get("timestamp_ns", 0)))
        lines.extend(
            [
                f"## {number}. {timestamp} — {html.escape(str(entry.get('category', '')))}",
                "",
                f"Censor: {html.escape(str(entry.get('reason', '')))}",
                "",
                "**Draft (model input)**",
                f"<pre>{html.escape(str(entry.get('source', '')))}</pre>",
                "",
                "**Original (training target)**",
                f"<pre>{html.escape(str(entry.get('target', '')))}</pre>",
                "",
            ]
        )
    atomic_write_text(output_path, "\n".join(lines) + "\n")
    return {
        "schema_version": 1,
        "task": "llm_dataset_censor_review",
        "screened": total,
        "excluded": len(excluded),
        "excluded_by_category": dict(sorted(counts.items())),
        "shown": len(shown),
        "private_review": str(Path(output_path)),
        "message_text_persisted_in_summary": False,
    }


async def censor_llm_dataset(
    results_path: str | Path,
    output_dir: str | Path,
    *,
    api_key: str,
    model: str,
    batch_size: int = 20,
    concurrency: int = 4,
    max_attempts: int = 3,
    valid_fraction: float = 0.05,
    test_fraction: float = 0.05,
) -> dict[str, Any]:
    """Screen every accepted pair and publish only censor-allowed training splits."""
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required")
    if batch_size <= 0 or concurrency <= 0 or max_attempts <= 0:
        raise ValueError("Batch size, concurrency, and attempts must be positive")

    accepted = load_accepted_examples(results_path)
    output = ensure_private_dir(output_dir)
    censor_path = output / "censor.jsonl"
    report_path = output / "censor-report.json"
    review_path = output / "censor-review.md"
    dataset_dir = output / "dataset"
    metadata = _censor_metadata(model)
    screenings = _load_screenings(censor_path, metadata)

    by_pair_id = {str(pair["pair_id"]): pair for pair in accepted}
    pending = [
        pair
        for pair in accepted
        if str(pair["pair_id"]) not in screenings
        or screenings[str(pair["pair_id"])].get("screening") == "failed"
    ]
    batches = [pending[start : start + batch_size] for start in range(0, len(pending), batch_size)]

    counters: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    usage_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    semaphore = asyncio.Semaphore(concurrency)
    client = AsyncOpenAI(api_key=api_key, timeout=180.0, max_retries=5)
    processed_pairs = len(accepted) - len(pending)
    progress_started = time.monotonic()

    def render_progress() -> None:
        fraction = processed_pairs / len(accepted) if accepted else 1.0
        width = 30
        filled = min(width, int(width * fraction))
        bar = "#" * filled + "-" * (width - filled)
        elapsed = max(time.monotonic() - progress_started, 0.001)
        newly_processed = processed_pairs - (len(accepted) - len(pending))
        rate = newly_processed / elapsed
        remaining = max(len(accepted) - processed_pairs, 0)
        eta_minutes = remaining / rate / 60 if rate else 0.0
        print(
            f"\r[{bar}] {processed_pairs}/{len(accepted)} "
            f"({fraction:6.2%}) {rate * 60:5.1f} pairs/min "
            f"ETA {eta_minutes:5.1f}m",
            end="",
            file=sys.stderr,
            flush=True,
        )

    render_progress()

    async def screen(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        async with semaphore:
            last_error: str | None = None
            for _attempt in range(max_attempts):
                counters["censor_attempts"] += 1
                try:
                    verdicts, usage = await screen_pair_batch(
                        client,
                        model=model,
                        pairs=batch,
                        previous_attempt_error=last_error,
                    )
                except CensorResponseError as error:
                    failure_reasons[type(error).__name__] += 1
                    last_error = f"{type(error).__name__}: {error}"
                    continue
                except Exception as error:  # transport failures are counted, never fatal
                    failure_reasons[f"api:{type(error).__name__}"] += 1
                    last_error = f"{type(error).__name__}"
                    continue
                for name, amount in usage.items():
                    usage_totals[name] += int(amount)
                return [
                    {
                        **metadata,
                        **pair,
                        "screening": "completed",
                        **verdicts[str(pair["pair_id"])],
                    }
                    for pair in batch
                ]
            # Fail closed: pairs whose screening never succeeded are excluded, not published.
            return [
                {
                    **metadata,
                    **pair,
                    "screening": "failed",
                    "allowed": False,
                    "category": "unscreened_failure",
                    "reason": last_error or "screening failed",
                }
                for pair in batch
            ]

    try:
        for start in range(0, len(batches), max(1, concurrency)):
            wave = batches[start : start + max(1, concurrency)]
            with censor_path.open("a", encoding="utf-8") as handle:
                tasks = [asyncio.create_task(screen(batch)) for batch in wave]
                for finished in asyncio.as_completed(tasks):
                    records = await finished
                    for record in records:
                        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
                        screenings[str(record["pair_id"])] = record
                    handle.flush()
                    processed_pairs += len(records)
                    render_progress()
            censor_path.chmod(0o600)
    finally:
        print(file=sys.stderr)
        close = getattr(client, "close", None)
        if close is not None:
            await close()

    allowed: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    screening_failures = 0
    for pair_id, record in screenings.items():
        if pair_id not in by_pair_id:
            continue
        if record.get("screening") == "failed":
            screening_failures += 1
        if record.get("allowed"):
            allowed.append(by_pair_id[pair_id])
        else:
            category_counts[str(record.get("category", "unknown"))] += 1

    split_counts = _publish_datasets(
        dataset_dir,
        allowed,
        valid_fraction=valid_fraction,
        test_fraction=test_fraction,
    )
    review = create_censor_review(censor_path, review_path)

    report = {
        **metadata,
        "task": "llm_dataset_censor",
        "provider": "openai",
        "parameters": {
            "batch_size": batch_size,
            "concurrency": concurrency,
            "max_attempts": max_attempts,
            "valid_fraction": valid_fraction,
            "test_fraction": test_fraction,
        },
        "pairs": {
            "accepted_input": len(accepted),
            "screened_previously": len(accepted) - len(pending),
            "screened_this_run": len(pending),
            "allowed": len(allowed),
            "excluded": len(accepted) - len(allowed),
            "excluded_by_category": dict(sorted(category_counts.items())),
            "screening_failures_excluded": screening_failures,
        },
        "splits": split_counts,
        "attempts": counters["censor_attempts"],
        "failures": dict(sorted(failure_reasons.items())),
        "usage": usage_totals,
        "artifacts": {
            "censor": sha256_file(censor_path),
            **{
                f"dataset_{name}": sha256_file(dataset_dir / f"{name}.jsonl")
                for name in SPLIT_NAMES
            },
        },
        "review": {
            "excluded_rows_rendered": review["shown"],
            "private_review": review["private_review"],
        },
        "published_splits_censored_only": True,
        "message_text_persisted_in_report": False,
    }
    write_json(report_path, report)
    return report
