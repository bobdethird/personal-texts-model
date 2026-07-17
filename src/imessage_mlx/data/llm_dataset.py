# ruff: noqa: E501
"""Ground-zero LLM-only rewrite dataset builder.

Every data decision is delegated to a hosted GPT model through two structured calls per
conversation window: a proposal call that groups the owner's messages into turns, decides
usability, and writes an assistant-style draft for each turn; and a judge call that accepts or
rejects every proposed pair. Local code performs no linguistic or quality heuristics; it only
slices fixed-size windows, verifies that model responses are mechanically well formed (echoed
identifiers, in-range owner-only indices), caches finished windows for resume, and serializes
the accepted pairs into chronological train/valid/test splits.
"""

from __future__ import annotations

import asyncio
import html
import json
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from imessage_mlx.utils import (
    atomic_write_text,
    ensure_private_dir,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_json,
)

LLM_DATASET_SCHEMA_VERSION = 1
PROPOSAL_PROMPT_VERSION = "llm-dataset-proposal-v1"
JUDGE_PROMPT_VERSION = "llm-dataset-judge-v1"
SPLIT_NAMES = ("train", "valid", "test")

# The MLX-LM prompt prefix used by the Qwen adapter lane; defined here so the dataset
# builder has no dependency on the adapter worker module.
REWRITE_INSTRUCTION = (
    "Rewrite the draft in the learned casual texting style. Preserve every fact, intent, "
    "question, negation, and degree of uncertainty. Return only the rewritten message.\n\nDraft:\n"
)

REQUIRED_MESSAGE_KEYS = (
    "message_id",
    "chat_id",
    "timestamp_ns",
    "sender_role",
    "participant_id",
    "text",
)

PROPOSAL_INSTRUCTIONS = """
You are building training data for a personal rewrite model that turns an assistant-style draft
into the device owner's real texting style. You receive one window of a real conversation. The
messages are numbered; "Me" is the device owner and every other name is another participant.
Timestamps are local time. Decide everything by reading the conversation; there are no
mechanical rules.

1. Group the owner's messages into turns. A turn is one or more "Me" messages, in order and
   possibly interrupted by another participant's bubble, that express one connected thought the
   owner sent as a burst of texts.
2. Decide which turns are usable as rewrite targets. Skip pure reactions such as "lol", "ok",
   or a lone emoji, fragments with no restatable content, automated or mass texts, and any turn
   whose meaning you cannot actually determine from this window.
3. For each usable turn write "draft": the message a generic, polished writing assistant would
   have produced when asked to say the same thing in the same situation. Preserve the complete
   meaning: every fact, intent, question, negation, number, time, and degree of uncertainty.
   Write plain, natural standard English and do not imitate the owner's wording, slang,
   abbreviations, casing, or punctuation habits.
   - Translate slang and texting shorthand into its plain-English meaning using your knowledge
     of current texting culture, and read unfamiliar coinages from the conversation context.
   - Keep proper nouns, project names, and coined names exactly as written; do not expand,
     explain, or respell them.
4. Write "note": one short sentence for a human reviewer saying what is happening and what the
   turn communicates in context.

Return one entry per usable turn. "message_indices" lists the window indices of the owner's
messages in that turn in ascending order; indices must refer to "Me" messages only and no index
may appear in more than one entry. Return an empty list when nothing is usable. Text such as
"<|attachment|>" marks an attachment and other <|...|> tokens are privacy redactions. If
previous_attempt_error is present, your prior response was rejected for exactly that reason;
return a corrected response. Copy window_id verbatim. Return only the requested structured
output.
""".strip()

JUDGE_INSTRUCTIONS = """
You are auditing candidate training pairs for a personal rewrite model. You receive one window
of a real conversation with numbered messages ("Me" is the device owner) and a list of
candidates. Each candidate pairs an assistant-style "draft" with the owner's real messages at
"message_indices". Accepted pairs teach a model to turn the draft into the owner's real text.

Judge every candidate by your own reading of the conversation and decide accept or reject:
- The draft must communicate the same thing as the owner's messages do in this context: the
  same intent, facts, questions, negation, numbers, and uncertainty. Reject when meaning was
  added, lost, or changed, or when slang or shorthand was misread.
- The draft must read like a neutral assistant wrote it in plain standard English. Reject
  drafts that copy the owner's distinctive wording or style, and near-copies of the original.
- Proper nouns and coined names must appear verbatim, neither expanded nor respelled.
- The grouped messages must form one coherent turn by the owner. Reject groupings that glue
  unrelated thoughts together or that cut one thought incoherently.
- Reject turns that were never usable: pure reactions, contentless fragments, automated texts,
  or meaning that cannot be determined from this window.

Return exactly one verdict per candidate, echoing its example_id, with accept true or false and
a short reason. If previous_attempt_error is present, your prior response was rejected for
exactly that reason; return a corrected response. Copy window_id verbatim. Return only the
requested structured output.
""".strip()


class ProposedExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_indices: list[int] = Field(
        description=(
            "Ascending window indices of the owner's messages forming this one turn; 'Me' "
            "messages only, and no index may be reused across entries."
        )
    )
    draft: str = Field(
        description="The assistant-style draft carrying the turn's complete meaning."
    )
    note: str = Field(
        description="One short sentence of context for a human reviewer.",
    )


class WindowProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_id: str
    examples: list[ProposedExample]


class ExampleVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    example_id: str
    accept: bool
    reason: str = Field(description="Short justification for the verdict.")


class WindowJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_id: str
    verdicts: list[ExampleVerdict]


class WindowResponseError(RuntimeError):
    """The model response was mechanically malformed for this window."""


class LlmDatasetResumeError(RuntimeError):
    """Existing artifacts are incompatible with the requested run."""


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
    raise WindowResponseError("Response did not contain the requested structured output")


def _local_time(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000).strftime("%Y-%m-%d %H:%M")


def _participant_labels(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Give every non-owner participant a chat-stable first-name label."""
    labels: dict[str, str] = {}
    used = {"Me"}
    unnamed = 0
    for row in rows:
        if str(row.get("sender_role")) == "me":
            continue
        participant = str(row["participant_id"])
        if participant in labels:
            continue
        name = str(row.get("sender_name") or "").strip()
        base = name.split()[0] if name else ""
        if not base:
            unnamed += 1
            base = f"P{unnamed}"
        label = base
        suffix = 2
        while label in used:
            label = f"{base} {suffix}"
            suffix += 1
        labels[participant] = label
        used.add(label)
    return labels


def build_windows(
    messages: Sequence[Mapping[str, Any]],
    *,
    max_window_messages: int = 40,
) -> list[dict[str, Any]]:
    """Slice each chat's chronological messages into fixed-size windows.

    Purely mechanical transport slicing: no session gaps, no content decisions.
    """
    if max_window_messages < 2:
        raise ValueError("Windows must allow at least two messages")
    chats: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for message in messages:
        if any(key not in message for key in REQUIRED_MESSAGE_KEYS):
            raise ValueError("Extracted message is missing required dataset fields")
        chats[str(message["chat_id"])].append(dict(message))

    windows: list[dict[str, Any]] = []
    for chat_id in sorted(chats):
        rows = sorted(
            chats[chat_id],
            key=lambda row: (int(row["timestamp_ns"]), str(row["message_id"])),
        )
        labels = _participant_labels(rows)
        for start in range(0, len(rows), max_window_messages):
            segment = rows[start : start + max_window_messages]
            entries = [
                {
                    "i": index,
                    "who": (
                        "Me"
                        if str(row["sender_role"]) == "me"
                        else labels[str(row["participant_id"])]
                    ),
                    "at": _local_time(int(row["timestamp_ns"])),
                    "text": str(row["text"]),
                }
                for index, row in enumerate(segment)
            ]
            window_id = sha256_text(
                "llm-window:" + chat_id + ":" + ",".join(str(row["message_id"]) for row in segment)
            )[:24]
            is_group = any(bool(row.get("is_group")) for row in segment)
            payload = {
                "window_id": window_id,
                "chat_type": "group" if is_group else "direct",
                "messages": entries,
            }
            windows.append(
                {
                    "window_id": window_id,
                    "chat_id": chat_id,
                    "is_group": is_group,
                    "rows": segment,
                    "payload": payload,
                    "content_fingerprint": sha256_text(_canonical_json(payload)),
                    "has_me": any(str(row["sender_role"]) == "me" for row in segment),
                }
            )
    return windows


def example_id(window_id: str, message_indices: Sequence[int]) -> str:
    return sha256_text(
        f"llm-example:{LLM_DATASET_SCHEMA_VERSION}:{window_id}:"
        + ",".join(str(index) for index in message_indices)
    )[:24]


def validate_window_proposal(
    window: Mapping[str, Any],
    proposal: WindowProposal,
) -> list[dict[str, Any]]:
    """Mechanical response-integrity checks only; all judgment stays with the models."""
    if proposal.window_id != str(window["window_id"]):
        raise WindowResponseError("Proposal echoed the wrong window identifier")
    rows = window["rows"]
    used: set[int] = set()
    examples: list[dict[str, Any]] = []
    for entry in proposal.examples:
        indices = [int(value) for value in entry.message_indices]
        if not indices:
            raise WindowResponseError("An example lists no message indices")
        if any(index < 0 or index >= len(rows) for index in indices):
            raise WindowResponseError("An example references an index outside the window")
        if any(second <= first for first, second in zip(indices, indices[1:], strict=False)):
            raise WindowResponseError("Example message indices must be strictly ascending")
        if any(str(rows[index]["sender_role"]) != "me" for index in indices):
            raise WindowResponseError("Examples may target only the owner's own messages")
        if used.intersection(indices):
            raise WindowResponseError("A message index appears in more than one example")
        used.update(indices)
        draft = entry.draft.strip()
        if not draft:
            raise WindowResponseError("An example has an empty draft")
        examples.append(
            {
                "example_id": example_id(str(window["window_id"]), indices),
                "message_indices": indices,
                "draft": draft,
                "note": entry.note.strip(),
            }
        )
    return examples


def validate_window_judgement(
    window: Mapping[str, Any],
    judgement: WindowJudgement,
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if judgement.window_id != str(window["window_id"]):
        raise WindowResponseError("Judgement echoed the wrong window identifier")
    expected = {str(entry["example_id"]) for entry in examples}
    verdicts: dict[str, dict[str, Any]] = {}
    for verdict in judgement.verdicts:
        if verdict.example_id not in expected:
            raise WindowResponseError("Judgement references an unknown example identifier")
        if verdict.example_id in verdicts:
            raise WindowResponseError("Judgement repeats an example identifier")
        verdicts[verdict.example_id] = {
            "accept": bool(verdict.accept),
            "reason": verdict.reason.strip(),
        }
    if set(verdicts) != expected:
        raise WindowResponseError("Judgement must return exactly one verdict per candidate")
    return verdicts


async def propose_window_examples(
    client: AsyncOpenAI,
    *,
    model: str,
    window: Mapping[str, Any],
    previous_attempt_error: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    payload: dict[str, Any] = dict(window["payload"])
    if previous_attempt_error:
        payload["previous_attempt_error"] = previous_attempt_error
    response = await client.responses.parse(
        model=model,
        instructions=PROPOSAL_INSTRUCTIONS,
        input=json.dumps(payload, sort_keys=True, ensure_ascii=False),
        text_format=WindowProposal,
        store=False,
    )
    parsed = _parsed_model(response, WindowProposal)
    assert isinstance(parsed, WindowProposal)
    return validate_window_proposal(window, parsed), _usage(response)


async def judge_window_examples(
    client: AsyncOpenAI,
    *,
    model: str,
    window: Mapping[str, Any],
    examples: Sequence[Mapping[str, Any]],
    previous_attempt_error: str | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    payload: dict[str, Any] = dict(window["payload"])
    payload["candidates"] = [
        {
            "example_id": str(entry["example_id"]),
            "message_indices": list(entry["message_indices"]),
            "draft": str(entry["draft"]),
        }
        for entry in examples
    ]
    if previous_attempt_error:
        payload["previous_attempt_error"] = previous_attempt_error
    response = await client.responses.parse(
        model=model,
        instructions=JUDGE_INSTRUCTIONS,
        input=json.dumps(payload, sort_keys=True, ensure_ascii=False),
        text_format=WindowJudgement,
        store=False,
    )
    parsed = _parsed_model(response, WindowJudgement)
    assert isinstance(parsed, WindowJudgement)
    return validate_window_judgement(window, parsed, examples), _usage(response)


def _run_metadata(model: str, judge_model: str) -> dict[str, Any]:
    return {
        "schema_version": LLM_DATASET_SCHEMA_VERSION,
        "proposal_prompt_version": PROPOSAL_PROMPT_VERSION,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "model": model,
        "judge_model": judge_model,
    }


def _load_completed_windows(
    path: Path,
    metadata: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Load prior window results; the newest record per fingerprint wins."""
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        for field, expected in metadata.items():
            if record.get(field) != expected:
                raise LlmDatasetResumeError(
                    "Existing results are incompatible with the requested "
                    f"{field.replace('_', ' ')}; use a fresh output directory"
                )
        fingerprint = record.get("window_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise LlmDatasetResumeError("Existing results lack window fingerprints")
        records[fingerprint] = record
    return records


def load_accepted_examples(results_path: str | Path) -> list[dict[str, Any]]:
    """Collect judge-accepted pairs from a generation run, newest record per window."""
    latest: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(results_path):
        fingerprint = str(record.get("window_fingerprint", ""))
        if record.get("status") == "completed" and fingerprint:
            latest[fingerprint] = record
    accepted: list[dict[str, Any]] = []
    for fingerprint in sorted(latest):
        record = latest[fingerprint]
        for entry in record.get("examples", []):
            if not entry.get("accepted"):
                continue
            accepted.append(
                {
                    "pair_id": str(entry["pair_id"]),
                    "chat_id": str(record["chat_id"]),
                    "is_group": bool(record["is_group"]),
                    "message_ids": list(entry["message_ids"]),
                    "timestamp_ns": int(entry["timestamp_ns"]),
                    "source": str(entry["source"]),
                    "target": str(entry["target"]),
                    "note": str(entry.get("note", "")),
                }
            )
    return accepted


async def build_llm_dataset(
    messages_path: str | Path,
    output_dir: str | Path,
    *,
    api_key: str,
    model: str,
    judge_model: str | None = None,
    max_window_messages: int = 40,
    limit_windows: int | None = None,
    concurrency: int = 4,
    max_attempts: int = 3,
    checkpoint_every: int = 50,
) -> dict[str, Any]:
    """Generate judge-screened rewrite pairs with model-only decisions.

    Generation never publishes training splits; the separate censor stage screens the
    accepted pairs and owns every published dataset file.
    """
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required")
    if concurrency <= 0 or max_attempts <= 0 or checkpoint_every <= 0:
        raise ValueError("Concurrency, attempts, and checkpoint interval must be positive")
    if limit_windows is not None and limit_windows <= 0:
        raise ValueError("The window limit must be positive when provided")
    judge = judge_model or model

    messages = list(read_jsonl(messages_path))
    if not messages:
        raise ValueError("Dataset generation requires extracted messages")
    windows = build_windows(messages, max_window_messages=max_window_messages)
    windows.sort(key=lambda window: str(window["content_fingerprint"]))
    eligible = [window for window in windows if window["has_me"]]
    skipped_no_me = len(windows) - len(eligible)
    selected = eligible[:limit_windows] if limit_windows is not None else eligible

    output = ensure_private_dir(output_dir)
    results_path = output / "results.jsonl"
    report_path = output / "report.json"
    metadata = _run_metadata(model, judge)
    completed = {
        fingerprint: record
        for fingerprint, record in _load_completed_windows(results_path, metadata).items()
        if record.get("status") == "completed"
    }
    known_fingerprints = {str(window["content_fingerprint"]) for window in windows}
    for fingerprint in completed:
        if fingerprint not in known_fingerprints:
            raise LlmDatasetResumeError(
                "Existing results reference unknown windows; the extracted messages or window "
                "size changed, so use a fresh output directory"
            )

    pending = [window for window in selected if str(window["content_fingerprint"]) not in completed]
    resumed = len(selected) - len(pending)

    counters: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    usage_totals = {
        "proposal": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "judge": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }
    semaphore = asyncio.Semaphore(concurrency)
    client = AsyncOpenAI(api_key=api_key, timeout=180.0, max_retries=5)

    async def call_with_retries(kind: str, request) -> tuple[Any, str | None]:
        last_error: str | None = None
        for _attempt in range(max_attempts):
            counters[f"{kind}_attempts"] += 1
            try:
                value, usage = await request(last_error)
            except WindowResponseError as error:
                failure_reasons[f"{kind}:{type(error).__name__}"] += 1
                last_error = f"{type(error).__name__}: {error}"
                continue
            except Exception as error:  # transport failures are counted, never fatal
                failure_reasons[f"{kind}:api:{type(error).__name__}"] += 1
                last_error = f"{type(error).__name__}"
                continue
            for name, amount in usage.items():
                usage_totals[kind][name] += int(amount)
            return value, None
        return None, last_error or "unknown"

    async def process(window: Mapping[str, Any]) -> dict[str, Any]:
        async with semaphore:
            record = {
                **metadata,
                "window_id": str(window["window_id"]),
                "window_fingerprint": str(window["content_fingerprint"]),
                "chat_id": str(window["chat_id"]),
                "is_group": bool(window["is_group"]),
                "window_message_count": len(window["rows"]),
            }

            async def propose(previous: str | None):
                return await propose_window_examples(
                    client,
                    model=model,
                    window=window,
                    previous_attempt_error=previous,
                )

            examples, proposal_error = await call_with_retries("proposal", propose)
            if examples is None:
                return {
                    **record,
                    "status": "proposal_failed",
                    "failure_reason": proposal_error,
                    "examples": [],
                }
            if not examples:
                return {**record, "status": "completed", "failure_reason": None, "examples": []}

            async def judge_request(previous: str | None):
                return await judge_window_examples(
                    client,
                    model=judge,
                    window=window,
                    examples=examples,
                    previous_attempt_error=previous,
                )

            verdicts, judge_error = await call_with_retries("judge", judge_request)
            if verdicts is None:
                return {
                    **record,
                    "status": "judge_failed",
                    "failure_reason": judge_error,
                    "examples": [],
                }

            rows = window["rows"]
            published = []
            for entry in examples:
                indices = list(entry["message_indices"])
                verdict = verdicts[str(entry["example_id"])]
                published.append(
                    {
                        "example_id": entry["example_id"],
                        "pair_id": entry["example_id"],
                        "message_indices": indices,
                        "message_ids": [str(rows[index]["message_id"]) for index in indices],
                        "timestamp_ns": int(rows[indices[-1]]["timestamp_ns"]),
                        "target": "\n".join(str(rows[index]["text"]) for index in indices),
                        "source": entry["draft"],
                        "note": entry["note"],
                        "accepted": verdict["accept"],
                        "judge_reason": verdict["reason"],
                    }
                )
            return {**record, "status": "completed", "failure_reason": None, "examples": published}

    processed_this_run = 0
    progress_started = time.monotonic()
    progress_total = len(selected)

    def render_progress() -> None:
        completed_count = resumed + processed_this_run
        fraction = completed_count / progress_total if progress_total else 1.0
        width = 30
        filled = min(width, int(width * fraction))
        bar = "#" * filled + "-" * (width - filled)
        elapsed = max(time.monotonic() - progress_started, 0.001)
        rate = processed_this_run / elapsed
        remaining = max(progress_total - completed_count, 0)
        eta_minutes = remaining / rate / 60 if rate else 0.0
        print(
            f"\r[{bar}] {completed_count}/{progress_total} "
            f"({fraction:6.2%}) {rate * 60:5.1f} windows/min "
            f"ETA {eta_minutes:5.1f}m",
            end="",
            file=sys.stderr,
            flush=True,
        )

    render_progress()
    try:
        for start in range(0, len(pending), checkpoint_every):
            batch = pending[start : start + checkpoint_every]
            with results_path.open("a", encoding="utf-8") as handle:
                tasks = [asyncio.create_task(process(window)) for window in batch]
                for finished in asyncio.as_completed(tasks):
                    result = await finished
                    handle.write(json.dumps(result, sort_keys=True, ensure_ascii=False) + "\n")
                    handle.flush()
                    processed_this_run += 1
                    counters[str(result["status"])] += 1
                    if result["status"] == "completed":
                        completed[str(result["window_fingerprint"])] = result
                    render_progress()
            results_path.chmod(0o600)
    finally:
        print(file=sys.stderr)
        close = getattr(client, "close", None)
        if close is not None:
            await close()

    selected_fingerprints = {str(window["content_fingerprint"]) for window in selected}
    accepted_examples = 0
    proposed_examples = 0
    rejected_examples = 0
    for fingerprint in sorted(selected_fingerprints):
        record = completed.get(fingerprint)
        if record is None:
            continue
        for entry in record.get("examples", []):
            proposed_examples += 1
            if entry.get("accepted"):
                accepted_examples += 1
            else:
                rejected_examples += 1

    report = {
        **metadata,
        "task": "llm_only_rewrite_dataset",
        "provider": "openai",
        "parameters": {
            "max_window_messages": max_window_messages,
            "limit_windows": limit_windows,
            "concurrency": concurrency,
            "max_attempts": max_attempts,
        },
        "windows": {
            "total": len(windows),
            "skipped_no_owner_messages": skipped_no_me,
            "eligible": len(eligible),
            "selected": len(selected),
            "resumed_completed": resumed,
            "processed_this_run": processed_this_run,
            "completed": counters["completed"] + resumed,
            "proposal_failed": counters["proposal_failed"],
            "judge_failed": counters["judge_failed"],
        },
        "examples": {
            "proposed": proposed_examples,
            "accepted": accepted_examples,
            "rejected": rejected_examples,
            "acceptance_rate": (
                round(accepted_examples / proposed_examples, 4) if proposed_examples else None
            ),
        },
        "attempts": {
            "proposal": counters["proposal_attempts"],
            "judge": counters["judge_attempts"],
        },
        "failures": dict(sorted(failure_reasons.items())),
        "usage": usage_totals,
        "artifacts": {"results": sha256_file(results_path)},
        "publication": "run censor-llm-dataset to screen and publish training splits",
        "message_text_persisted_in_report": False,
    }
    write_json(report_path, report)
    return report


def create_llm_dataset_review(
    results_path: str | Path,
    output_path: str | Path,
    *,
    sample_size: int = 50,
) -> dict[str, Any]:
    """Render a private Markdown sample of generated pairs for human review."""
    if sample_size <= 0:
        raise ValueError("Review sample size must be positive")
    examples: list[dict[str, Any]] = []
    for record in read_jsonl(results_path):
        if record.get("status") != "completed":
            continue
        for entry in record.get("examples", []):
            examples.append({**entry, "chat_id": record.get("chat_id", "")})
    if not examples:
        raise ValueError("Results contain no generated examples to review")
    examples.sort(key=lambda entry: (int(entry.get("timestamp_ns", 0)), str(entry["pair_id"])))
    count = min(sample_size, len(examples))
    if count == 1:
        indices = [len(examples) - 1]
    else:
        indices = [round(index * (len(examples) - 1) / (count - 1)) for index in range(count)]
    selected = [examples[index] for index in sorted(set(indices))]
    accepted_total = sum(1 for entry in examples if entry.get("accepted"))

    lines = [
        "# LLM Dataset Review",
        "",
        "Private sample of generated pairs, evenly spread across the timeline. The draft is the",
        "model input; the original bubbles are the training target. The note and verdict are the",
        "generating and judging models' own explanations.",
        "",
        f"- Total examples: {len(examples)}",
        f"- Accepted: {accepted_total}",
        f"- Rejected: {len(examples) - accepted_total}",
        f"- Sampled: {len(selected)}",
        "",
    ]
    for number, entry in enumerate(selected, start=1):
        timestamp = _local_time(int(entry.get("timestamp_ns", 0)))
        verdict = "accepted" if entry.get("accepted") else "rejected"
        lines.extend(
            [
                f"## {number}. {timestamp} — {verdict}",
                "",
                f"Context note: {html.escape(str(entry.get('note', '')))}",
                "",
                f"Judge: {html.escape(str(entry.get('judge_reason', '')))}",
                "",
                "**Draft (model input)**",
                f"<pre>{html.escape(str(entry.get('source', entry.get('draft', ''))))}</pre>",
                "",
                "**Original (training target)**",
                f"<pre>{html.escape(str(entry.get('target', '')))}</pre>",
                "",
            ]
        )
    atomic_write_text(output_path, "\n".join(lines) + "\n")
    return {
        "schema_version": 1,
        "task": "llm_dataset_review",
        "examples": len(examples),
        "accepted": accepted_total,
        "sampled": len(selected),
        "private_review": str(Path(output_path)),
        "message_text_persisted_in_summary": False,
    }
