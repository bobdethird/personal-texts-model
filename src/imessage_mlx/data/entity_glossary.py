from __future__ import annotations

import html
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from imessage_mlx.data.context_retrieval import STOPWORDS
from imessage_mlx.utils import atomic_write_text, read_jsonl, sha256_text, write_json

GLOSSARY_SCHEMA_VERSION = 1
TOKEN_RE = re.compile(r"\b[\w'-]{2,}\b", re.UNICODE)
PLACEHOLDER_RE = re.compile(r"<\|[^|>]*\|>")
INFORMATIVE_RE = re.compile(
    r"\b(?:is|are|uses?|calls?|project|agent|app|tool|system|model|building|working)\b",
    re.IGNORECASE,
)
# Coined terms are identified by corpus rarity instead of capitalization because texting is
# largely lowercase. The rarity band mirrors TemporalMessageIndex.query_terms so the glossary
# proposes the same kind of distinctive vocabulary that retrieval treats as searchable.
RARE_DOCUMENT_FRACTION = 0.002
MAX_RARE_DOCUMENTS = 500
# Texting filler is style vocabulary the rewrite adapter should learn, not an entity whose
# meaning needs a reviewed definition.
TEXTING_FILLER = {
    "aight",
    "alr",
    "brb",
    "bruh",
    "btw",
    "cuz",
    "def",
    "fr",
    "gonna",
    "gotta",
    "hbu",
    "idk",
    "idc",
    "ig",
    "imo",
    "jk",
    "kinda",
    "lmao",
    "lmfao",
    "lol",
    "nah",
    "ngl",
    "nvm",
    "omg",
    "omw",
    "prolly",
    "probs",
    "pls",
    "plz",
    "rly",
    "smth",
    "sorta",
    "sry",
    "tbh",
    "tho",
    "thx",
    "tryna",
    "wanna",
    "wya",
    "wyd",
    "yea",
    "yeah",
    "yep",
    "yup",
}


def _candidate_terms(text: str) -> set[str]:
    cleaned = PLACEHOLDER_RE.sub(" ", text.replace("\u2019", "'"))
    return {
        token
        for token in TOKEN_RE.findall(cleaned)
        if token.casefold() not in STOPWORDS
        and token.casefold() not in TEXTING_FILLER
        and "'" not in token
        and not token.replace("-", "").isdigit()
    }


def _entry_id(term: str) -> str:
    return sha256_text(f"entity-glossary:{GLOSSARY_SCHEMA_VERSION}:{term.casefold()}")[:24]


def propose_entity_glossary(
    messages_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    *,
    minimum_occurrences: int = 2,
    maximum_entries: int = 200,
    evidence_per_entry: int = 5,
) -> dict[str, Any]:
    if minimum_occurrences <= 0 or maximum_entries <= 0 or evidence_per_entry <= 0:
        raise ValueError("Glossary proposal limits must be positive")
    messages = list(read_jsonl(messages_path))
    document_frequency: Counter[str] = Counter()
    for message in messages:
        for term in _candidate_terms(str(message.get("text", "")).strip()):
            document_frequency[term.casefold()] += 1
    rare_ceiling = max(
        minimum_occurrences,
        min(MAX_RARE_DOCUMENTS, math.ceil(RARE_DOCUMENT_FRACTION * len(messages))),
    )

    occurrences: Counter[str] = Counter()
    display: dict[str, Counter[str]] = defaultdict(Counter)
    evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for message in messages:
        if message.get("sender_role") != "me":
            continue
        text = str(message.get("text", "")).strip()
        for term in _candidate_terms(text):
            key = term.casefold()
            if document_frequency[key] > rare_ceiling:
                continue
            occurrences[key] += 1
            display[key][term] += 1
            if len(evidence[key]) < evidence_per_entry or INFORMATIVE_RE.search(text):
                evidence[key].append(
                    {
                        "message_id": str(message["message_id"]),
                        "chat_id": str(message["chat_id"]),
                        "timestamp_ns": int(message["timestamp_ns"]),
                        "text": text,
                    }
                )
                evidence[key] = sorted(
                    evidence[key],
                    key=lambda value: (
                        INFORMATIVE_RE.search(str(value["text"])) is None,
                        int(value["timestamp_ns"]),
                    ),
                )[:evidence_per_entry]
    selected = sorted(
        (key for key, count in occurrences.items() if count >= minimum_occurrences),
        key=lambda key: (-occurrences[key], key),
    )[:maximum_entries]
    entries = []
    for key in selected:
        term = display[key].most_common(1)[0][0]
        examples = evidence[key]
        proposed = next(
            (str(value["text"]) for value in examples if INFORMATIVE_RE.search(str(value["text"]))),
            str(examples[0]["text"]) if examples else "",
        )
        entries.append(
            {
                "entry_id": _entry_id(term),
                "term": term,
                "proposed_definition": proposed,
                "definition": "",
                "approved": False,
                "occurrences": occurrences[key],
                "valid_from_timestamp_ns": min(
                    (int(value["timestamp_ns"]) for value in examples),
                    default=0,
                ),
                "evidence_message_ids": [str(value["message_id"]) for value in examples],
                "evidence": examples,
            }
        )
    artifact = {
        "schema_version": GLOSSARY_SCHEMA_VERSION,
        "task": "private_entity_glossary",
        "approval_required": True,
        "entries": entries,
    }
    write_json(output_path, artifact)
    report = {
        "schema_version": 1,
        "task": "entity_glossary_proposal",
        "proposed_entries": len(entries),
        "approved_entries": 0,
        "minimum_occurrences": minimum_occurrences,
        "mining_policy": "case_insensitive_rare_terms",
        "corpus_messages": len(messages),
        "rare_document_ceiling": rare_ceiling,
        "message_text_persisted_in_report": False,
        "message_text_persisted_only_in_private_glossary": True,
    }
    write_json(report_path, report)
    return report


def load_glossary(path: str | Path | None, *, approved_only: bool = True) -> list[dict[str, Any]]:
    if path is None:
        return []
    glossary_path = Path(path)
    if not glossary_path.exists():
        return []
    value = json.loads(glossary_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != GLOSSARY_SCHEMA_VERSION:
        raise ValueError("Entity glossary has an incompatible schema version")
    entries = value.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Entity glossary entries must be a list")
    validated: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Entity glossary entry must be an object")
        term = entry.get("term")
        if not isinstance(term, str) or not term.strip():
            raise ValueError("Entity glossary entry requires a term")
        if entry.get("entry_id") != _entry_id(term):
            raise ValueError("Entity glossary entry has an incompatible identifier")
        approved = entry.get("approved") is True
        definition = entry.get("definition")
        evidence_ids = entry.get("evidence_message_ids")
        if approved and (
            not isinstance(definition, str)
            or not definition.strip()
            or not isinstance(evidence_ids, list)
            or not evidence_ids
        ):
            raise ValueError("Approved glossary entries require a definition and evidence")
        if not approved_only or approved:
            validated.append(dict(entry))
    return validated


def create_entity_glossary_review(
    glossary_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    entries = load_glossary(glossary_path, approved_only=False)
    lines = [
        "# Private Entity Glossary Review",
        "",
        "Edit the private glossary JSON to supply a concise definition and set `approved` to true.",
        "Approve only definitions supported by the cited evidence.",
        "",
    ]
    for index, entry in enumerate(entries, start=1):
        lines.extend(
            [
                f"## {index}. {html.escape(str(entry['term']))}",
                "",
                f"Occurrences: {int(entry.get('occurrences', 0))}",
                "",
                "**Proposed definition evidence**",
                f"<pre>{html.escape(str(entry.get('proposed_definition', '')))}</pre>",
                "",
            ]
        )
        for evidence in entry.get("evidence", []):
            if isinstance(evidence, Mapping):
                lines.extend(
                    [
                        f"- `{html.escape(str(evidence.get('message_id', '')))}`",
                        f"  <pre>{html.escape(str(evidence.get('text', '')))}</pre>",
                    ]
                )
        lines.extend(["", "- [ ] Definition is evidence-supported", ""])
    atomic_write_text(output_path, "\n".join(lines) + "\n")
    return {
        "schema_version": 1,
        "task": "private_entity_glossary_review",
        "entries": len(entries),
        "private_review": str(Path(output_path)),
        "message_text_persisted_in_summary": False,
    }


def glossary_fingerprint(entries: Iterable[Mapping[str, Any]]) -> str:
    canonical = [
        {
            "entry_id": str(entry["entry_id"]),
            "term": str(entry["term"]),
            "definition": str(entry.get("definition", "")),
            "approved": entry.get("approved") is True,
            "valid_from_timestamp_ns": int(entry.get("valid_from_timestamp_ns", 0) or 0),
            "evidence_message_ids": [str(value) for value in entry.get("evidence_message_ids", [])],
        }
        for entry in entries
    ]
    return sha256_text(json.dumps(canonical, sort_keys=True, ensure_ascii=False))
