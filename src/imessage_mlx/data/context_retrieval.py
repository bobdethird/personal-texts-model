from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from imessage_mlx.data.rewrite import STRUCTURAL_TOKEN_RE

CONTEXT_BUNDLE_SCHEMA_VERSION = 2
RETRIEVER_VERSION = "temporal-bm25-v2-short-tokens"
TOKEN_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "because",
    "been",
    "before",
    "but",
    "can",
    "could",
    "did",
    "does",
    "for",
    "from",
    "get",
    "got",
    "have",
    "how",
    "just",
    "like",
    "not",
    "that",
    "the",
    "their",
    "then",
    "there",
    "they",
    "this",
    "too",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "would",
    "you",
    "your",
}


def retrieval_tokens(text: str) -> tuple[str, ...]:
    # Two-character slang and initialisms such as "ts" or "IK" must stay searchable; the
    # per-query rarity band keeps ubiquitous short words from ever becoming query terms.
    return tuple(
        token.casefold()
        for token in TOKEN_RE.findall(text)
        if len(token) >= 2 and token.casefold() not in STOPWORDS
    )


class TemporalMessageIndex:
    def __init__(self, messages: Iterable[Mapping[str, Any]]) -> None:
        self.documents: list[dict[str, Any]] = []
        self.tokens: list[Counter[str]] = []
        self.lengths: list[int] = []
        self.postings: dict[str, list[int]] = defaultdict(list)
        for message in messages:
            text = str(message.get("text", "")).strip()
            if not text or STRUCTURAL_TOKEN_RE.search(text):
                continue
            tokens = Counter(retrieval_tokens(text))
            if not tokens:
                continue
            index = len(self.documents)
            self.documents.append(dict(message))
            self.tokens.append(tokens)
            self.lengths.append(sum(tokens.values()))
            for token in tokens:
                self.postings[token].append(index)
        self.document_count = len(self.documents)
        self.average_length = (
            sum(self.lengths) / self.document_count if self.document_count else 1.0
        )

    def _idf(self, token: str) -> float:
        frequency = len(self.postings.get(token, ()))
        if not frequency:
            return 0.0
        return math.log(1.0 + (self.document_count - frequency + 0.5) / (frequency + 0.5))

    def query_terms(self, query: str, *, maximum: int = 6) -> tuple[str, ...]:
        unique = set(retrieval_tokens(query))
        if not unique:
            return ()
        rare_limit = max(20, min(500, int(self.document_count * 0.002)))
        distinctive = [
            token
            for token in unique
            if self.postings.get(token) and len(self.postings[token]) <= rare_limit
        ]
        return tuple(sorted(distinctive, key=lambda token: (-self._idf(token), token))[:maximum])

    def plan_queries(self, query: str, *, maximum: int = 6) -> list[str]:
        return list(self.query_terms(query, maximum=maximum))

    def search(
        self,
        query: str,
        *,
        before_timestamp_ns: int,
        exclude_message_ids: Iterable[str] = (),
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        terms = self.query_terms(query)
        if not terms:
            return []
        excluded = {str(value) for value in exclude_message_ids}
        candidates = {
            document_index for term in terms for document_index in self.postings.get(term, ())
        }
        scored: list[tuple[float, int]] = []
        k1 = 1.5
        b = 0.75
        for index in candidates:
            document = self.documents[index]
            if int(document["timestamp_ns"]) >= before_timestamp_ns:
                continue
            if str(document["message_id"]) in excluded:
                continue
            frequencies = self.tokens[index]
            score = 0.0
            for term in terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + k1 * (
                    1.0 - b + b * self.lengths[index] / self.average_length
                )
                score += self._idf(term) * frequency * (k1 + 1.0) / denominator
            if score > 0:
                scored.append((score, index))
        selected: list[dict[str, Any]] = []
        seen_text: set[str] = set()
        for score, index in sorted(
            scored,
            key=lambda value: (
                -value[0],
                -int(self.documents[value[1]]["timestamp_ns"]),
                str(self.documents[value[1]]["message_id"]),
            ),
        ):
            document = self.documents[index]
            normalized = " ".join(str(document["text"]).casefold().split())
            if normalized in seen_text:
                continue
            seen_text.add(normalized)
            matched_terms = [term for term in terms if term in self.tokens[index]]
            selected.append(
                {
                    "message_id": str(document["message_id"]),
                    "chat_id": str(document["chat_id"]),
                    "timestamp_ns": int(document["timestamp_ns"]),
                    "role": str(document["sender_role"]),
                    "participant_id": str(document["participant_id"]),
                    "text": str(document["text"]).strip(),
                    "relation": "historical_retrieval",
                    "retrieval_score": round(score, 6),
                    "retrieval_reason": "matched:" + ",".join(matched_terms),
                }
            )
            if len(selected) == limit:
                break
        return selected


def matched_glossary_entries(
    text: str,
    entries: Iterable[Mapping[str, Any]],
    *,
    target_timestamp_ns: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for entry in entries:
        term = str(entry.get("term", "")).strip()
        definition = str(entry.get("definition", "")).strip()
        valid_from = int(entry.get("valid_from_timestamp_ns", 0) or 0)
        if (
            entry.get("approved") is not True
            or not term
            or not definition
            or valid_from >= target_timestamp_ns
            or re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.IGNORECASE) is None
        ):
            continue
        selected.append(
            {
                "entry_id": str(entry["entry_id"]),
                "term": term,
                "definition": definition,
                "evidence_message_ids": [
                    str(value) for value in entry.get("evidence_message_ids", [])
                ],
                "valid_from_timestamp_ns": valid_from,
                "approved": True,
            }
        )
    return sorted(selected, key=lambda value: value["term"].casefold())
