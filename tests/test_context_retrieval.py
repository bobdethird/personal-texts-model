import json
from pathlib import Path

from imessage_mlx.data.context_retrieval import TemporalMessageIndex
from imessage_mlx.data.entity_glossary import (
    create_entity_glossary_review,
    load_glossary,
    propose_entity_glossary,
)
from imessage_mlx.utils import sha256_text, write_json, write_jsonl


def _message(
    message_id: str,
    timestamp_ns: int,
    text: str,
    *,
    chat_id: str = "chat",
    role: str = "me",
) -> dict[str, object]:
    return {
        "message_id": message_id,
        "chat_id": chat_id,
        "timestamp_ns": timestamp_ns,
        "sender_role": role,
        "participant_id": role,
        "text": text,
        "has_attachment": False,
        "is_group": False,
    }


def test_temporal_search_crosses_chats_but_never_uses_future_messages() -> None:
    index = TemporalMessageIndex(
        [
            _message(
                "prior",
                1,
                "Cercor is my AI agent project and it calls tools",
                chat_id="project-chat",
            ),
            _message("future", 30, "Cercor uses browser tools now", chat_id="other-chat"),
        ]
    )

    results = index.search("my Cercor uses tool calls", before_timestamp_ns=20)

    assert [row["message_id"] for row in results] == ["prior"]
    assert results[0]["chat_id"] == "project-chat"
    assert results[0]["relation"] == "historical_retrieval"


def test_short_slang_tokens_are_searchable() -> None:
    index = TemporalMessageIndex(
        [
            _message("prior", 1, "ts means this stuff when we text"),
            _message("noise", 2, "completely unrelated message"),
        ]
    )

    results = index.search("Post it ts tuff", before_timestamp_ns=10)

    assert [row["message_id"] for row in results] == ["prior"]


def test_glossary_proposals_require_human_definition_and_approval(tmp_path: Path) -> None:
    messages = tmp_path / "messages.jsonl"
    glossary = tmp_path / "glossary.json"
    write_jsonl(
        messages,
        [
            _message("one", 1, "Cercor is my AI agent project"),
            _message("two", 2, "Cercor uses tool calls"),
        ],
    )

    report = propose_entity_glossary(
        messages,
        glossary,
        tmp_path / "report.json",
        minimum_occurrences=2,
    )
    artifact = json.loads(glossary.read_text(encoding="utf-8"))
    entry = next(value for value in artifact["entries"] if value["term"] == "Cercor")

    assert report["proposed_entries"] >= 1
    assert entry["approved"] is False
    assert load_glossary(glossary) == []

    entry["definition"] = "An AI-agent project developed by the author that invokes tools."
    entry["approved"] = True
    write_json(glossary, artifact)
    approved = load_glossary(glossary)
    assert approved[0]["entry_id"] == sha256_text("entity-glossary:1:cercor")[:24]

    review = create_entity_glossary_review(glossary, tmp_path / "review.md")
    assert review["entries"] >= 1
    assert "Cercor" in (tmp_path / "review.md").read_text(encoding="utf-8")


def test_glossary_mining_finds_lowercase_and_acronym_terms_but_skips_common_words(
    tmp_path: Path,
) -> None:
    messages = tmp_path / "messages.jsonl"
    rows = [
        _message(f"common-{index}", index + 1, f"see you tonight friend number {index}")
        for index in range(10)
    ]
    rows.extend(
        [
            _message("coined-1", 20, "cercor deploy went smoothly"),
            _message("coined-2", 21, "cercor is my agent project"),
            _message("acronym-1", 22, "the MLXQ build passed"),
            _message("acronym-2", 23, "MLXQ needs another retry"),
            _message("short-1", 24, "ik solver is broken"),
            _message("short-2", 25, "still tuning ik gains"),
        ]
    )
    write_jsonl(messages, rows)

    report = propose_entity_glossary(
        messages,
        tmp_path / "glossary.json",
        tmp_path / "report.json",
        minimum_occurrences=2,
    )
    artifact = json.loads((tmp_path / "glossary.json").read_text(encoding="utf-8"))
    terms = {str(entry["term"]).casefold() for entry in artifact["entries"]}

    assert "cercor" in terms
    assert "mlxq" in terms
    assert "ik" in terms
    assert "tonight" not in terms
    assert "friend" not in terms
    assert report["mining_policy"] == "case_insensitive_rare_terms"
    assert report["rare_document_ceiling"] >= 2
