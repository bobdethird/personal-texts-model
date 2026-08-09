from pathlib import Path
from typing import Any

import numpy as np

from imessage_mlx.data.pairs import PAIR_FORMAT
from imessage_mlx.retrieval import RetrievalIndex, build_retrieval_index
from imessage_mlx.utils import write_jsonl


class FakeEncoder:
    def __init__(self, values: dict[str, list[float]]) -> None:
        self.values = values

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        assert kwargs["convert_to_numpy"] is True
        assert kwargs["normalize_embeddings"] is True
        return np.asarray([self.values[text] for text in texts], dtype=np.float32)


def _pair(
    pair_id: str,
    session_id: str,
    query: str,
    reply: str,
    context_query: str | None = None,
) -> dict[str, str]:
    return {
        "format": PAIR_FORMAT,
        "pair_id": pair_id,
        "session_id": session_id,
        "split": "train",
        "query": query,
        "context_query": context_query or query,
        "reply": reply,
    }


def test_build_load_and_search_excludes_same_session(tmp_path: Path) -> None:
    pairs_path = tmp_path / "pairs.jsonl"
    write_jsonl(
        pairs_path,
        [
            _pair("a", "session-a", "dinner?", "yes"),
            _pair("b", "session-b", "movie?", "sure"),
            _pair("c", "session-c", "later?", "okay"),
        ],
    )
    encoder = FakeEncoder(
        {
            "dinner?": [1.0, 0.0],
            "movie?": [0.8, 0.6],
            "later?": [0.0, 1.0],
        }
    )

    manifest = build_retrieval_index(
        pairs_path,
        tmp_path / "index",
        model_name="fake-model",
        encoder=encoder,
    )
    index = RetrievalIndex.load(tmp_path / "index")
    results = index.search(
        np.asarray([1.0, 0.0], dtype=np.float32),
        top_k=2,
        exclude_session_id="session-a",
    )

    assert manifest["count"] == 3
    assert manifest["dimensions"] == 2
    assert [result["pair_id"] for result in results] == ["b", "c"]
    assert results[0]["score"] > results[1]["score"]


def test_index_embeds_context_query_not_the_bare_message(tmp_path: Path) -> None:
    pairs_path = tmp_path / "pairs.jsonl"
    write_jsonl(
        pairs_path,
        [
            _pair("a", "session-a", "wdym", "about the retainer", context_query="Them: wdym"),
        ],
    )
    encoder = FakeEncoder({"Them: wdym": [1.0, 0.0], "wdym": [0.0, 1.0]})

    manifest = build_retrieval_index(
        pairs_path,
        tmp_path / "index",
        model_name="fake-model",
        encoder=encoder,
    )
    index = RetrievalIndex.load(tmp_path / "index")

    assert manifest["query_field"] == "context_query"
    assert index.metadata[0]["context_query"] == "Them: wdym"
    # The stored embedding matches the context text, not the bare message.
    assert np.allclose(index.embeddings[0], np.asarray([1.0, 0.0], dtype=np.float32))


def test_index_preserves_context_messages_for_demonstrations(tmp_path: Path) -> None:
    pairs_path = tmp_path / "pairs.jsonl"
    pair = _pair("a", "session-a", "wdym", "about the retainer", context_query="Them: wdym")
    pair["context_messages"] = [
        {"role": "assistant", "content": "notion is good for medical stuff"},
        {"role": "user", "content": "wdym"},
    ]
    write_jsonl(pairs_path, [pair])
    encoder = FakeEncoder({"Them: wdym": [1.0, 0.0]})

    build_retrieval_index(pairs_path, tmp_path / "index", model_name="fake-model", encoder=encoder)
    index = RetrievalIndex.load(tmp_path / "index")

    assert index.metadata[0]["context_messages"] == [
        {"role": "assistant", "content": "notion is good for medical stuff"},
        {"role": "user", "content": "wdym"},
    ]


def test_search_is_stable_when_scores_tie() -> None:
    index = RetrievalIndex(
        metadata=[
            {"pair_id": "first", "session_id": "one"},
            {"pair_id": "second", "session_id": "two"},
        ],
        embeddings=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        model_name="fake",
    )

    results = index.search([1.0, 0.0], top_k=2)

    assert [result["pair_id"] for result in results] == ["first", "second"]
