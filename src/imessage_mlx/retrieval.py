from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from imessage_mlx.data.pairs import PAIR_FORMAT
from imessage_mlx.utils import (
    ensure_private_dir,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)

INDEX_FORMAT = "imessage-retrieval-v1"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Retrieval requires the personalization dependencies; "
            "run `uv sync --extra personalize`."
        ) from error
    return SentenceTransformer(model_name)


def encode_texts(
    encoder: Any,
    texts: list[str],
    *,
    batch_size: int = 64,
    show_progress: bool = False,
) -> Any:
    """Return normalized float32 NumPy embeddings from a SentenceTransformer-like encoder."""
    import numpy as np

    if not texts:
        raise ValueError("At least one text is required for embedding")
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    result = np.asarray(embeddings, dtype=np.float32)
    if result.ndim != 2 or result.shape[0] != len(texts):
        raise ValueError("Embedding model returned an unexpected shape")
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding model returned a zero vector")
    return result / norms


def _write_embeddings(path: Path, embeddings: Any) -> None:
    import numpy as np

    ensure_private_dir(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, embeddings, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def build_retrieval_index(
    train_pairs_path: str | Path,
    index_dir: str | Path,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    encoder: Any | None = None,
) -> dict[str, Any]:
    """Embed training queries and write a private, deterministic cosine index."""
    source_path = Path(train_pairs_path)
    pairs = list(read_jsonl(source_path))
    if not pairs:
        raise ValueError("Training pair file is empty")

    metadata: list[dict[str, Any]] = []
    for pair in pairs:
        if pair.get("format") != PAIR_FORMAT:
            raise ValueError(f"Unsupported pair format {pair.get('format')!r}")
        if pair.get("split") != "train":
            raise ValueError("Retrieval index may only contain training pairs")
        if not all(
            str(pair.get(field, "")).strip()
            for field in ("pair_id", "session_id", "query", "reply")
        ):
            raise ValueError("Training pairs must contain pair_id, session_id, query, and reply")
        # Retrieval matches on the multi-turn context so short messages are grounded;
        # the single-turn query is kept for display in demonstrations and reports.
        context_query = str(pair.get("context_query", "")).strip() or str(pair["query"]).strip()
        context_messages = [
            {"role": str(turn.get("role", "")), "content": str(turn.get("content", ""))}
            for turn in pair.get("context_messages", [])
            if str(turn.get("role", "")) in {"user", "assistant"}
            and str(turn.get("content", "")).strip()
        ]
        metadata.append(
            {
                "pair_id": str(pair["pair_id"]),
                "session_id": str(pair["session_id"]),
                "query": str(pair["query"]),
                "context_query": context_query,
                "context_messages": context_messages,
                "reply": str(pair["reply"]),
            }
        )

    embedding_model = encoder if encoder is not None else load_embedding_model(model_name)
    embeddings = encode_texts(
        embedding_model,
        [item["context_query"] for item in metadata],
        batch_size=batch_size,
        show_progress=True,
    )

    output_dir = ensure_private_dir(index_dir)
    embeddings_path = output_dir / "embeddings.npy"
    metadata_path = output_dir / "metadata.jsonl"
    _write_embeddings(embeddings_path, embeddings)
    write_jsonl(metadata_path, metadata)
    manifest = {
        "format": INDEX_FORMAT,
        "model_name": model_name,
        "source_path": str(source_path.resolve()),
        "source_sha256": sha256_file(source_path),
        "count": len(metadata),
        "dimensions": int(embeddings.shape[1]),
        "normalized": True,
        "query_field": "context_query",
        "embeddings_path": embeddings_path.name,
        "metadata_path": metadata_path.name,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


@dataclass
class RetrievalIndex:
    metadata: list[dict[str, Any]]
    embeddings: Any
    model_name: str

    @classmethod
    def load(cls, index_dir: str | Path) -> RetrievalIndex:
        import json

        import numpy as np

        directory = Path(index_dir)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") != INDEX_FORMAT:
            raise ValueError(f"Unsupported index format {manifest.get('format')!r}")
        metadata = list(read_jsonl(directory / str(manifest["metadata_path"])))
        embeddings = np.load(
            directory / str(manifest["embeddings_path"]),
            allow_pickle=False,
        )
        expected = (int(manifest["count"]), int(manifest["dimensions"]))
        if embeddings.shape != expected or len(metadata) != expected[0]:
            raise ValueError("Retrieval index files do not match the manifest")
        return cls(
            metadata=metadata,
            embeddings=np.asarray(embeddings, dtype=np.float32),
            model_name=str(manifest["model_name"]),
        )

    def search(
        self,
        query_embedding: Any,
        *,
        top_k: int,
        exclude_session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        import numpy as np

        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if self.embeddings.ndim != 2 or query.shape[0] != self.embeddings.shape[1]:
            raise ValueError("Query embedding dimension does not match the index")
        norm = float(np.linalg.norm(query))
        if norm == 0:
            raise ValueError("Query embedding must be non-zero")
        scores = self.embeddings @ (query / norm)
        ranked = np.argsort(-scores, kind="stable")

        results: list[dict[str, Any]] = []
        for index in ranked:
            item = self.metadata[int(index)]
            if exclude_session_id and item.get("session_id") == exclude_session_id:
                continue
            results.append({**item, "score": float(scores[int(index)])})
            if len(results) >= top_k:
                break
        return results

    def retrieve(
        self,
        query: str,
        encoder: Any,
        *,
        top_k: int,
        exclude_session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        embedding = encode_texts(encoder, [query])[0]
        return self.search(
            embedding,
            top_k=top_k,
            exclude_session_id=exclude_session_id,
        )
