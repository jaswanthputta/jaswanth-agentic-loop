"""Persistent Chroma vector memory with a lightweight deterministic embedding."""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from pathlib import Path
from typing import Any


def _embedding(text: str, dimensions: int = 384) -> list[float]:
    """Hash tokens and character trigrams into a normalized, stable vector."""
    vector = [0.0] * dimensions
    tokens = re.findall(r"[a-z0-9+#.-]+", text.lower())
    features = tokens + [
        token[index : index + 3]
        for token in tokens
        for index in range(max(0, len(token) - 2))
    ]
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % dimensions
        vector[index] += 1.0 if value & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector


class MemoryManager:
    """Session-scoped semantic recall backed by a persistent Chroma collection."""

    def __init__(
        self,
        persist_directory: str | Path,
        collection_name: str = "interview_question_agent",
        *,
        collection: Any | None = None,
    ) -> None:
        if collection is None:
            try:
                import chromadb
            except ImportError as error:
                raise RuntimeError(
                    "ChromaDB is required for persistent memory. Install requirements.txt."
                ) from error
            client = chromadb.PersistentClient(path=str(persist_directory))
            collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        self._collection = collection

    def save(self, session_id: str, content: str, metadata: dict[str, Any]) -> None:
        if not session_id.strip():
            raise ValueError("session_id must not be empty.")
        if not content.strip():
            raise ValueError("Memory content must not be empty.")
        safe_metadata: dict[str, str | int | float | bool] = {
            "session_id": session_id
        }
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                safe_metadata[key] = value
            else:
                safe_metadata[key] = str(value)
        self._collection.add(
            ids=[str(uuid.uuid4())],
            documents=[content],
            embeddings=[_embedding(content)],
            metadatas=[safe_metadata],
        )

    def recall(self, session_id: str, query: str, top_k: int = 3) -> list[str]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if not query.strip():
            return []
        response = self._collection.query(
            query_embeddings=[_embedding(query)],
            n_results=top_k,
            where={"session_id": session_id},
            include=["documents"],
        )
        documents = response.get("documents") or [[]]
        return [str(document) for document in documents[0] if document is not None]

    def clear(self, session_id: str) -> None:
        if not session_id.strip():
            raise ValueError("session_id must not be empty.")
        self._collection.delete(where={"session_id": session_id})
