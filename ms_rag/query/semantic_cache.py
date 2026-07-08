"""Session-scoped semantic answer cache.

Keys answered questions by their embedding so a repeated or near-duplicate
question is served from cache — regardless of how many queries ago it was
asked — skipping retrieval and generation entirely.

Session-scoped and in-memory: the vector index is fixed for the life of a
session, so a cached answer for a repeated corpus question stays correct (no
staleness). It is intentionally NOT persisted across sessions, where a
re-ingest could invalidate answers.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class _CacheEntry:
    embedding: list[float]
    query: str
    answer: str


@dataclass
class CacheHit:
    answer: str
    matched_query: str
    similarity: float


class SemanticQueryCache:
    """In-memory embedding-similarity cache of (query -> answer).

    A high default threshold (0.97) means only near-identical questions are
    served from cache, so semantically different questions are never confused.
    """

    def __init__(
        self,
        embeddings: object,
        *,
        threshold: float = 0.97,
        max_entries: int = 256,
    ) -> None:
        self.embeddings = embeddings
        self.threshold = float(threshold)
        self.max_entries = int(max_entries)
        self._entries: list[_CacheEntry] = []

    def _embed(self, text: str) -> list[float] | None:
        embed_query = getattr(self.embeddings, "embed_query", None)
        if not callable(embed_query):
            return None
        try:
            return [float(x) for x in embed_query(text)]
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Semantic cache could not embed the query; skipping cache for this turn: {exc}",
                stacklevel=2,
            )
            return None

    def lookup(self, query: str) -> CacheHit | None:
        """Return the best cached answer at/above threshold, or None."""
        if not self._entries:
            return None
        vector = self._embed(query)
        if vector is None:
            return None
        best: _CacheEntry | None = None
        best_sim = -1.0
        for entry in self._entries:
            sim = _cosine(vector, entry.embedding)
            if sim > best_sim:
                best_sim = sim
                best = entry
        if best is not None and best_sim >= self.threshold:
            return CacheHit(answer=best.answer, matched_query=best.query, similarity=best_sim)
        return None

    def add(self, query: str, answer: str) -> None:
        """Store an answered question. Embeds the query for future lookups."""
        query = str(query or "").strip()
        answer = str(answer or "").strip()
        if not query or not answer:
            return
        vector = self._embed(query)
        if vector is None:
            return
        # Replace an existing near-identical entry rather than duplicating it.
        for entry in self._entries:
            if _cosine(vector, entry.embedding) >= self.threshold:
                entry.answer = answer
                entry.query = query
                return
        self._entries.append(_CacheEntry(embedding=vector, query=query, answer=answer))
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries :]

    def clear(self) -> None:
        self._entries.clear()
