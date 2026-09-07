from __future__ import annotations

from abc import ABC, abstractmethod

from graphrag.graph.models import ChunkMatch


class GraphStore(ABC):
    """Storage + retrieval interface for text chunks and their embeddings.

    Callers depend on this, not on a specific database client, so the
    backend (Kuzu today) can be swapped without touching them.
    """

    @abstractmethod
    def replace_chunks(self, chunks: list[str], embeddings: list[list[float]]) -> None:
        """Replace the entire chunk corpus with the given chunks/embeddings."""

    @abstractmethod
    def vector_search(self, query_embedding: list[float], k: int) -> list[ChunkMatch]: ...

    @abstractmethod
    def keyword_search(self, query: str, k: int) -> list[ChunkMatch]: ...

    def hybrid_search(self, query: str, query_embedding: list[float], k: int) -> list[ChunkMatch]:
        """Combine vector + keyword search, matching the book's Neo4j
        ``hybrid_query`` (ch02): normalize each result set to its own max
        score, then keep the best normalized score per chunk.
        """
        vector_hits = self.vector_search(query_embedding, k)
        keyword_hits = self.keyword_search(query, k)
        return self._merge_normalized(vector_hits, keyword_hits, k=k)

    @staticmethod
    def _merge_normalized(*result_sets: list[ChunkMatch], k: int) -> list[ChunkMatch]:
        best: dict[int, ChunkMatch] = {}
        for hits in result_sets:
            if not hits:
                continue
            max_score = max(hit.score for hit in hits) or 1.0
            for hit in hits:
                normalized = ChunkMatch(hit.index, hit.text, hit.score / max_score)
                existing = best.get(hit.index)
                if existing is None or normalized.score > existing.score:
                    best[hit.index] = normalized
        return sorted(best.values(), key=lambda m: m.score, reverse=True)[:k]
