from __future__ import annotations

from abc import ABC, abstractmethod

from graphrag.graph.models import ParentMatch


class ParentChildGraphStore(ABC):
    """Storage + retrieval interface for hierarchical (parent/child)
    chunks (ch03): children are embedded and searched, but retrieval
    returns their parent's full text, so the LLM gets more surrounding
    context than a single small chunk would give it.

    Callers depend on this, not on a specific database client, so the
    backend (Kuzu today) can be swapped without touching them.
    """

    @abstractmethod
    def replace_parents(
        self,
        parents: list[str],
        children_per_parent: list[list[str]],
        child_embeddings_per_parent: list[list[list[float]]],
    ) -> None:
        """Replace the entire parent/child corpus."""

    @abstractmethod
    def parent_count(self) -> int:
        """How many parents are stored -- 0 until ``replace_parents``
        runs, so callers can index once and query on later runs.
        """

    @abstractmethod
    def parent_retrieval(self, query_embedding: list[float], k: int) -> list[ParentMatch]:
        """Vector-search over children, aggregated up to their parent."""
