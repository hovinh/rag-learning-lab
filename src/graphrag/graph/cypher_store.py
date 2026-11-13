from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from graphrag.graph.models import GraphSchema


class CypherGraphStore(ABC):
    """Arbitrary-Cypher interface over a structured graph (ch04
    text2cypher): run a query, and describe the schema so an LLM can write
    queries against it.

    Callers depend on this, not on a specific database client, so the
    backend (Kuzu today) can be swapped without touching them.
    """

    @abstractmethod
    def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Run ``cypher`` and return one dict per row, keyed by column name.

        Raises ``GraphQueryError`` if the database rejects the query.
        """

    @abstractmethod
    def get_schema(self) -> GraphSchema: ...
