from __future__ import annotations

from typing import Any, Protocol

from graphrag.graph.cypher_store import CypherGraphStore
from graphrag.graph.text2cypher import Text2Cypher
from graphrag.observability import Trace

_trace = Trace(__name__)


class CypherContext(Protocol):
    question: str
    cypher: str


class GenerateCypherStep:
    title = "Generate Cypher (text2cypher)"

    def __init__(self, text2cypher: Text2Cypher) -> None:
        self._text2cypher = text2cypher

    def __call__(self, context: CypherContext) -> None:
        _trace.field("question", context.question)
        context.cypher = self._text2cypher.generate(context.question)
        _trace.field("cypher", context.cypher)


class CypherRowsContext(Protocol):
    cypher: str
    rows: list[dict[str, Any]]


class ExecuteCypherStep:
    """Runs the generated Cypher -- the book's ch04 notebook only prints
    it, but executing it is what shows the query is valid against the graph.
    """

    title = "Run the Cypher"

    def __init__(self, store: CypherGraphStore) -> None:
        self._store = store

    def __call__(self, context: CypherRowsContext) -> None:
        context.rows = self._store.query(context.cypher)
        _trace.table("rows", context.rows)
