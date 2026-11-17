from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RagAnswer:
    """A recipe's answer, with the retrieved contexts it was
    written from -- the shape ``BenchmarkRunner`` evaluates, so any
    recipe's ``query`` can be benchmarked.
    """

    question: str
    answer: str
    contexts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CypherAnswer:
    """text2cypher's result: the generated query and the rows it returned."""

    question: str
    cypher: str
    rows: list[dict[str, Any]]
