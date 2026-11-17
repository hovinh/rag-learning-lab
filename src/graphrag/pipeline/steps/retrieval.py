from __future__ import annotations

from typing import Protocol

from graphrag.observability import Trace
from graphrag.retrieval.retriever import Retriever, ScoredMatch

_trace = Trace(__name__)


class RetrieveContext(Protocol):
    question: str
    matches: list[ScoredMatch]


class RetrieveStep:
    """Fills ``context.matches`` from any ``Retriever``. ``label`` names
    the retrieval in the step's section title (``Retrieve (hybrid
    search)``).
    """

    def __init__(self, retriever: Retriever, *, label: str | None = None) -> None:
        self._retriever = retriever
        self.title = f"Retrieve ({label})" if label else "Retrieve"

    def __call__(self, context: RetrieveContext) -> None:
        context.matches = list(self._retriever.retrieve(context.question))
        _trace.field("retrieved", f"{len(context.matches)} matches")
