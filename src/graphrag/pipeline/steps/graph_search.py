from __future__ import annotations

import logging
from typing import Protocol

from graphrag.observability import Trace
from graphrag.retrieval.global_search import GlobalSearch, KeyPoint
from graphrag.retrieval.local_search import LocalContext, LocalSearch

_trace = Trace(__name__)


class GlobalSearchContext(Protocol):
    question: str
    key_points: list[KeyPoint]
    answer: str


class GlobalSearchStep:
    title = "Global search (map-reduce over reports)"

    def __init__(self, search: GlobalSearch) -> None:
        self._search = search

    def __call__(self, context: GlobalSearchContext) -> None:
        _trace.field("question", context.question)
        context.key_points = self._search.map(context.question)
        _trace.field("map", f"{len(context.key_points)} key points from the community reports")
        _trace.table(
            "points",
            [{"score": p.score, "point": p.description} for p in context.key_points],
            max_width=90,
            level=logging.DEBUG,
        )
        context.answer = self._search.reduce(context.question, context.key_points)
        _trace.field("answer", context.answer)


class LocalSearchContext(Protocol):
    question: str
    local_context: LocalContext
    answer: str


class LocalSearchStep:
    title = "Local search (entities + their neighbourhood)"

    def __init__(self, search: LocalSearch) -> None:
        self._search = search

    def __call__(self, context: LocalSearchContext) -> None:
        _trace.field("question", context.question)
        local = self._search.build_context(context.question)
        context.local_context = local
        _trace.field(
            "context",
            f"{len(local.entities)} entities, {len(local.chunks)} chunks, "
            f"{len(local.reports)} reports, {len(local.relationships)} relationships",
        )
        parts = {
            "Entities": local.entities,
            "Chunks": local.chunks,
            "Reports": local.reports,
            "Relationships": local.relationships,
        }
        _trace.block(
            "",
            "\n\n".join(
                f"{name}:\n" + "\n".join("• " + item.replace("\n", "\n  ") for item in items)
                for name, items in parts.items()
            ),
            level=logging.DEBUG,
        )
        context.answer = self._search.answer(context.question, local)
        _trace.field("answer", context.answer)
