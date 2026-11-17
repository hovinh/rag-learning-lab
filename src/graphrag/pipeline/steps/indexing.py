from __future__ import annotations

from typing import Protocol

from graphrag.graph.parent_child_store import ParentChildGraphStore
from graphrag.graph.store import GraphStore
from graphrag.ingestion.models import Chunk
from graphrag.llm.hypothetical_questions import HypotheticalQuestionGenerator
from graphrag.observability import Trace
from graphrag.retrieval.embedder import Embedder

_trace = Trace(__name__)


class EmbeddedChunksContext(Protocol):
    chunks: list[Chunk]
    embeddings: list[list[float]]


class EmbedChunksStep:
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def __call__(self, context: EmbeddedChunksContext) -> None:
        context.embeddings = self._embedder.embed([chunk.text for chunk in context.chunks])
        _trace.field("embedded", f"{len(context.embeddings)} chunks")


class StoreChunksStep:
    """Replaces ``store``'s whole chunk corpus with the embedded chunks
    (ch02).
    """

    title = "Store chunks"

    def __init__(self, store: GraphStore) -> None:
        self._store = store

    def __call__(self, context: EmbeddedChunksContext) -> None:
        self._store.replace_chunks([chunk.text for chunk in context.chunks], context.embeddings)
        _trace.field("stored", f"{len(context.chunks)} chunks")


class QuestionsContext(Protocol):
    chunks: list[Chunk]
    parents: list[str]
    children_per_parent: list[list[str]]


class GenerateQuestionsStep:
    """Hypothetical question embedding: each chunk becomes a parent whose
    children are the questions it answers, so the parent/child steps
    after it embed the questions and retrieval hands back the chunk.
    """

    title = "Generate hypothetical questions"

    def __init__(self, generator: HypotheticalQuestionGenerator) -> None:
        self._generator = generator

    def __call__(self, context: QuestionsContext) -> None:
        context.parents = [chunk.text for chunk in context.chunks]
        progress = _trace.progress("chunks", len(context.parents))
        questions_per_parent = []
        for parent in context.parents:
            questions = self._generator.generate(parent)
            questions_per_parent.append(questions)
            progress.advance(f"{len(questions)} questions")
        context.children_per_parent = questions_per_parent
        total = sum(len(questions) for questions in questions_per_parent)
        _trace.field("questions", f"{total} for {len(context.parents)} chunks")
        if questions_per_parent:
            _trace.bullets("first chunk's questions", questions_per_parent[0])


class EmbeddedChildrenContext(Protocol):
    parents: list[str]
    children_per_parent: list[list[str]]
    child_embeddings_per_parent: list[list[list[float]]]


class EmbedChildrenStep:
    """Embeds only the child chunks (ch03): parents are retrieved through
    their children, never matched themselves.
    """

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def __call__(self, context: EmbeddedChildrenContext) -> None:
        context.child_embeddings_per_parent = [
            self._embedder.embed(children) for children in context.children_per_parent
        ]
        embedded = sum(len(vectors) for vectors in context.child_embeddings_per_parent)
        _trace.field("embedded", f"{embedded} children")


class StoreParentChildStep:
    """Replaces ``store``'s whole parent/child corpus (ch03)."""

    title = "Store the parent/child hierarchy"

    def __init__(self, store: ParentChildGraphStore) -> None:
        self._store = store

    def __call__(self, context: EmbeddedChildrenContext) -> None:
        self._store.replace_parents(
            context.parents, context.children_per_parent, context.child_embeddings_per_parent
        )
        _trace.field("stored", f"{len(context.parents)} parents")
