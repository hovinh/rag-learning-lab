from __future__ import annotations

from collections.abc import Sequence

from graphrag.graph.models import ChunkMatch, ParentMatch
from graphrag.graph.parent_child_store import ParentChildGraphStore
from graphrag.graph.store import GraphStore
from graphrag.llm.query_rewriter import QueryRewriter
from graphrag.observability import Trace
from graphrag.retrieval.embedder import Embedder
from graphrag.retrieval.retriever import Retriever, ScoredMatch

_trace = Trace(__name__)


class VectorRetriever(Retriever):
    """Embeds the question and vector-searches the chunks (ch02)."""

    def __init__(self, store: GraphStore, embedder: Embedder, *, k: int = 4) -> None:
        self._store = store
        self._embedder = embedder
        self._k = k

    def retrieve(self, question: str) -> list[ChunkMatch]:
        [embedding] = self._embedder.embed([question])
        return self._store.vector_search(embedding, k=self._k)


class HybridRetriever(Retriever):
    """Vector search plus keyword search over the chunks, merged by
    ``GraphStore.hybrid_search`` (ch02).
    """

    def __init__(self, store: GraphStore, embedder: Embedder, *, k: int = 4) -> None:
        self._store = store
        self._embedder = embedder
        self._k = k

    def retrieve(self, question: str) -> list[ChunkMatch]:
        [embedding] = self._embedder.embed([question])
        return self._store.hybrid_search(question, embedding, k=self._k)


class ParentChildRetriever(Retriever):
    """Matches the question against child chunks and returns their
    parents (ch03).
    """

    def __init__(self, store: ParentChildGraphStore, embedder: Embedder, *, k: int = 4) -> None:
        self._store = store
        self._embedder = embedder
        self._k = k

    def retrieve(self, question: str) -> list[ParentMatch]:
        [embedding] = self._embedder.embed([question])
        return self._store.parent_retrieval(embedding, k=self._k)


class StepBackRetriever(Retriever):
    """Rewrites the question into a broader step-back question (ch03) and
    retrieves with that instead. Only retrieval sees the rewrite: the
    answer is still written for the original question.
    """

    def __init__(self, rewriter: QueryRewriter, retriever: Retriever) -> None:
        self._rewriter = rewriter
        self._retriever = retriever

    def retrieve(self, question: str) -> Sequence[ScoredMatch]:
        step_back = self._rewriter.step_back(question)
        _trace.field("step-back", step_back)
        return self._retriever.retrieve(step_back)
