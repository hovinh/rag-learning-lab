from __future__ import annotations

from typing import Literal

from graphrag.graph.store import GraphStore
from graphrag.ingestion.chunker import TextChunker
from graphrag.ingestion.sources import DocumentSource
from graphrag.llm.client import LLMClient, OpenAILLMClient
from graphrag.pipeline import (
    AnswerStep,
    ChunkStep,
    EmbedChunksStep,
    LoadDocumentsStep,
    Pipeline,
    RetrieveStep,
    StoreChunksStep,
)
from graphrag.pipelines._contexts import AnswerContext, ChunkIndexContext
from graphrag.pipelines.models import RagAnswer
from graphrag.retrieval.embedder import Embedder
from graphrag.retrieval.retriever import Retriever
from graphrag.retrieval.retrievers import HybridRetriever, VectorRetriever

SearchMode = Literal["vector", "hybrid"]


class VectorRag:
    """ch02 as a reusable pipeline: chunk and embed documents into a
    ``GraphStore`` once, then answer questions over them with vector or
    hybrid (vector + keyword) search.

    ``index`` replaces the whole corpus; the store keeps it, so later runs
    check ``is_indexed`` and go straight to ``query``.
    """

    def __init__(
        self,
        store: GraphStore,
        embedder: Embedder,
        llm: LLMClient,
        *,
        chunker: TextChunker | None = None,
        k: int = 4,
        model: str = OpenAILLMClient.DEFAULT_MODEL,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._llm = llm
        self._chunker = chunker or TextChunker(chunk_size=500, overlap=40)
        self._retrievers: dict[SearchMode, Retriever] = {
            "vector": VectorRetriever(store, embedder, k=k),
            "hybrid": HybridRetriever(store, embedder, k=k),
        }
        self._model = model

    def is_indexed(self) -> bool:
        return self._store.chunk_count() > 0

    def index(self, source: DocumentSource) -> None:
        indexing: Pipeline[ChunkIndexContext] = Pipeline(
            [
                LoadDocumentsStep(source),
                ChunkStep(self._chunker),
                EmbedChunksStep(self._embedder),
                StoreChunksStep(self._store),
            ]
        )
        indexing.run(ChunkIndexContext())

    def query(self, question: str, *, search: SearchMode = "vector") -> RagAnswer:
        answering: Pipeline[AnswerContext] = Pipeline(
            [
                RetrieveStep(self._retrievers[search], label=f"{search} search"),
                AnswerStep(self._llm, model=self._model),
            ]
        )
        return answering.run(AnswerContext(question)).to_answer()
