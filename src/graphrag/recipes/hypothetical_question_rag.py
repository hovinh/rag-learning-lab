from __future__ import annotations

from dataclasses import dataclass, field

from graphrag.graph.parent_child_store import ParentChildGraphStore
from graphrag.ingestion.chunker import TextChunker
from graphrag.ingestion.models import Chunk
from graphrag.ingestion.sources import DocumentSource
from graphrag.llm.client import LLMClient, OpenAILLMClient
from graphrag.llm.hypothetical_questions import HypotheticalQuestionGenerator
from graphrag.pipeline import (
    AnswerStep,
    ChunkStep,
    EmbedChildrenStep,
    GenerateQuestionsStep,
    LoadDocumentsStep,
    Pipeline,
    RetrieveStep,
    StoreParentChildStep,
)
from graphrag.recipes._contexts import AnswerContext
from graphrag.recipes.models import RagAnswer
from graphrag.retrieval.embedder import Embedder
from graphrag.retrieval.retrievers import ParentChildRetriever


@dataclass
class _QuestionIndexContext:
    documents: list[str] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)
    children_per_parent: list[list[str]] = field(default_factory=list)
    child_embeddings_per_parent: list[list[list[float]]] = field(default_factory=list)


class HypotheticalQuestionRag:
    """Hypothetical question embedding as a recipe: the LLM writes the
    questions each chunk answers, steered by ``purpose`` (what the index
    is for), and only those questions are embedded. A user's question is
    matched against them and the LLM is handed the chunks they came from.

    It reuses ch03's parent/child store, with the chunk as the parent and
    its questions as the children. Indexing costs one LLM call per chunk;
    ``index`` replaces the whole corpus, so later runs check
    ``is_indexed`` and go straight to ``query``.
    """

    def __init__(
        self,
        store: ParentChildGraphStore,
        embedder: Embedder,
        llm: LLMClient,
        *,
        purpose: str,
        questions_per_chunk: int = 5,
        chunker: TextChunker | None = None,
        k: int = 4,
        model: str = OpenAILLMClient.DEFAULT_MODEL,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._llm = llm
        self._chunker = chunker or TextChunker(chunk_size=2000, overlap=40)
        self._generator = HypotheticalQuestionGenerator(
            llm, purpose=purpose, questions_per_chunk=questions_per_chunk, model=model
        )
        self._retriever = ParentChildRetriever(store, embedder, k=k)
        self._model = model

    def is_indexed(self) -> bool:
        return self._store.parent_count() > 0

    def index(self, source: DocumentSource) -> None:
        indexing: Pipeline[_QuestionIndexContext] = Pipeline(
            [
                LoadDocumentsStep(source),
                ChunkStep(self._chunker),
                GenerateQuestionsStep(self._generator),
                EmbedChildrenStep(self._embedder),
                StoreParentChildStep(self._store),
            ]
        )
        indexing.run(_QuestionIndexContext())

    def query(self, question: str) -> RagAnswer:
        answering: Pipeline[AnswerContext] = Pipeline(
            [
                RetrieveStep(self._retriever, label="hypothetical question search"),
                AnswerStep(self._llm, model=self._model),
            ]
        )
        return answering.run(AnswerContext(question)).to_answer()
