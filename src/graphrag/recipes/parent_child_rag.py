from __future__ import annotations

from dataclasses import dataclass, field

from graphrag.graph.parent_child_store import ParentChildGraphStore
from graphrag.ingestion.chunker import TextChunker
from graphrag.ingestion.section_splitter import SectionSplitter
from graphrag.ingestion.sources import DocumentSource
from graphrag.llm.client import LLMClient, OpenAILLMClient
from graphrag.llm.query_rewriter import QueryRewriter
from graphrag.llm.token_counter import TokenCounter
from graphrag.pipeline import (
    AnswerStep,
    EmbedChildrenStep,
    LoadDocumentsStep,
    ParentChildChunkStep,
    Pipeline,
    PipelineStep,
    RetrieveStep,
    SplitSectionsStep,
    StoreParentChildStep,
)
from graphrag.recipes._contexts import AnswerContext
from graphrag.recipes.models import RagAnswer
from graphrag.retrieval.embedder import Embedder
from graphrag.retrieval.retriever import Retriever
from graphrag.retrieval.retrievers import ParentChildRetriever, StepBackRetriever


@dataclass
class _ParentChildIndexContext:
    documents: list[str] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)
    children_per_parent: list[list[str]] = field(default_factory=list)
    child_embeddings_per_parent: list[list[list[float]]] = field(default_factory=list)


class ParentChildRag:
    """ch03 as a recipe: split documents into sections (if a
    ``section_splitter`` is given), chunk them into parents and children,
    embed only the children, and answer by matching children but handing
    the LLM their parents -- retrieving with a step-back question unless
    ``step_back`` is off.

    ``index`` replaces the whole corpus; the store keeps it, so later runs
    check ``is_indexed`` and go straight to ``query``.
    """

    def __init__(
        self,
        store: ParentChildGraphStore,
        embedder: Embedder,
        llm: LLMClient,
        *,
        parent_chunker: TextChunker | None = None,
        child_chunker: TextChunker | None = None,
        section_splitter: SectionSplitter | None = None,
        token_counter: TokenCounter | None = None,
        step_back: bool = True,
        k: int = 4,
        model: str = OpenAILLMClient.DEFAULT_MODEL,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._llm = llm
        self._parent_chunker = parent_chunker or TextChunker(chunk_size=2000, overlap=40)
        self._child_chunker = child_chunker or TextChunker(chunk_size=500, overlap=20)
        self._section_splitter = section_splitter
        self._token_counter = token_counter
        retriever: Retriever = ParentChildRetriever(store, embedder, k=k)
        if step_back:
            retriever = StepBackRetriever(QueryRewriter(llm, model=model), retriever)
        self._retriever = retriever
        self._label = "step-back + parent retrieval" if step_back else "parent retrieval"
        self._model = model

    def is_indexed(self) -> bool:
        return self._store.parent_count() > 0

    def index(self, source: DocumentSource) -> None:
        steps: list[PipelineStep[_ParentChildIndexContext]] = [LoadDocumentsStep(source)]
        if self._section_splitter is not None:
            steps.append(SplitSectionsStep(self._section_splitter, self._token_counter))
        steps += [
            ParentChildChunkStep(self._parent_chunker, self._child_chunker),
            EmbedChildrenStep(self._embedder),
            StoreParentChildStep(self._store),
        ]
        Pipeline(steps).run(_ParentChildIndexContext())

    def query(self, question: str) -> RagAnswer:
        answering: Pipeline[AnswerContext] = Pipeline(
            [
                RetrieveStep(self._retriever, label=self._label),
                AnswerStep(self._llm, model=self._model),
            ]
        )
        return answering.run(AnswerContext(question)).to_answer()
