from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from graphrag.extraction.community_detector import KuzuLouvainCommunityDetector
from graphrag.extraction.community_reporter import CommunityReporter
from graphrag.extraction.description_summarizer import DescriptionSummarizer
from graphrag.extraction.entity_extractor import EntityRelationshipExtractor
from graphrag.extraction.entity_graph_importer import EntityGraphImporter
from graphrag.graph.cypher_store import CypherGraphStore
from graphrag.ingestion.chunker import TextChunker
from graphrag.ingestion.models import Chunk
from graphrag.ingestion.sources import DocumentSource
from graphrag.llm.client import LLMClient, OpenAILLMClient
from graphrag.llm.token_counter import TokenCounter
from graphrag.pipeline import (
    ChunkStep,
    DetectCommunitiesStep,
    ExtractEntityGraphStep,
    GlobalSearchStep,
    IndexEntitiesStep,
    LoadDocumentsStep,
    LocalSearchStep,
    Pipeline,
    ReportCommunitiesStep,
    SummarizeDescriptionsStep,
)
from graphrag.recipes.models import RagAnswer
from graphrag.retrieval.embedder import Embedder
from graphrag.retrieval.entity_index import EntityEmbeddingIndexer
from graphrag.retrieval.global_search import GlobalSearch, KeyPoint
from graphrag.retrieval.local_search import LocalContext, LocalSearch


@dataclass
class _GraphIndexContext:
    documents: list[str] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)


@dataclass
class _GlobalContext:
    question: str
    key_points: list[KeyPoint] = field(default_factory=list)
    answer: str = ""


@dataclass
class _LocalContext:
    question: str
    local_context: LocalContext = field(default_factory=LocalContext)
    answer: str = ""


class GraphRag:
    """ch07 (Microsoft-style GraphRAG) as a recipe. ``index``
    has the LLM extract entities of ``entity_types`` and their
    relationships from every chunk, summarizes them, clusters the entities
    into communities with a report each, and embeds the entities. Then
    ``global_search`` answers broad questions from the community reports
    and ``local_search`` answers questions about specific entities from
    their neighbourhood.

    Indexing is the expensive part (100-150 LLM calls per Odyssey book),
    and the graph persists in ``store``: check ``is_indexed`` first. Each
    ``index`` starts from an empty graph, since re-importing would
    duplicate descriptions.
    """

    def __init__(
        self,
        store: CypherGraphStore,
        llm: LLMClient,
        embedder: Embedder,
        *,
        entity_types: Sequence[str],
        chunker: TextChunker | None = None,
        model: str = OpenAILLMClient.DEFAULT_MODEL,
    ) -> None:
        self._store = store
        self._llm = llm
        self._embedder = embedder
        self._entity_types = list(entity_types)
        self._chunker = chunker or TextChunker(chunk_size=1000, overlap=40)
        self._model = model
        self._importer = EntityGraphImporter(store, embedder.dimensions)
        self._indexer = EntityEmbeddingIndexer(store, embedder)

    def is_indexed(self) -> bool:
        return self._indexer.is_indexed()

    def index(
        self,
        source: DocumentSource,
        *,
        limit: int | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        """``limit`` keeps only the first documents (ch07 uses one book)."""
        self._importer.create_schema(reset=True)
        store, llm, model = self._store, self._llm, self._model
        indexing: Pipeline[_GraphIndexContext] = Pipeline(
            [
                LoadDocumentsStep(source, limit=limit, token_counter=token_counter),
                ChunkStep(self._chunker),
                ExtractEntityGraphStep(
                    EntityRelationshipExtractor(llm, self._entity_types, model=model),
                    self._importer,
                ),
                SummarizeDescriptionsStep(DescriptionSummarizer(store, llm, model=model)),
                DetectCommunitiesStep(KuzuLouvainCommunityDetector(store)),
                ReportCommunitiesStep(CommunityReporter(store, llm, model=model)),
                IndexEntitiesStep(self._indexer),
            ]
        )
        indexing.run(_GraphIndexContext())

    def global_search(self, question: str) -> RagAnswer:
        """Map-reduce over the community reports. The contexts are the
        key points the map step found.
        """
        searching: Pipeline[_GlobalContext] = Pipeline(
            [GlobalSearchStep(GlobalSearch(self._store, self._llm, model=self._model))]
        )
        context = searching.run(_GlobalContext(question))
        return RagAnswer(question, context.answer, [p.description for p in context.key_points])

    def local_search(self, question: str) -> RagAnswer:
        """Answers from the entities nearest the question and their
        chunks, reports and relationships, which are the contexts.
        """
        searching: Pipeline[_LocalContext] = Pipeline(
            [
                LocalSearchStep(
                    LocalSearch(self._store, self._llm, self._embedder, model=self._model)
                )
            ]
        )
        context = searching.run(_LocalContext(question))
        local = context.local_context
        contexts = [*local.entities, *local.chunks, *local.reports, *local.relationships]
        return RagAnswer(question, context.answer, contexts)
