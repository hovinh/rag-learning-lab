from __future__ import annotations

from typing import Protocol

from graphrag.extraction.community_detector import KuzuLouvainCommunityDetector
from graphrag.extraction.community_reporter import CommunityReporter
from graphrag.extraction.description_summarizer import DescriptionSummarizer
from graphrag.extraction.entity_extractor import EntityRelationshipExtractor
from graphrag.extraction.entity_graph_importer import EntityGraphImporter
from graphrag.ingestion.models import Chunk
from graphrag.observability import Trace
from graphrag.retrieval.entity_index import EntityEmbeddingIndexer

_trace = Trace(__name__)


class HasChunks(Protocol):
    chunks: list[Chunk]


class ExtractEntityGraphStep:
    """Has the LLM extract entities and relationships from every chunk and
    imports each chunk's extraction (ch07), keyed by the chunk's document
    and position.
    """

    title = "Extract entities and relationships (LLM, per chunk)"

    def __init__(
        self, extractor: EntityRelationshipExtractor, importer: EntityGraphImporter
    ) -> None:
        self._extractor = extractor
        self._importer = importer

    def __call__(self, context: HasChunks) -> None:
        progress = _trace.progress("chunk", len(context.chunks))
        for chunk in context.chunks:
            extraction = self._extractor.extract(chunk.text)
            self._importer.import_chunk(chunk.document_index, chunk.index, chunk.text, extraction)
            progress.advance(
                f"document {chunk.document_index}: {len(extraction.entities)} entities, "
                f"{len(extraction.relationships)} relationships"
            )
        _trace.field(
            "graph",
            f"{self._importer.count_entities()} entities, "
            f"{self._importer.count_relationships()} relationships",
        )


# The steps below read and write the graph only, so they accept any
# context and fit any pipeline.


class SummarizeDescriptionsStep:
    title = "Summarize descriptions (LLM)"

    def __init__(self, summarizer: DescriptionSummarizer) -> None:
        self._summarizer = summarizer

    def __call__(self, context: object) -> None:
        _trace.fields(
            {
                "entities": f"{self._summarizer.summarize_entities()} summarized",
                "pairs": f"{self._summarizer.summarize_relationships()} relationship pairs "
                "summarized",
            }
        )


class DetectCommunitiesStep:
    title = "Detect communities (Louvain)"

    def __init__(self, detector: KuzuLouvainCommunityDetector) -> None:
        self._detector = detector

    def __call__(self, context: object) -> None:
        sizes = sorted(self._detector.detect().values())
        if not sizes:
            raise ValueError("Louvain found no communities; the graph has no relationships")
        _trace.field(
            "communities",
            f"{len(sizes)}, sized min {sizes[0]}, max {sizes[-1]}, "
            f"mean {sum(sizes) / len(sizes):.1f}",
        )


class ReportCommunitiesStep:
    title = "Write community reports (LLM)"

    def __init__(self, reporter: CommunityReporter) -> None:
        self._reporter = reporter

    def __call__(self, context: object) -> None:
        reports = self._reporter.report_communities()
        _trace.table(
            "reports",
            [{"community": r.community_id, "rating": r.rating, "title": r.title} for r in reports],
            max_width=60,
        )


class IndexEntitiesStep:
    title = "Embed entity summaries"

    def __init__(self, indexer: EntityEmbeddingIndexer) -> None:
        self._indexer = indexer

    def __call__(self, context: object) -> None:
        _trace.field("embedded", f"{self._indexer.index()} entity summaries")
