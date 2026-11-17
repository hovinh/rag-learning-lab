"""Reusable pipeline steps. Each step declares the context fields it
reads and writes as a ``Protocol``, so any context dataclass with those
fields can run it; field names are shared across steps (``documents``,
``chunks``, ``question``, ``matches``, ``answer``, ...) so steps chain.
"""

from graphrag.pipeline.steps.answer import AnswerStep
from graphrag.pipeline.steps.cypher import ExecuteCypherStep, GenerateCypherStep
from graphrag.pipeline.steps.extraction import ExtractStructuredStep, ImportExtractedStep
from graphrag.pipeline.steps.graph_index import (
    DetectCommunitiesStep,
    ExtractEntityGraphStep,
    IndexEntitiesStep,
    ReportCommunitiesStep,
    SummarizeDescriptionsStep,
)
from graphrag.pipeline.steps.graph_search import GlobalSearchStep, LocalSearchStep
from graphrag.pipeline.steps.indexing import (
    EmbedChildrenStep,
    EmbedChunksStep,
    StoreChunksStep,
    StoreParentChildStep,
)
from graphrag.pipeline.steps.ingestion import (
    ChunkStep,
    LoadDocumentsStep,
    ParentChildChunkStep,
    SplitSectionsStep,
)

__all__ = [
    "AnswerStep",
    "ChunkStep",
    "DetectCommunitiesStep",
    "EmbedChildrenStep",
    "EmbedChunksStep",
    "ExecuteCypherStep",
    "ExtractEntityGraphStep",
    "ExtractStructuredStep",
    "GenerateCypherStep",
    "GlobalSearchStep",
    "ImportExtractedStep",
    "IndexEntitiesStep",
    "LoadDocumentsStep",
    "LocalSearchStep",
    "ParentChildChunkStep",
    "ReportCommunitiesStep",
    "SplitSectionsStep",
    "StoreChunksStep",
    "StoreParentChildStep",
    "SummarizeDescriptionsStep",
]
