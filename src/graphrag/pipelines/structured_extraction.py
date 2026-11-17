from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic

from graphrag.extraction.extractor import StructuredExtractor
from graphrag.ingestion.sources import DocumentSource
from graphrag.llm.client import ModelT
from graphrag.pipeline import (
    ExtractStructuredStep,
    ImportExtractedStep,
    LoadDocumentsStep,
    Pipeline,
)


@dataclass
class _ExtractionContext(Generic[ModelT]):
    documents: list[str] = field(default_factory=list)
    extracted: list[ModelT] = field(default_factory=list)
    imported_ids: list[str] = field(default_factory=list)


class StructuredGraphExtraction(Generic[ModelT]):
    """ch06 as a reusable pipeline: extract one typed model per document
    with structured outputs, and write each into the graph with
    ``import_one`` (e.g. ``ContractGraphImporter.import_contract``).

    Every ``index`` call adds to the graph, so a caller that runs it on
    each start should first check what's already there (e.g.
    ``ContractGraphImporter.contract_ids``).
    """

    def __init__(
        self, extractor: StructuredExtractor[ModelT], import_one: Callable[[ModelT], str]
    ) -> None:
        self._extractor = extractor
        self._import_one = import_one

    def index(self, source: DocumentSource) -> list[str]:
        """Returns the imported ids, one per document."""
        pipeline: Pipeline[_ExtractionContext[ModelT]] = Pipeline(
            [
                LoadDocumentsStep(source),
                ExtractStructuredStep(self._extractor),
                ImportExtractedStep(self._import_one),
            ]
        )
        return pipeline.run(_ExtractionContext()).imported_ids
