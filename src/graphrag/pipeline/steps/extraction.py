from __future__ import annotations

from collections.abc import Callable
from typing import Generic, Protocol

from graphrag.extraction.extractor import StructuredExtractor
from graphrag.llm.client import ModelT
from graphrag.observability import Trace

_trace = Trace(__name__)


class ExtractionContext(Protocol[ModelT]):
    documents: list[str]
    extracted: list[ModelT]


class ExtractStructuredStep(Generic[ModelT]):
    """Extracts one typed model per document with structured outputs
    (ch06), whatever the model is.
    """

    title = "Extract (structured outputs)"

    def __init__(self, extractor: StructuredExtractor[ModelT]) -> None:
        self._extractor = extractor

    def __call__(self, context: ExtractionContext[ModelT]) -> None:
        context.extracted = [self._extractor.extract(document) for document in context.documents]
        for index, item in enumerate(context.extracted, start=1):
            _trace.block(f"extracted {index}", item.model_dump(mode="json"))


class ImportContext(Protocol[ModelT]):
    extracted: list[ModelT]
    imported_ids: list[str]


class ImportExtractedStep(Generic[ModelT]):
    """Writes each extracted model into the graph with ``import_one``
    (e.g. ``ContractGraphImporter.import_contract``), keeping the ids it
    returns.
    """

    title = "Import into the graph"

    def __init__(self, import_one: Callable[[ModelT], str]) -> None:
        self._import_one = import_one

    def __call__(self, context: ImportContext[ModelT]) -> None:
        context.imported_ids = [self._import_one(item) for item in context.extracted]
        _trace.field("imported", ", ".join(context.imported_ids) or "nothing")
