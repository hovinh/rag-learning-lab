from __future__ import annotations

from typing import Protocol

from graphrag.ingestion.chunker import TextChunker
from graphrag.ingestion.models import Chunk
from graphrag.ingestion.section_splitter import SectionSplitter
from graphrag.ingestion.sources import DocumentSource
from graphrag.llm.token_counter import TokenCounter
from graphrag.observability import Trace
from graphrag.pipeline.steps._stats import token_stats

_trace = Trace(__name__)


class HasDocuments(Protocol):
    documents: list[str]


class LoadDocumentsStep:
    """Loads ``source``'s documents into ``context.documents``, keeping
    only the first ``limit`` if given (ch07 builds its graph from one of
    the Odyssey's 24 books). With a ``token_counter``, it also logs how
    big the kept documents are.
    """

    title = "Load documents"

    def __init__(
        self,
        source: DocumentSource,
        *,
        limit: int | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._source = source
        self._limit = limit
        self._token_counter = token_counter

    def __call__(self, context: HasDocuments) -> None:
        documents = self._source.load()
        context.documents = documents[: self._limit]
        characters = sum(len(document) for document in context.documents)
        if len(context.documents) < len(documents):
            kept = f"{len(documents)} loaded, processing {len(context.documents)}"
        else:
            kept = str(len(documents))
        _trace.field("documents", f"{kept} ({characters:,} characters)")
        if self._token_counter is not None:
            _trace.field("tokens", token_stats(self._token_counter, context.documents))


class SplitSectionsStep:
    """Replaces each document with its numbered-title sections (ch03), so
    the steps after it chunk sections instead of whole documents.
    """

    title = "Split into titled sections"

    def __init__(
        self, splitter: SectionSplitter, token_counter: TokenCounter | None = None
    ) -> None:
        self._splitter = splitter
        self._token_counter = token_counter

    def __call__(self, context: HasDocuments) -> None:
        context.documents = [
            section for document in context.documents for section in self._splitter.split(document)
        ]
        _trace.field("sections", len(context.documents))
        # The book prints the token counts to show why sections are
        # chunked further: some are far longer than one passage of context
        # should be.
        if self._token_counter is not None:
            _trace.field("tokens", token_stats(self._token_counter, context.documents))


class ChunkContext(HasDocuments, Protocol):
    chunks: list[Chunk]


class ChunkStep:
    """Chunks every document, remembering which document each chunk came
    from (ch07 keys its graph's chunks by book).
    """

    def __init__(self, chunker: TextChunker) -> None:
        self._chunker = chunker

    def __call__(self, context: ChunkContext) -> None:
        context.chunks = [
            Chunk(document_index, index, text)
            for document_index, document in enumerate(context.documents)
            for index, text in enumerate(self._chunker.split(document))
        ]
        if len(context.documents) > 1:
            per_document = [0] * len(context.documents)
            for chunk in context.chunks:
                per_document[chunk.document_index] += 1
            _trace.field("chunks", f"{len(context.chunks)}, per document: {per_document}")
        else:
            _trace.field("chunks", len(context.chunks))


class ParentChildChunkContext(HasDocuments, Protocol):
    parents: list[str]
    children_per_parent: list[list[str]]


class ParentChildChunkStep:
    """Chunks every document into large parent chunks, and each parent
    into small child chunks (ch03): children are what gets embedded and
    matched, parents what the LLM is given.
    """

    title = "Chunk into parents and children"

    def __init__(self, parent_chunker: TextChunker, child_chunker: TextChunker) -> None:
        self._parent_chunker = parent_chunker
        self._child_chunker = child_chunker

    def __call__(self, context: ParentChildChunkContext) -> None:
        context.parents = [
            parent
            for document in context.documents
            for parent in self._parent_chunker.split(document)
        ]
        context.children_per_parent = [
            self._child_chunker.split(parent) for parent in context.parents
        ]
        _trace.fields(
            {
                "parents": len(context.parents),
                "children": sum(len(children) for children in context.children_per_parent),
            }
        )
