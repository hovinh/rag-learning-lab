from dataclasses import dataclass, field

from graphrag.ingestion.chunker import TextChunker
from graphrag.ingestion.models import Chunk
from graphrag.ingestion.sources import DocumentSource
from graphrag.pipeline.steps.ingestion import (
    ChunkStep,
    LoadDocumentsStep,
    ParentChildChunkStep,
    SplitSectionsStep,
)


class ListSource(DocumentSource):
    def __init__(self, documents):
        self.documents = documents

    def load(self):
        return list(self.documents)


@dataclass
class IngestionContext:
    documents: list[str] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)
    children_per_parent: list[list[str]] = field(default_factory=list)


def test_load_documents_step_loads_every_document():
    context = IngestionContext()

    LoadDocumentsStep(ListSource(["one", "two"]))(context)

    assert context.documents == ["one", "two"]


def test_load_documents_step_keeps_only_the_first_limit_documents():
    context = IngestionContext()

    LoadDocumentsStep(ListSource(["one", "two", "three"]), limit=2)(context)

    assert context.documents == ["one", "two"]


class FakeSplitter:
    def split(self, text):
        return text.split("|")


def test_split_sections_step_replaces_documents_with_their_sections():
    context = IngestionContext(documents=["a|b", "c"])

    SplitSectionsStep(FakeSplitter())(context)

    assert context.documents == ["a", "b", "c"]


def test_chunk_step_keeps_each_chunks_document_and_position():
    context = IngestionContext(documents=["aaa bbb ccc", "ddd"])

    ChunkStep(TextChunker(chunk_size=3, overlap=0))(context)

    assert context.chunks == [
        Chunk(0, 0, "aaa"),
        Chunk(0, 1, "bbb"),
        Chunk(0, 2, "ccc"),
        Chunk(1, 0, "ddd"),
    ]


def test_parent_child_chunk_step_chunks_parents_then_children():
    context = IngestionContext(documents=["aaa bbb ccc ddd", "eee"])

    ParentChildChunkStep(
        parent_chunker=TextChunker(chunk_size=7, overlap=0),
        child_chunker=TextChunker(chunk_size=3, overlap=0),
    )(context)

    assert context.parents == ["aaa bbb", "ccc ddd", "eee"]
    assert context.children_per_parent == [["aaa", "bbb"], ["ccc", "ddd"], ["eee"]]
