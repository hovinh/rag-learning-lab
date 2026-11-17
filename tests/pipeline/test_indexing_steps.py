from dataclasses import dataclass, field

from graphrag.ingestion.models import Chunk
from graphrag.pipeline.steps.indexing import (
    EmbedChildrenStep,
    EmbedChunksStep,
    StoreChunksStep,
    StoreParentChildStep,
)


@dataclass
class IndexContext:
    chunks: list[Chunk] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)
    children_per_parent: list[list[str]] = field(default_factory=list)
    child_embeddings_per_parent: list[list[list[float]]] = field(default_factory=list)


class RecordingStore:
    def __init__(self):
        self.calls = []

    def replace_chunks(self, chunks, embeddings):
        self.calls.append((chunks, embeddings))

    def replace_parents(self, parents, children_per_parent, child_embeddings_per_parent):
        self.calls.append((parents, children_per_parent, child_embeddings_per_parent))


def test_embed_then_store_chunks(embedder):
    embedder.vectors = {"a": [1.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0]}
    context = IndexContext(chunks=[Chunk(0, 0, "a"), Chunk(1, 0, "b")])
    store = RecordingStore()

    EmbedChunksStep(embedder)(context)
    StoreChunksStep(store)(context)

    assert embedder.calls == [["a", "b"]]
    assert store.calls == [(["a", "b"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])]


def test_embed_only_children_then_store_hierarchy(embedder):
    embedder.vectors = {"a": [1.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0], "c": [0.0, 0.0, 1.0]}
    context = IndexContext(parents=["a b", "c"], children_per_parent=[["a", "b"], ["c"]])
    store = RecordingStore()

    EmbedChildrenStep(embedder)(context)
    StoreParentChildStep(store)(context)

    assert embedder.calls == [["a", "b"], ["c"]]
    assert store.calls == [
        (
            ["a b", "c"],
            [["a", "b"], ["c"]],
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]]],
        )
    ]
