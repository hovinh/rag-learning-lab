import pytest

from graphrag.graph.kuzu_parent_child_store import KuzuParentChildGraphStore
from graphrag.graph.kuzu_store import KuzuGraphStore
from graphrag.ingestion.chunker import TextChunker
from graphrag.ingestion.sources import DocumentSource
from graphrag.recipes import ParentChildRag, RagAnswer, VectorRag


class ListSource(DocumentSource):
    def __init__(self, *documents):
        self.documents = list(documents)

    def load(self):
        return list(self.documents)


EMBEDDING_DIM = 3  # FakeEmbedder's

BERN = "Einstein worked at the patent office in Bern"
PRINCETON = "Einstein later moved to Princeton"


@pytest.fixture
def chunk_store(tmp_path):
    with KuzuGraphStore(tmp_path / "chunks_db", embedding_dim=EMBEDDING_DIM) as store:
        yield store


def test_vector_rag_indexes_once_then_answers_from_the_closest_chunks(chunk_store, embedder, llm):
    embedder.vectors = {
        BERN: [1.0, 0.0, 0.0],
        PRINCETON: [0.0, 1.0, 0.0],
        "Where was the patent office?": [0.9, 0.1, 0.0],
    }
    rag = VectorRag(chunk_store, embedder, llm, chunker=TextChunker(1000, 0), k=1, model="m")

    assert not rag.is_indexed()
    rag.index(ListSource(BERN, PRINCETON))
    assert rag.is_indexed()

    llm.streamed = ["In Bern."]
    answer = rag.query("Where was the patent office?")

    assert answer == RagAnswer("Where was the patent office?", "In Bern.", [BERN])
    assert BERN in llm.stream_calls[0]["messages"][1]["content"]


def test_vector_rag_hybrid_search_also_matches_keywords(chunk_store, embedder, llm):
    embedder.vectors = {
        BERN: [1.0, 0.0, 0.0],
        PRINCETON: [0.0, 1.0, 0.0],
        "Princeton": [1.0, 0.0, 0.0],
    }
    rag = VectorRag(chunk_store, embedder, llm, chunker=TextChunker(1000, 0), k=2)
    rag.index(ListSource(BERN, PRINCETON))
    llm.streamed = ["Princeton."]

    answer = rag.query("Princeton", search="hybrid")

    assert set(answer.contexts) == {BERN, PRINCETON}


@pytest.fixture
def parent_child_store(tmp_path):
    with KuzuParentChildGraphStore(
        tmp_path / "parent_child_db", embedding_dim=EMBEDDING_DIM
    ) as store:
        yield store


def test_parent_child_rag_matches_children_and_answers_from_parents(
    parent_child_store, embedder, llm
):
    embedder.vectors = {
        "aaa": [1.0, 0.0, 0.0],
        "bbb": [0.0, 1.0, 0.0],
        "ccc": [0.0, 0.0, 1.0],
        "broader question": [0.0, 0.0, 1.0],
    }
    llm.completions = ["broader question"]  # step-back rewrite
    llm.streamed = ["From the second parent."]
    rag = ParentChildRag(
        parent_child_store,
        embedder,
        llm,
        parent_chunker=TextChunker(7, 0),
        child_chunker=TextChunker(3, 0),
        k=1,
    )

    assert not rag.is_indexed()
    rag.index(ListSource("aaa bbb ccc"))
    assert rag.is_indexed()
    answer = rag.query("specific question")

    assert answer.contexts == ["ccc"]
    assert answer.answer == "From the second parent."
    # The step-back question is only used to retrieve.
    assert llm.complete_calls[0]["messages"][1]["content"] == "specific question"
    assert "specific question" in llm.stream_calls[0]["messages"][1]["content"]
