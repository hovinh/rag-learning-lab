import pytest

from graphrag.graph.kuzu_store import KuzuGraphStore

EMBEDDING_DIM = 4

CHUNKS = [
    "cats are great pets",
    "dogs are loyal companions",
    "the stock market rose today",
]
EMBEDDINGS = [
    [1.0, 0.1, 0.0, 0.0],
    [0.9, 0.2, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.5],
]


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test_kuzu_db"
    with KuzuGraphStore(db_path, embedding_dim=EMBEDDING_DIM) as store:
        store.replace_chunks(CHUNKS, EMBEDDINGS)
        yield store


def test_vector_search_returns_closest_chunk(store):
    query_embedding = [1.0, 0.15, 0.0, 0.0]

    results = store.vector_search(query_embedding, k=2)

    assert len(results) == 2
    assert results[0].text in {"cats are great pets", "dogs are loyal companions"}
    assert results[0].score >= results[1].score


def test_keyword_search_finds_matching_text(store):
    results = store.keyword_search("loyal dogs", k=2)

    assert results
    assert results[0].text == "dogs are loyal companions"


def test_hybrid_search_merges_and_deduplicates(store):
    query_embedding = [1.0, 0.15, 0.0, 0.0]

    results = store.hybrid_search("loyal dogs", query_embedding, k=3)

    indices = [r.index for r in results]
    assert len(indices) == len(set(indices))
    assert len(results) <= 3


def test_replace_chunks_rejects_mismatched_lengths(store):
    with pytest.raises(ValueError):
        store.replace_chunks(["one"], [])


def test_replace_chunks_can_run_again_against_existing_indexes(store):
    # Regression test: re-running replace_chunks (e.g. re-running the
    # ingestion script against an existing db) used to fail because Kuzu
    # refuses to write to a column that's already covered by an index,
    # and dropping the index alone doesn't lift that restriction.
    store.replace_chunks(CHUNKS, EMBEDDINGS)

    results = store.vector_search([1.0, 0.15, 0.0, 0.0], k=2)

    assert len(results) == 2


def test_replace_chunks_after_reopening_store(tmp_path):
    db_path = tmp_path / "reopened_kuzu_db"
    with KuzuGraphStore(db_path, embedding_dim=EMBEDDING_DIM) as store:
        store.replace_chunks(CHUNKS, EMBEDDINGS)

    with KuzuGraphStore(db_path, embedding_dim=EMBEDDING_DIM) as store:
        store.replace_chunks(CHUNKS, EMBEDDINGS)
        results = store.keyword_search("loyal dogs", k=2)

    assert results
    assert results[0].text == "dogs are loyal companions"
