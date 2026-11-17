import pytest

from graphrag.graph.kuzu_parent_child_store import KuzuParentChildGraphStore

EMBEDDING_DIM = 4

PARENTS = [
    "Section about pets: cats and dogs make loyal companions.",
    "Section about markets: the stock market had an eventful day.",
]
CHILDREN_PER_PARENT = [
    ["cats are great pets", "dogs are loyal companions"],
    ["the stock market rose today"],
]
CHILD_EMBEDDINGS_PER_PARENT = [
    [[1.0, 0.1, 0.0, 0.0], [0.9, 0.2, 0.0, 0.0]],
    [[0.0, 0.0, 1.0, 0.5]],
]


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test_kuzu_parent_child_db"
    with KuzuParentChildGraphStore(db_path, embedding_dim=EMBEDDING_DIM) as store:
        store.replace_parents(PARENTS, CHILDREN_PER_PARENT, CHILD_EMBEDDINGS_PER_PARENT)
        yield store


def test_parent_retrieval_returns_parent_of_closest_child(store):
    query_embedding = [1.0, 0.15, 0.0, 0.0]

    results = store.parent_retrieval(query_embedding, k=1)

    assert len(results) == 1
    assert results[0].text == PARENTS[0]


def test_parent_retrieval_orders_by_score_descending(store):
    query_embedding = [1.0, 0.15, 0.0, 0.0]

    results = store.parent_retrieval(query_embedding, k=2)

    assert len(results) == 2
    assert results[0].score >= results[1].score


def test_replace_parents_rejects_mismatched_top_level_lengths(store):
    with pytest.raises(ValueError):
        store.replace_parents(["one parent"], [], [])


def test_replace_parents_rejects_mismatched_children_and_embeddings(store):
    with pytest.raises(ValueError):
        store.replace_parents(["one parent"], [["a child"]], [[]])


def test_replace_parents_can_run_again_against_existing_indexes(store):
    # Regression test, mirroring test_kuzu_store.py: re-running
    # replace_parents (e.g. re-running the ingestion script against an
    # existing db) used to fail because Kuzu refuses to write to a column
    # already covered by an index, and dropping the index alone doesn't
    # lift that restriction.
    store.replace_parents(PARENTS, CHILDREN_PER_PARENT, CHILD_EMBEDDINGS_PER_PARENT)

    results = store.parent_retrieval([1.0, 0.15, 0.0, 0.0], k=1)

    assert len(results) == 1
    assert results[0].text == PARENTS[0]


def test_replace_parents_after_reopening_store(tmp_path):
    db_path = tmp_path / "reopened_kuzu_parent_child_db"
    with KuzuParentChildGraphStore(db_path, embedding_dim=EMBEDDING_DIM) as store:
        store.replace_parents(PARENTS, CHILDREN_PER_PARENT, CHILD_EMBEDDINGS_PER_PARENT)

    with KuzuParentChildGraphStore(db_path, embedding_dim=EMBEDDING_DIM) as store:
        store.replace_parents(PARENTS, CHILDREN_PER_PARENT, CHILD_EMBEDDINGS_PER_PARENT)
        results = store.parent_retrieval([1.0, 0.15, 0.0, 0.0], k=1)

    assert len(results) == 1
    assert results[0].text == PARENTS[0]


def test_parent_count_is_zero_until_parents_are_stored(tmp_path, store):
    with KuzuParentChildGraphStore(tmp_path / "empty_db", embedding_dim=EMBEDDING_DIM) as empty:
        assert empty.parent_count() == 0

    assert store.parent_count() == len(PARENTS)
