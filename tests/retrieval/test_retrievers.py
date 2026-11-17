from graphrag.graph.models import ChunkMatch, ParentMatch
from graphrag.retrieval.retrievers import (
    HybridRetriever,
    ParentChildRetriever,
    StepBackRetriever,
    VectorRetriever,
)

QUESTION_VECTOR = [1.0, 0.0, 0.0]
HIT = ChunkMatch(0, "Einstein moved to Bern", 0.9)


class FakeChunkStore:
    def __init__(self):
        self.calls = []

    def vector_search(self, query_embedding, k):
        self.calls.append(("vector", query_embedding, k))
        return [HIT]

    def hybrid_search(self, query, query_embedding, k):
        self.calls.append(("hybrid", query, query_embedding, k))
        return [HIT]


class FakeParentChildStore:
    def __init__(self):
        self.calls = []

    def parent_retrieval(self, query_embedding, k):
        self.calls.append((query_embedding, k))
        return [ParentMatch(3, "parent text", 0.8)]


def test_vector_retriever_embeds_the_question_then_searches(embedder):
    embedder.vectors = {"When?": QUESTION_VECTOR}
    store = FakeChunkStore()

    matches = VectorRetriever(store, embedder, k=2).retrieve("When?")

    assert matches == [HIT]
    assert store.calls == [("vector", QUESTION_VECTOR, 2)]


def test_hybrid_retriever_passes_question_and_embedding(embedder):
    embedder.vectors = {"When?": QUESTION_VECTOR}
    store = FakeChunkStore()

    HybridRetriever(store, embedder, k=3).retrieve("When?")

    assert store.calls == [("hybrid", "When?", QUESTION_VECTOR, 3)]


def test_parent_child_retriever_returns_parents(embedder):
    embedder.vectors = {"When?": QUESTION_VECTOR}
    store = FakeParentChildStore()

    matches = ParentChildRetriever(store, embedder, k=4).retrieve("When?")

    assert matches == [ParentMatch(3, "parent text", 0.8)]
    assert store.calls == [(QUESTION_VECTOR, 4)]


class FakeRewriter:
    def step_back(self, question):
        return f"broader: {question}"


class RecordingRetriever:
    def __init__(self):
        self.questions = []

    def retrieve(self, question):
        self.questions.append(question)
        return [HIT]


def test_step_back_retriever_retrieves_with_the_rewritten_question():
    inner = RecordingRetriever()

    matches = StepBackRetriever(FakeRewriter(), inner).retrieve("Who helped Einstein?")

    assert inner.questions == ["broader: Who helped Einstein?"]
    assert matches == [HIT]
