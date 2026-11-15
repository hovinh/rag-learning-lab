import pytest

from graphrag.extraction.entity_graph_importer import EntityGraphImporter
from graphrag.graph.kuzu_cypher_store import KuzuCypherGraphStore
from graphrag.llm.client import LLMClient
from graphrag.retrieval.embedder import Embedder

# Small enough to write embeddings by hand in tests.
TEST_EMBEDDING_DIM = 3


class ScriptedLLMClient(LLMClient):
    """Returns queued responses in order -- ``completions`` for
    ``complete``, ``tool_calls`` for ``choose_tools`` -- and records every
    call.
    """

    def __init__(self) -> None:
        self.completions: list[str] = []
        self.tool_calls: list[list] = []
        self.complete_calls: list[dict] = []
        self.choose_tools_calls: list[dict] = []

    def complete(self, messages, *, model, temperature=0.0, json_mode=False):
        self.complete_calls.append({"messages": messages, "model": model, "json_mode": json_mode})
        return self.completions.pop(0)

    def stream(self, messages, *, model, temperature=0.0):
        raise NotImplementedError

    def choose_tools(self, messages, tools, *, model, temperature=0.0):
        self.choose_tools_calls.append({"messages": messages, "tools": tools, "model": model})
        return self.tool_calls.pop(0)

    def parse(self, messages, response_model, *, model, temperature=0.0):
        raise NotImplementedError


@pytest.fixture
def llm():
    return ScriptedLLMClient()


class FakeEmbedder(Embedder):
    """Looks each text up in ``vectors`` and records what it was asked to
    embed.
    """

    def __init__(self) -> None:
        self.vectors: dict[str, list[float]] = {}
        self.calls: list[list[str]] = []

    @property
    def dimensions(self) -> int:
        return TEST_EMBEDDING_DIM

    def embed(self, texts):
        self.calls.append(texts)
        return [self.vectors[text] for text in texts]


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def entity_store(tmp_path):
    """A real Kuzu database with ch07's entity-graph schema."""
    with KuzuCypherGraphStore(tmp_path / "test_kuzu_db") as store:
        EntityGraphImporter(store, TEST_EMBEDDING_DIM).create_schema()
        yield store


@pytest.fixture
def entity_importer(entity_store):
    return EntityGraphImporter(entity_store, TEST_EMBEDDING_DIM)
