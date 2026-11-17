import pytest
from pydantic import BaseModel

from graphrag.agent.tools import AnswerGivenTool
from graphrag.domains.movies import MovieGraphSeeder
from graphrag.evaluation.metrics import Metric
from graphrag.evaluation.models import BenchmarkCase
from graphrag.extraction.entity_models import ChunkExtraction, ExtractedEntity
from graphrag.graph.kuzu_cypher_store import KuzuCypherGraphStore
from graphrag.ingestion.sources import DocumentSource
from graphrag.llm.client import ToolCall
from graphrag.recipes import (
    AgenticRag,
    CypherAnswer,
    GraphRag,
    RagAnswer,
    RagEvaluation,
    StructuredGraphExtraction,
    Text2CypherQA,
)


class ListSource(DocumentSource):
    def __init__(self, *documents):
        self.documents = list(documents)

    def load(self):
        return list(self.documents)


@pytest.fixture
def movies_store(tmp_path):
    with KuzuCypherGraphStore(tmp_path / "movies_db") as store:
        MovieGraphSeeder(store).seed()
        yield store


def test_text2cypher_qa_generates_and_runs_cypher(movies_store, llm):
    cypher = "MATCH (m:Movie {title: 'The Matrix'}) RETURN m.released AS released"
    llm.completions = [cypher]
    qa = Text2CypherQA(movies_store, llm, terminology="Movies are Movie nodes.")

    answer = qa.query("When was The Matrix released?")

    assert answer == CypherAnswer("When was The Matrix released?", cypher, [{"released": 1999}])
    assert "Movie" in qa.schema.node_props
    assert "Movies are Movie nodes." in llm.complete_calls[0]["messages"][0]["content"]


class Person(BaseModel):
    name: str


class FakeExtractor:
    def extract(self, document):
        return Person(name=document)


def test_structured_graph_extraction_imports_one_model_per_document():
    imported = []

    def import_one(person):
        imported.append(person.name)
        return f"id-{len(imported)}"

    ids = StructuredGraphExtraction(FakeExtractor(), import_one).index(ListSource("Ada", "Alan"))

    assert ids == ["id-1", "id-2"]
    assert imported == ["Ada", "Alan"]


def test_graph_rag_is_indexed_once_the_entity_index_exists(tmp_path, llm, embedder):
    with KuzuCypherGraphStore(tmp_path / "graph_db") as store:
        rag = GraphRag(store, llm, embedder, entity_types=["PERSON"])
        assert not rag.is_indexed()

        # Build the smallest finished graph by hand: the full index() is
        # covered step by step in tests/pipeline and tests/extraction.
        rag._importer.create_schema()
        rag._importer.import_chunk(
            0, 0, "text", ChunkExtraction([ExtractedEntity("ULYSSES", "PERSON", "A hero")])
        )
        store.query("MATCH (e:Entity) SET e.summary = 'A hero'")
        embedder.vectors = {"A hero": [1.0, 0.0, 0.0]}
        rag._indexer.index()

        assert rag.is_indexed()


def test_agentic_rag_returns_the_answer_with_the_gathered_contexts(llm):
    llm.completions = [
        '{"question": "Who are you?"}',  # query update
        '{"questions": []}',  # critique
        "I answer questions about the data.",  # final answer
    ]
    llm.tool_calls = [[ToolCall("respond", {"answer": "A data assistant."})]]

    answer = AgenticRag(llm, [AnswerGivenTool()], model="m").query("Who are you?")

    assert answer.question == "Who are you?"
    assert answer.answer == "I answer questions about the data."
    assert len(answer.contexts) == 1
    assert "A data assistant." in answer.contexts[0]


class ContextCount(Metric):
    name = "contexts"

    def score(self, sample):
        return float(len(sample.contexts))


def test_rag_evaluation_answers_with_any_pipeline_then_scores(movies_store, llm, embedder):
    evaluation = RagEvaluation(llm, embedder, metrics=[ContextCount()])
    cases = [BenchmarkCase("How many movies?", "MATCH (m:Movie) RETURN count(*) AS n")]

    samples = evaluation.answer(
        movies_store, cases, lambda question: RagAnswer(question, "14", ["a", "b"])
    )
    report = evaluation.score(samples)

    assert samples[0].ground_truth == '[{"n": 14}]'
    assert samples[0].answer == "14"
    assert report.means() == {"contexts": 2.0}
