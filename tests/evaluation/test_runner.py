import json

import pytest

from graphrag.agent.agent import AgentResponse
from graphrag.domains.movies import MovieGraphSeeder
from graphrag.evaluation.models import BenchmarkCase
from graphrag.evaluation.runner import BenchmarkRunner
from graphrag.graph.errors import GraphQueryError
from graphrag.graph.kuzu_cypher_store import KuzuCypherGraphStore

CASE = BenchmarkCase("How many movies are there?", "MATCH (m:Movie) RETURN count(*) AS movies")


@pytest.fixture
def store(tmp_path):
    with KuzuCypherGraphStore(tmp_path / "test_kuzu_db") as store:
        MovieGraphSeeder(store).seed()
        yield store


def test_run_records_ground_truth_answer_and_contexts(store):
    asked = []

    def respond(question):
        asked.append(question)
        return AgentResponse("There are 14 movies.", ["14 movies"])

    [sample] = BenchmarkRunner(store, respond).run([CASE])

    assert asked == [CASE.question]
    assert sample.question == CASE.question
    assert json.loads(sample.ground_truth) == [{"movies": 14}]
    assert sample.answer == "There are 14 movies."
    assert sample.contexts == ["14 movies"]
    assert sample.latency >= 0
    assert sample.error is None


def test_run_records_a_failed_answer_and_carries_on(store):
    def respond(question):
        if question == CASE.question:
            raise RuntimeError("boom")
        return AgentResponse("Hi!", [])

    hello = BenchmarkCase("Hello", 'RETURN "hi" AS x')
    failed, answered = BenchmarkRunner(store, respond).run([CASE, hello])

    assert failed.error == "RuntimeError('boom')"
    assert failed.answer == ""
    assert failed.contexts == []
    assert json.loads(failed.ground_truth) == [{"movies": 14}]
    assert answered.error is None
    assert answered.answer == "Hi!"


def test_run_raises_on_a_broken_ground_truth_query(store):
    runner = BenchmarkRunner(store, lambda question: AgentResponse("", []))

    with pytest.raises(GraphQueryError):
        runner.run([BenchmarkCase("?", "MATCH (n:NoSuchTable) RETURN n")])
