import json

import pytest

from graphrag.datasets import MOVIES_BENCHMARK_PATH, MovieGraphSeeder
from graphrag.evaluation import load_benchmark
from graphrag.graph.kuzu_cypher_store import KuzuCypherGraphStore


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    with KuzuCypherGraphStore(tmp_path_factory.mktemp("kuzu") / "test_kuzu_db") as store:
        MovieGraphSeeder(store).seed()
        yield store


def test_benchmark_has_the_books_questions():
    cases = load_benchmark(MOVIES_BENCHMARK_PATH)

    assert len(cases) == 17
    assert cases[0].question == "Hello"
    assert cases[-1].question == "Which movie has the most Oscars?"


@pytest.mark.parametrize(
    "case", load_benchmark(MOVIES_BENCHMARK_PATH), ids=lambda case: case.question
)
def test_every_ground_truth_runs_on_the_seeded_graph(store, case):
    rows = store.query(case.cypher)

    assert rows
    json.dumps(rows)


def test_ground_truths_match_the_seeded_graph(store):
    cases = {case.question: case for case in load_benchmark(MOVIES_BENCHMARK_PATH)}

    assert store.query(cases["Who acted in the most movies?"].cypher) == [
        {"p.name": "Tom Hanks", "movieCount": 8}
    ]
    assert {"p.name": "Tom Cruise"} in store.query(cases["Who acted in Top Gun?"].cypher)
