from dataclasses import dataclass, field
from typing import Any

from graphrag.graph.models import GraphSchema
from graphrag.graph.text2cypher import Text2Cypher
from graphrag.pipeline.steps.cypher import ExecuteCypherStep, GenerateCypherStep


@dataclass
class CypherTestContext:
    question: str
    cypher: str = ""
    rows: list[dict[str, Any]] = field(default_factory=list)


class FakeCypherStore:
    def __init__(self, rows):
        self.rows = rows
        self.queries = []

    def query(self, cypher, params=None):
        self.queries.append(cypher)
        return self.rows


def test_generate_then_execute_cypher(llm):
    llm.completions = ["  MATCH (m:Movie) RETURN count(m) AS n  "]
    store = FakeCypherStore([{"n": 38}])
    context = CypherTestContext(question="How many movies?")

    GenerateCypherStep(Text2Cypher(llm, GraphSchema({}, {}, []), model="test-model"))(context)
    ExecuteCypherStep(store)(context)

    assert context.cypher == "MATCH (m:Movie) RETURN count(m) AS n"
    assert store.queries == ["MATCH (m:Movie) RETURN count(m) AS n"]
    assert context.rows == [{"n": 38}]
    assert "How many movies?" in llm.complete_calls[0]["messages"][0]["content"]
