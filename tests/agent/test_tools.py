import pytest

from graphrag.agent.tools import AnswerGivenTool, Text2CypherTool, string_parameters
from graphrag.graph.errors import GraphQueryError
from graphrag.graph.kuzu_cypher_store import KuzuCypherGraphStore


class FakeText2Cypher:
    """Generates ``cypher``, then hands out ``corrected`` queries in order,
    recording each ``correct`` call.
    """

    def __init__(self, cypher: str, corrected: list[str] | None = None) -> None:
        self.cypher = cypher
        self.corrected = corrected or []
        self.questions: list[str] = []
        self.corrections: list[tuple[str, str, str]] = []

    def generate(self, question):
        self.questions.append(question)
        return self.cypher

    def correct(self, question, cypher, error):
        self.corrections.append((question, cypher, error))
        return self.corrected.pop(0)


class FakeCypherStore:
    def __init__(self, rows=None, error: str | None = None) -> None:
        self.rows = rows or []
        self.error = error
        self.queries: list[str] = []

    def query(self, cypher, params=None):
        self.queries.append(cypher)
        if self.error:
            raise GraphQueryError(self.error)
        return self.rows

    def get_schema(self):
        raise NotImplementedError


def test_definition_uses_tool_name_description_and_parameters():
    definition = AnswerGivenTool().definition()

    assert definition == {
        "type": "function",
        "function": {
            "name": "respond",
            "description": AnswerGivenTool.description,
            "parameters": string_parameters("answer", "Respond directly with the answer"),
        },
    }


def test_string_parameters_requires_the_single_argument():
    assert string_parameters("title", "The movie title") == {
        "type": "object",
        "properties": {"title": {"type": "string", "description": "The movie title"}},
        "required": ["title"],
    }


def test_answer_given_returns_the_answer():
    assert AnswerGivenTool().run({"answer": "I only know about movies."}) == (
        "I only know about movies."
    )


def test_text2cypher_tool_runs_generated_cypher():
    t2c = FakeText2Cypher("MATCH (p:Person) RETURN count(p) AS n")
    store = FakeCypherStore(rows=[{"n": 48}])

    result = Text2CypherTool(t2c, store).run({"question": "How many people?"})

    assert result == [{"n": 48}]
    assert t2c.questions == ["How many people?"]
    assert store.queries == ["MATCH (p:Person) RETURN count(p) AS n"]


def test_text2cypher_tool_returns_query_errors_as_text_without_corrections():
    t2c = FakeText2Cypher("MATCH (n:Nope) RETURN n")
    store = FakeCypherStore(error="Table Nope does not exist.")

    result = Text2CypherTool(t2c, store, max_corrections=0).run({"question": "?"})

    assert result == "MATCH (n:Nope) RETURN n caused an error: Table Nope does not exist."
    assert t2c.corrections == []


def test_text2cypher_tool_returns_the_last_error_when_the_correction_fails_too():
    t2c = FakeText2Cypher("MATCH (n:Nope) RETURN n", corrected=["MATCH (n:Still) RETURN n"])
    store = FakeCypherStore(error="Table does not exist.")

    result = Text2CypherTool(t2c, store).run({"question": "?"})

    assert result == "MATCH (n:Still) RETURN n caused an error: Table does not exist."
    assert t2c.corrections == [("?", "MATCH (n:Nope) RETURN n", "Table does not exist.")]
    assert store.queries == ["MATCH (n:Nope) RETURN n", "MATCH (n:Still) RETURN n"]


def test_text2cypher_tool_rejects_negative_corrections():
    with pytest.raises(ValueError):
        Text2CypherTool(FakeText2Cypher(""), FakeCypherStore(), max_corrections=-1)


def test_text2cypher_tool_corrects_a_query_kuzu_rejects(tmp_path):
    question = "Who has the longest name?"
    t2c = FakeText2Cypher(
        "MATCH (p:Person) RETURN p.name AS name ORDER BY length(p.name) DESC LIMIT 1",
        corrected=["MATCH (p:Person) RETURN p.name AS name ORDER BY size(p.name) DESC LIMIT 1"],
    )
    with KuzuCypherGraphStore(tmp_path / "test_kuzu_db") as store:
        store.query("CREATE NODE TABLE Person(name STRING, PRIMARY KEY(name))")
        store.query(
            "CREATE (:Person {name: 'Tom Hanks'}), (:Person {name: 'Michael Clarke Duncan'})"
        )

        result = Text2CypherTool(t2c, store).run({"question": question})

    assert result == [{"name": "Michael Clarke Duncan"}]
    [(asked, rejected, error)] = t2c.corrections
    assert (asked, rejected) == (question, t2c.cypher)
    assert "length" in error.lower()


def test_retriever_tool_returns_each_match_text_and_score():
    from graphrag.agent.retriever_tool import RetrieverTool
    from graphrag.graph.models import ChunkMatch

    class FixedRetriever:
        def retrieve(self, question):
            return [ChunkMatch(0, f"about {question}", 0.5)]

    tool = RetrieverTool(FixedRetriever(), name="search_papers", description="Search papers")

    assert tool.definition()["function"]["name"] == "search_papers"
    assert tool.definition()["function"]["description"] == "Search papers"
    assert tool.run({"question": "Bern"}) == [{"text": "about Bern", "score": 0.5}]
