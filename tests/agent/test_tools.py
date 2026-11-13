from graphrag.agent.tools import AnswerGivenTool, Text2CypherTool, string_parameters
from graphrag.graph.errors import GraphQueryError


class FakeText2Cypher:
    def __init__(self, cypher: str) -> None:
        self.cypher = cypher
        self.questions: list[str] = []

    def generate(self, question):
        self.questions.append(question)
        return self.cypher


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


def test_text2cypher_tool_returns_query_errors_as_text():
    t2c = FakeText2Cypher("MATCH (n:Nope) RETURN n")
    store = FakeCypherStore(error="Table Nope does not exist.")

    result = Text2CypherTool(t2c, store).run({"question": "?"})

    assert result == "MATCH (n:Nope) RETURN n caused an error: Table Nope does not exist."
