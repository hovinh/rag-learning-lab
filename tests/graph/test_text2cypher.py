import pytest

from graphrag.graph.models import GraphSchema, Property, RelationshipPattern
from graphrag.graph.text2cypher import CypherExample, Text2Cypher
from graphrag.llm.client import LLMClient, Message
from graphrag.prompts.repository import PromptRepository

SCHEMA = GraphSchema(
    node_props={"Person": [Property("name", "STRING")], "Movie": [Property("title", "STRING")]},
    rel_props={},
    relationships=[RelationshipPattern("Person", "DIRECTED", "Movie")],
)


class FakeLLMClient(LLMClient):
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[list[Message], str]] = []

    def complete(self, messages, *, model, temperature=0.0):
        self.calls.append((messages, model))
        return self.response

    def stream(self, messages, *, model, temperature=0.0):
        raise NotImplementedError


@pytest.fixture
def prompts(tmp_path):
    for name, text in {
        "text2cypher_instructions.txt": "INSTRUCTIONS",
        "text2cypher_schema.txt": "SCHEMA:\n{schema}",
        "text2cypher_terminology.txt": "TERMS:\n{terminology}",
        "text2cypher_examples.txt": "EXAMPLES:\n{examples}",
        "text2cypher_question.txt": "Q: {question}",
    }.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return PromptRepository(tmp_path)


def test_build_prompt_includes_all_sections_in_order(prompts):
    t2c = Text2Cypher(
        FakeLLMClient(""),
        SCHEMA,
        terminology="\nDirectors are Person nodes.\n",
        examples=[
            CypherExample("How many movies?", "MATCH (m:Movie) RETURN count(m)"),
            CypherExample("How many people?", "MATCH (p:Person) RETURN count(p)"),
        ],
        prompts=prompts,
    )

    assert t2c.build_prompt("Who directed the most movies?") == (
        "INSTRUCTIONS\n\n"
        f"SCHEMA:\n{SCHEMA.to_prompt()}\n\n"
        "TERMS:\nDirectors are Person nodes.\n\n"
        "EXAMPLES:\n"
        "Question: How many movies?\nCypher: MATCH (m:Movie) RETURN count(m)\n"
        "Question: How many people?\nCypher: MATCH (p:Person) RETURN count(p)\n\n"
        "Q: Who directed the most movies?"
    )


def test_build_prompt_omits_terminology_and_examples_when_not_given(prompts):
    t2c = Text2Cypher(FakeLLMClient(""), SCHEMA, prompts=prompts)

    prompt = t2c.build_prompt("Who directed the most movies?")

    assert "TERMS" not in prompt
    assert "EXAMPLES" not in prompt
    assert (
        prompt
        == f"INSTRUCTIONS\n\nSCHEMA:\n{SCHEMA.to_prompt()}\n\nQ: Who directed the most movies?"
    )


def test_generate_sends_prompt_as_user_message_and_strips_output(prompts):
    llm = FakeLLMClient("  MATCH (p:Person) RETURN p.name\n")
    t2c = Text2Cypher(llm, SCHEMA, model="gpt-4o-mini", prompts=prompts)

    cypher = t2c.generate("List people")

    assert cypher == "MATCH (p:Person) RETURN p.name"
    [(messages, model)] = llm.calls
    assert model == "gpt-4o-mini"
    assert messages == [{"role": "user", "content": t2c.build_prompt("List people")}]


def test_default_prompts_mention_kuzu_and_contain_schema():
    prompt = Text2Cypher(FakeLLMClient(""), SCHEMA).build_prompt("List people")

    assert "Kuzu" in prompt
    assert SCHEMA.to_prompt() in prompt
    assert prompt.endswith("User question: List people")
