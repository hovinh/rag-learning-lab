from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

from graphrag.graph.cypher_store import CypherGraphStore
from graphrag.graph.errors import GraphQueryError
from graphrag.graph.text2cypher import Text2Cypher
from graphrag.llm.client import ToolDefinition

ToolResult = str | list[dict[str, Any]]


def string_parameters(name: str, description: str) -> dict[str, Any]:
    """JSON schema for a tool taking a single required string argument --
    the shape every ch05 tool uses.
    """
    return {
        "type": "object",
        "properties": {name: {"type": "string", "description": description}},
        "required": [name],
    }


class Tool(ABC):
    """A function the agent's ``ToolRouter`` can offer the LLM (ch05).

    ``name`` is both what the LLM sees and what the router looks the tool
    up by. The book keys its tools dict separately from the function name
    in each description, and the two disagree for ``respond`` (keyed
    ``answer_given``), so picking that tool raises a ``KeyError`` there;
    a single ``name`` rules that out.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    parameters: ClassVar[dict[str, Any]]

    def definition(self) -> ToolDefinition:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    def run(self, arguments: Mapping[str, Any]) -> ToolResult:
        """``arguments`` holds at least the ``required`` parameters -- the
        router checks that before calling.
        """


class AnswerGivenTool(Tool):
    """Lets the LLM answer directly when the conversation already holds
    the answer, or turn away small talk.
    """

    name = "respond"
    description = (
        "If the conversation already contains a complete answer to the question, use this "
        "tool to extract it. Additionally, if the user engages in small talk, use this tool "
        "to remind them that you can only answer questions about movies and their cast."
    )
    parameters = string_parameters("answer", "Respond directly with the answer")

    def run(self, arguments: Mapping[str, Any]) -> ToolResult:
        return str(arguments["answer"])


class Text2CypherTool(Tool):
    """The router's fallback: generate Cypher for the question (ch04's
    ``Text2Cypher``) and run it.
    """

    name = "text2cypher"
    description = (
        "Query the database with a user question. When other tools don't fit, "
        "fallback to use this one."
    )
    parameters = string_parameters("question", "The user question to find the answer for")

    def __init__(self, text2cypher: Text2Cypher, store: CypherGraphStore) -> None:
        self._text2cypher = text2cypher
        self._store = store

    def run(self, arguments: Mapping[str, Any]) -> ToolResult:
        cypher = self._text2cypher.generate(str(arguments["question"]))
        try:
            return self._store.query(cypher)
        except GraphQueryError as exc:
            # Returned rather than raised, as in the book: the error becomes
            # part of the conversation, so the critique step can ask again.
            return f"{cypher} caused an error: {exc}"
