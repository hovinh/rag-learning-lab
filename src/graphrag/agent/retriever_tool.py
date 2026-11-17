from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from graphrag.agent.tools import Tool, ToolResult, string_parameters
from graphrag.retrieval.retriever import Retriever


class RetrieverTool(Tool):
    """Offers any ``Retriever`` to the agent as a tool, so it can search
    unstructured text next to the Cypher tools. ``description`` is what
    the LLM routes on, so say what the text is about.
    """

    def __init__(
        self,
        retriever: Retriever,
        *,
        name: str = "search_documents",
        description: str = "Search the documents for passages relevant to the question",
    ) -> None:
        self._retriever = retriever
        self.name = name
        self.description = description
        self.parameters = string_parameters("question", "The question to search for")

    def run(self, arguments: Mapping[str, Any]) -> ToolResult:
        matches = self._retriever.retrieve(str(arguments["question"]))
        return [{"text": match.text, "score": match.score} for match in matches]
