from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from graphrag.agent.errors import ToolRoutingError
from graphrag.agent.tools import Tool, ToolResult
from graphrag.llm.client import LLMClient, Message, OpenAILLMClient, ToolCall
from graphrag.observability import Trace
from graphrag.prompts import PromptRepository

_SYSTEM_PROMPT = "agent_tool_picker_system.txt"
_USER_PROMPT = "agent_tool_picker_user.txt"

_trace = Trace(__name__)


class ToolRouter:
    """Has the LLM pick tools for a question via function calling, then
    runs them (ch05's ``route_question``/``handle_tool_calls``).
    """

    def __init__(
        self,
        llm: LLMClient,
        tools: Sequence[Tool],
        *,
        model: str = OpenAILLMClient.DEFAULT_MODEL,
        prompts: PromptRepository | None = None,
    ) -> None:
        self._llm = llm
        self._model = model
        self._tools = {tool.name: tool for tool in tools}
        prompts = prompts or PromptRepository()
        self._system_message = prompts.load(_SYSTEM_PROMPT)
        self._user_template = prompts.load(_USER_PROMPT)

    def route(self, question: str, history: Sequence[Message] = ()) -> list[ToolResult]:
        """Returns one result per tool the LLM called, in call order.

        Raises ``ToolRoutingError`` if it called an unknown tool or left out
        a required argument.
        """
        calls = self._llm.choose_tools(
            [
                {"role": "system", "content": self._system_message},
                *history,
                {"role": "user", "content": self._user_template.format(question=question)},
            ],
            [tool.definition() for tool in self._tools.values()],
            model=self._model,
        )
        if not calls:
            _trace.field("tool", "none chosen", level=logging.WARNING)
        return [self._run(call) for call in calls]

    def _run(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            raise ToolRoutingError(f"LLM chose unknown tool {call.name!r}")
        missing = set(tool.parameters.get("required", [])) - call.arguments.keys()
        if missing:
            raise ToolRoutingError(
                f"LLM called {call.name!r} without required arguments {sorted(missing)}"
            )
        _trace.field("tool", f"{call.name}({format_arguments(call.arguments)})")
        result = tool.run(call.arguments)
        _trace.summary("result", result)
        return result


def format_arguments(arguments: Mapping[str, Any]) -> str:
    """``{'title': 'Matrix'}`` -> ``title="Matrix"``."""
    return ", ".join(
        f"{name}={json.dumps(value, ensure_ascii=False, default=str)}"
        for name, value in arguments.items()
    )
