from __future__ import annotations

from collections.abc import Sequence

from logzero import logger

from graphrag.agent.errors import ToolRoutingError
from graphrag.agent.tools import Tool, ToolResult
from graphrag.llm.client import LLMClient, Message, OpenAILLMClient, ToolCall
from graphrag.prompts import PromptRepository

_SYSTEM_PROMPT = "agent_tool_picker_system.txt"
_USER_PROMPT = "agent_tool_picker_user.txt"


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
        logger.info(f"Tool: {call.name}({call.arguments})")
        return tool.run(call.arguments)
