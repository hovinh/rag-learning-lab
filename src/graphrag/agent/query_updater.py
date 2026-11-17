from __future__ import annotations

from collections.abc import Sequence

from graphrag.agent.query_planner import QueryPlanner
from graphrag.agent.scope import AgentScope
from graphrag.llm.client import LLMClient, Message, OpenAILLMClient
from graphrag.llm.errors import LLMResponseError
from graphrag.llm.json_response import load_json_field
from graphrag.prompts import PromptRepository

_SYSTEM_PROMPT = "agent_query_update_system.txt"
_USER_PROMPT = "agent_query_update_user.txt"


class QueryUpdater(QueryPlanner):
    """Rewrites a question into one atomic, specific question, filling in
    references (e.g. "that person") from earlier answers in ``history``
    (ch05's ``query_update``).
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        scope: AgentScope | None = None,
        model: str = OpenAILLMClient.DEFAULT_MODEL,
        prompts: PromptRepository | None = None,
    ) -> None:
        self._llm = llm
        self._model = model
        prompts = prompts or PromptRepository()
        self._system_message = (scope or AgentScope()).fill(prompts.load(_SYSTEM_PROMPT))
        self._user_template = prompts.load(_USER_PROMPT)

    def plan(self, question: str, history: Sequence[Message] = ()) -> list[str]:
        return [self.update(question, history)]

    def update(self, question: str, history: Sequence[Message] = ()) -> str:
        """Raises ``LLMResponseError`` if the response has no question."""
        response = self._llm.complete(
            [
                {"role": "system", "content": self._system_message},
                *history,
                {"role": "user", "content": self._user_template.format(question=question)},
            ],
            model=self._model,
            json_mode=True,
        )
        updated = load_json_field(response, "question")
        if not isinstance(updated, str):
            raise LLMResponseError(f"'question' isn't a string: {response}")
        return updated
