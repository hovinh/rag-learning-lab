from __future__ import annotations

from collections.abc import Sequence

from graphrag.agent.query_planner import QueryPlanner
from graphrag.agent.scope import AgentScope
from graphrag.llm.client import LLMClient, Message, OpenAILLMClient
from graphrag.llm.errors import LLMResponseError
from graphrag.llm.json_response import load_json_field
from graphrag.prompts import PromptRepository

_SYSTEM_PROMPT = "agent_query_decompose_system.txt"
# The same user message as ch05's query update, as in the book.
_USER_PROMPT = "agent_query_update_user.txt"


class QueryDecomposer(QueryPlanner):
    """Splits a question into standalone atomic questions, e.g. "Who is
    Mick Jagger and how old is he?" into "Who is Mick Jagger?" and "How old
    is Mick Jagger?" (ch08's ``query_rewrite``).

    The book's ``query_rewrite`` sees only the question. Here it also sees
    ``history``, like ``QueryUpdater``, so a critic follow-up such as "When
    was he born?" can name who "he" is.
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
        return self.decompose(question, history)

    def decompose(self, question: str, history: Sequence[Message] = ()) -> list[str]:
        """Raises ``LLMResponseError`` unless the response is a non-empty
        list of questions. The book returns ``[]`` on bad JSON, which
        silently routes nothing.
        """
        response = self._llm.complete(
            [
                {"role": "system", "content": self._system_message},
                *history,
                {"role": "user", "content": self._user_template.format(question=question)},
            ],
            model=self._model,
            json_mode=True,
        )
        questions = load_json_field(response, "questions")
        if (
            not isinstance(questions, list)
            or not questions
            or not all(isinstance(q, str) and q for q in questions)
        ):
            raise LLMResponseError(f"'questions' isn't a non-empty list of strings: {response}")
        return questions
