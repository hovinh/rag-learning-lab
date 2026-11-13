from __future__ import annotations

from collections.abc import Sequence

from graphrag.llm.client import LLMClient, Message, OpenAILLMClient
from graphrag.llm.errors import LLMResponseError
from graphrag.llm.json_response import load_json_field
from graphrag.prompts import PromptRepository

_SYSTEM_PROMPT = "agent_answer_critique_system.txt"
_USER_PROMPT = "agent_answer_critique_user.txt"


class AnswerCritic:
    """Checks whether the answers gathered in ``history`` fully answer the
    question, and if not, returns follow-up questions for what's missing
    (ch05's ``critique_answers``).
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        model: str = OpenAILLMClient.DEFAULT_MODEL,
        prompts: PromptRepository | None = None,
    ) -> None:
        self._llm = llm
        self._model = model
        prompts = prompts or PromptRepository()
        self._system_message = prompts.load(_SYSTEM_PROMPT)
        self._user_template = prompts.load(_USER_PROMPT)

    def critique(self, question: str, history: Sequence[Message]) -> list[str]:
        """Returns an empty list if the question is already answered.

        Raises ``LLMResponseError`` if the response has no list of
        questions.
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
        if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
            raise LLMResponseError(f"'questions' isn't a list of strings: {response}")
        return questions
