from __future__ import annotations

from graphrag.llm.client import LLMClient, OpenAILLMClient
from graphrag.prompts import PromptRepository

_STEPBACK_SYSTEM_PROMPT = "step_back_system.txt"


class QueryRewriter:
    """Step-back prompting (ch03): rewrites a specific question into a more
    generic one, so retrieval can surface context a literal match would
    miss.
    """

    def __init__(
        self,
        llm: LLMClient,
        model: str = OpenAILLMClient.DEFAULT_MODEL,
        prompts: PromptRepository | None = None,
    ) -> None:
        self._llm = llm
        self._model = model
        self._system_message = (prompts or PromptRepository()).load(_STEPBACK_SYSTEM_PROMPT)

    def step_back(self, question: str) -> str:
        return self._llm.complete(
            messages=[
                {"role": "system", "content": self._system_message},
                {"role": "user", "content": question},
            ],
            model=self._model,
        )
