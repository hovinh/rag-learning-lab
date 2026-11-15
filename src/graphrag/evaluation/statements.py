from __future__ import annotations

from graphrag.llm.client import LLMClient, OpenAILLMClient
from graphrag.llm.errors import LLMResponseError
from graphrag.llm.json_response import load_json_field
from graphrag.prompts import PromptRepository

_SYSTEM_PROMPT = "eval_statements_system.txt"
_USER_PROMPT = "eval_statements_user.txt"


class StatementExtractor:
    """Breaks a text into standalone statements, the first step of the
    RAGAS faithfulness and answer-correctness metrics (ch08): each
    statement is then judged on its own.
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

    def extract(self, question: str, text: str) -> list[str]:
        """``question`` gives the statements their subject, e.g. a bare
        "Tom Cruise" answer becomes "Tom Cruise acted in Top Gun".

        Raises ``LLMResponseError`` if the response has no list of
        statements.
        """
        response = self._llm.complete(
            [
                {"role": "system", "content": self._system_message},
                {
                    "role": "user",
                    "content": self._user_template.format(question=question, answer=text),
                },
            ],
            model=self._model,
            json_mode=True,
        )
        statements = load_json_field(response, "statements")
        if not isinstance(statements, list) or not all(isinstance(s, str) for s in statements):
            raise LLMResponseError(f"'statements' isn't a list of strings: {response}")
        return statements
