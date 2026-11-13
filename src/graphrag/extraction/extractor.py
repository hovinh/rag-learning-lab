from __future__ import annotations

from typing import Generic

from graphrag.llm.client import LLMClient, ModelT, OpenAILLMClient


class StructuredExtractor(Generic[ModelT]):
    """Extracts a ``response_model`` instance from a whole document in one
    LLM call (ch06's ``extract``). Not tied to contracts: the model and
    the system message decide what gets extracted.
    """

    def __init__(
        self,
        llm: LLMClient,
        response_model: type[ModelT],
        system_message: str,
        *,
        model: str = OpenAILLMClient.DEFAULT_MODEL,
    ) -> None:
        self._llm = llm
        self._response_model = response_model
        self._system_message = system_message
        self._model = model

    def extract(self, document: str) -> ModelT:
        """Raises ``LLMResponseError`` if the LLM returns nothing to parse."""
        return self._llm.parse(
            [
                {"role": "system", "content": self._system_message},
                {"role": "user", "content": document},
            ],
            self._response_model,
            model=self._model,
        )
