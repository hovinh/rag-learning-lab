from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import cast

from openai import OpenAI
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam

Message = dict[str, str]


class LLMClient(ABC):
    """Chat/completion interface. Callers depend on this, never on the
    OpenAI SDK directly, so the backend can be swapped later.
    """

    @abstractmethod
    def complete(self, messages: list[Message], *, model: str, temperature: float = 0.0) -> str: ...

    @abstractmethod
    def stream(
        self, messages: list[Message], *, model: str, temperature: float = 0.0
    ) -> Iterator[str]: ...


class OpenAILLMClient(LLMClient):
    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def complete(
        self, messages: list[Message], *, model: str = DEFAULT_MODEL, temperature: float = 0.0
    ) -> str:
        response = self._client.chat.completions.create(
            model=model,
            messages=cast("list[ChatCompletionMessageParam]", messages),
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def stream(
        self, messages: list[Message], *, model: str = DEFAULT_MODEL, temperature: float = 0.0
    ) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=model,
            messages=cast("list[ChatCompletionMessageParam]", messages),
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            chunk = cast(ChatCompletionChunk, chunk)
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
