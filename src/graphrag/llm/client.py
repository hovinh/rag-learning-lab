from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from openai import OpenAI, omit
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
    ChatCompletionToolUnionParam,
)
from pydantic import BaseModel

from graphrag.llm.errors import LLMResponseError
from graphrag.observability.usage import UsageMeter

Message = dict[str, str]
ToolDefinition = dict[str, Any]
ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class ToolCall:
    """A tool the LLM chose to call, with its arguments already parsed
    from the JSON string the API returns.
    """

    name: str
    arguments: dict[str, Any]


class LLMClient(ABC):
    """Chat/completion interface. Callers depend on this, never on the
    OpenAI SDK directly, so the backend can be swapped later.
    """

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> str:
        """``json_mode`` constrains the response to a JSON object; the
        prompt must still describe the shape it should take.
        """

    @abstractmethod
    def stream(
        self, messages: list[Message], *, model: str, temperature: float = 0.0
    ) -> Iterator[str]: ...

    @abstractmethod
    def choose_tools(
        self,
        messages: list[Message],
        tools: Sequence[ToolDefinition],
        *,
        model: str,
        temperature: float = 0.0,
    ) -> list[ToolCall]:
        """Function calling (ch05): let the LLM pick which of ``tools`` to
        call, and with what arguments. Returns an empty list if it picks
        none.

        Raises ``LLMResponseError`` if a call's arguments aren't valid JSON.
        """

    @abstractmethod
    def parse(
        self,
        messages: list[Message],
        response_model: type[ModelT],
        *,
        model: str,
        temperature: float = 0.0,
    ) -> ModelT:
        """Structured outputs (ch06): constrain the response to
        ``response_model``'s JSON schema and return it parsed. Unlike
        ``json_mode``, the schema itself is enforced, not just valid JSON.

        Raises ``LLMResponseError`` if the LLM refuses or returns nothing
        to parse.
        """


class OpenAILLMClient(LLMClient):
    """``usage``, if given, records every call and its tokens -- what the
    examples' trace sections and run footer report.
    """

    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, client: OpenAI, *, usage: UsageMeter | None = None) -> None:
        self._client = client
        self._usage = usage

    def complete(
        self,
        messages: list[Message],
        *,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> str:
        response = self._client.chat.completions.create(
            model=model,
            messages=cast("list[ChatCompletionMessageParam]", messages),
            temperature=temperature,
            response_format={"type": "json_object"} if json_mode else omit,
        )
        self._record(response)
        return response.choices[0].message.content or ""

    def stream(
        self, messages: list[Message], *, model: str = DEFAULT_MODEL, temperature: float = 0.0
    ) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=model,
            messages=cast("list[ChatCompletionMessageParam]", messages),
            temperature=temperature,
            stream=True,
            # Usage arrives in one last chunk, which has no choices.
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            chunk = cast(ChatCompletionChunk, chunk)
            if chunk.usage is not None:
                self._record(chunk)
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def choose_tools(
        self,
        messages: list[Message],
        tools: Sequence[ToolDefinition],
        *,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
    ) -> list[ToolCall]:
        response = self._client.chat.completions.create(
            model=model,
            messages=cast("list[ChatCompletionMessageParam]", messages),
            temperature=temperature,
            tools=cast("list[ChatCompletionToolUnionParam]", list(tools)) or omit,
        )
        self._record(response)
        calls = []
        for call in response.choices[0].message.tool_calls or []:
            # Custom (free-text) tool calls only come back for custom tool
            # definitions, which this client never sends.
            if not isinstance(call, ChatCompletionMessageFunctionToolCall):
                continue
            try:
                arguments = json.loads(call.function.arguments)
            except json.JSONDecodeError as exc:
                raise LLMResponseError(
                    f"Arguments for tool {call.function.name!r} aren't valid JSON: "
                    f"{call.function.arguments}"
                ) from exc
            calls.append(ToolCall(call.function.name, arguments))
        return calls

    def parse(
        self,
        messages: list[Message],
        response_model: type[ModelT],
        *,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
    ) -> ModelT:
        response = self._client.chat.completions.parse(
            model=model,
            messages=cast("list[ChatCompletionMessageParam]", messages),
            temperature=temperature,
            response_format=response_model,
        )
        self._record(response)
        message = response.choices[0].message
        if message.parsed is None:
            reason = f": {message.refusal}" if message.refusal else ""
            raise LLMResponseError(f"No {response_model.__name__} in the response{reason}")
        return message.parsed

    def _record(self, response: Any) -> None:
        if self._usage is None:
            return
        usage = getattr(response, "usage", None)
        self._usage.record_completion(
            getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0
        )
