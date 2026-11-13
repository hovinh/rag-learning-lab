from types import SimpleNamespace

import pytest
from openai import omit
from openai.types.chat import ChatCompletionMessageFunctionToolCall
from openai.types.chat.chat_completion_message_function_tool_call import Function
from pydantic import BaseModel

from graphrag.llm.client import OpenAILLMClient, ToolCall
from graphrag.llm.errors import LLMResponseError


class FakeCompletions:
    def __init__(self, message) -> None:
        self.message = message
        self.kwargs: dict = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=self.message)])

    parse = create


class Answer(BaseModel):
    text: str


def make_client(message):
    completions = FakeCompletions(message)
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return OpenAILLMClient(sdk), completions


def function_call(name, arguments):
    return ChatCompletionMessageFunctionToolCall(
        id="call_1", type="function", function=Function(name=name, arguments=arguments)
    )


def test_complete_requests_json_object_in_json_mode():
    client, completions = make_client(SimpleNamespace(content='{"a": 1}'))

    assert client.complete([], model="m", json_mode=True) == '{"a": 1}'
    assert completions.kwargs["response_format"] == {"type": "json_object"}


def test_complete_omits_response_format_by_default():
    client, completions = make_client(SimpleNamespace(content="text"))

    client.complete([], model="m")

    assert completions.kwargs["response_format"] is omit


def test_choose_tools_parses_function_calls():
    message = SimpleNamespace(tool_calls=[function_call("movie_info_by_title", '{"title": "x"}')])
    client, completions = make_client(message)
    tools = [{"type": "function", "function": {"name": "movie_info_by_title"}}]

    calls = client.choose_tools([{"role": "user", "content": "?"}], tools, model="m")

    assert calls == [ToolCall("movie_info_by_title", {"title": "x"})]
    assert completions.kwargs["tools"] == tools


def test_choose_tools_returns_empty_list_without_tool_calls():
    client, completions = make_client(SimpleNamespace(tool_calls=None))

    assert client.choose_tools([], [], model="m") == []
    assert completions.kwargs["tools"] is omit


def test_choose_tools_rejects_invalid_argument_json():
    client, _ = make_client(SimpleNamespace(tool_calls=[function_call("t", "{not json")]))

    with pytest.raises(LLMResponseError, match="'t'"):
        client.choose_tools([], [{"type": "function"}], model="m")


def test_parse_returns_parsed_model_and_sends_it_as_response_format():
    client, completions = make_client(SimpleNamespace(parsed=Answer(text="hi"), refusal=None))

    assert client.parse([], Answer, model="m") == Answer(text="hi")
    assert completions.kwargs["response_format"] is Answer


def test_parse_raises_with_refusal_reason():
    client, _ = make_client(SimpleNamespace(parsed=None, refusal="I can't help with that"))

    with pytest.raises(LLMResponseError, match="No Answer in the response: I can't help"):
        client.parse([], Answer, model="m")
