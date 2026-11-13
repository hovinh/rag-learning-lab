import pytest

from graphrag.llm.client import LLMClient


class ScriptedLLMClient(LLMClient):
    """Returns queued responses in order -- ``completions`` for
    ``complete``, ``tool_calls`` for ``choose_tools`` -- and records every
    call.
    """

    def __init__(self) -> None:
        self.completions: list[str] = []
        self.tool_calls: list[list] = []
        self.complete_calls: list[dict] = []
        self.choose_tools_calls: list[dict] = []

    def complete(self, messages, *, model, temperature=0.0, json_mode=False):
        self.complete_calls.append({"messages": messages, "model": model, "json_mode": json_mode})
        return self.completions.pop(0)

    def stream(self, messages, *, model, temperature=0.0):
        raise NotImplementedError

    def choose_tools(self, messages, tools, *, model, temperature=0.0):
        self.choose_tools_calls.append({"messages": messages, "tools": tools, "model": model})
        return self.tool_calls.pop(0)

    def parse(self, messages, response_model, *, model, temperature=0.0):
        raise NotImplementedError


@pytest.fixture
def llm():
    return ScriptedLLMClient()
