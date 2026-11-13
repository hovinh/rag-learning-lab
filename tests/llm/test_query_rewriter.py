from graphrag.llm.client import LLMClient, Message
from graphrag.llm.query_rewriter import QueryRewriter


class FakeLLMClient(LLMClient):
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[list[Message], str, float]] = []

    def complete(self, messages, *, model, temperature=0.0, json_mode=False):
        self.calls.append((messages, model, temperature))
        return self.response

    def stream(self, messages, *, model, temperature=0.0):
        raise NotImplementedError

    def choose_tools(self, messages, tools, *, model, temperature=0.0):
        raise NotImplementedError

    def parse(self, messages, response_model, *, model, temperature=0.0):
        raise NotImplementedError


def test_step_back_returns_llm_response():
    llm = FakeLLMClient("What is Jan Sindel's personal history?")

    result = QueryRewriter(llm).step_back("Jan Sindel's was born in what country?")

    assert result == "What is Jan Sindel's personal history?"


def test_step_back_sends_system_and_user_messages():
    llm = FakeLLMClient("step back question")

    QueryRewriter(llm, model="gpt-4o-mini").step_back("original question")

    [(messages, model, _temperature)] = llm.calls
    assert model == "gpt-4o-mini"
    assert messages[0]["role"] == "system"
    assert "step back" in messages[0]["content"].lower()
    assert messages[1] == {"role": "user", "content": "original question"}
