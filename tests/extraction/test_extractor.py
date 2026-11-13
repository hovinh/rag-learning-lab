from pydantic import BaseModel

from graphrag.extraction.extractor import StructuredExtractor
from graphrag.llm.client import LLMClient


class Person(BaseModel):
    name: str


class FakeLLMClient(LLMClient):
    def __init__(self, parsed: BaseModel) -> None:
        self.parsed = parsed
        self.calls: list[dict] = []

    def complete(self, messages, *, model, temperature=0.0, json_mode=False):
        raise NotImplementedError

    def stream(self, messages, *, model, temperature=0.0):
        raise NotImplementedError

    def choose_tools(self, messages, tools, *, model, temperature=0.0):
        raise NotImplementedError

    def parse(self, messages, response_model, *, model, temperature=0.0):
        self.calls.append({"messages": messages, "response_model": response_model, "model": model})
        return self.parsed


def test_extract_returns_parsed_model():
    llm = FakeLLMClient(Person(name="Ada"))

    result = StructuredExtractor(llm, Person, "Extract people.").extract("Ada wrote it.")

    assert result == Person(name="Ada")


def test_extract_sends_system_message_and_document():
    llm = FakeLLMClient(Person(name="Ada"))

    StructuredExtractor(llm, Person, "Extract people.", model="test-model").extract("Ada wrote it.")

    assert llm.calls == [
        {
            "messages": [
                {"role": "system", "content": "Extract people."},
                {"role": "user", "content": "Ada wrote it."},
            ],
            "response_model": Person,
            "model": "test-model",
        }
    ]
