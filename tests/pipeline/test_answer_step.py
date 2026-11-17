from dataclasses import dataclass, field

from graphrag.llm.client import LLMClient
from graphrag.pipeline.steps import AnswerStep
from graphrag.prompts.repository import PromptRepository


@dataclass
class Match:
    text: str
    score: float


@dataclass
class AnswerTestContext:
    question: str
    matches: list[Match] = field(default_factory=list)
    answer: str = ""


class FakeLLMClient(LLMClient):
    def __init__(self, deltas: list[str]) -> None:
        self.deltas = deltas
        self.calls: list[tuple[list, str]] = []

    def complete(self, messages, *, model, temperature=0.0, json_mode=False):
        raise NotImplementedError

    def stream(self, messages, *, model, temperature=0.0):
        self.calls.append((messages, model))
        yield from self.deltas

    def choose_tools(self, messages, tools, *, model, temperature=0.0):
        raise NotImplementedError

    def parse(self, messages, response_model, *, model, temperature=0.0):
        raise NotImplementedError


def test_answer_step_builds_prompt_from_prompt_files_and_streams_answer(tmp_path):
    (tmp_path / "answer_system.txt").write_text("Be helpful.", encoding="utf-8")
    (tmp_path / "answer_user.txt").write_text(
        "Docs:\n{context}\n\nQuestion: {question}", encoding="utf-8"
    )
    llm = FakeLLMClient(["Hello", ", ", "world!"])
    context = AnswerTestContext(
        question="What is this?", matches=[Match("doc one", 0.9), Match("doc two", 0.5)]
    )

    AnswerStep(llm, PromptRepository(tmp_path), model="test-model")(context)

    assert context.answer == "Hello, world!"
    [(messages, model)] = llm.calls
    assert model == "test-model"
    assert messages[0] == {"role": "system", "content": "Be helpful."}
    assert messages[1] == {
        "role": "user",
        "content": "Docs:\ndoc one\ndoc two\n\nQuestion: What is this?",
    }
