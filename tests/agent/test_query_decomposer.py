import pytest

from graphrag.agent.query_decomposer import QueryDecomposer
from graphrag.llm.errors import LLMResponseError


def test_decompose_returns_questions_from_json_response(llm):
    llm.completions = ['{"questions": ["Who is Mick Jagger?", "How old is Mick Jagger?"]}']
    history = [{"role": "assistant", "content": "Mick Jagger is a singer."}]

    questions = QueryDecomposer(llm, model="test-model").decompose(
        "Who is Mick Jagger and how old is he?", history
    )

    assert questions == ["Who is Mick Jagger?", "How old is Mick Jagger?"]
    [call] = llm.complete_calls
    assert call["model"] == "test-model"
    assert call["json_mode"] is True
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][1] == history[0]
    assert call["messages"][2] == {
        "role": "user",
        "content": "The user question to rewrite: 'Who is Mick Jagger and how old is he?'",
    }


def test_plan_is_decompose(llm):
    llm.completions = ['{"questions": ["Hello"]}']

    assert QueryDecomposer(llm).plan("Hello") == ["Hello"]


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        '{"question": "x"}',
        '{"questions": "x"}',
        '{"questions": []}',
        '{"questions": ["x", 1]}',
        '{"questions": ["x", ""]}',
    ],
)
def test_decompose_rejects_unusable_response(llm, response):
    llm.completions = [response]

    with pytest.raises(LLMResponseError):
        QueryDecomposer(llm).decompose("?")
