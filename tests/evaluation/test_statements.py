import pytest

from graphrag.evaluation.statements import StatementExtractor
from graphrag.llm.errors import LLMResponseError


def test_extract_returns_statements(llm):
    llm.completions = ['{"statements": ["Tom Cruise acted in Top Gun."]}']

    statements = StatementExtractor(llm, model="test-model").extract("Who acted in Top Gun?", "Tom")

    assert statements == ["Tom Cruise acted in Top Gun."]
    [call] = llm.complete_calls
    assert call["model"] == "test-model"
    assert call["json_mode"] is True
    assert call["messages"][1] == {
        "role": "user",
        "content": "Question: Who acted in Top Gun?\nAnswer: Tom",
    }


@pytest.mark.parametrize(
    "response", ["not json", "{}", '{"statements": "x"}', '{"statements": [1]}']
)
def test_extract_rejects_unusable_responses(llm, response):
    llm.completions = [response]

    with pytest.raises(LLMResponseError):
        StatementExtractor(llm).extract("?", "answer")
