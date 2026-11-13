import pytest

from graphrag.agent.query_updater import QueryUpdater
from graphrag.llm.errors import LLMResponseError


def test_update_returns_question_from_json_response(llm):
    llm.completions = ['{"question": "What other movies is Keanu Reeves in?"}']
    history = [{"role": "assistant", "content": "Keanu Reeves is the main actor."}]

    updated = QueryUpdater(llm, model="test-model").update(
        "What other movies is that person in?", history
    )

    assert updated == "What other movies is Keanu Reeves in?"
    [call] = llm.complete_calls
    assert call["model"] == "test-model"
    assert call["json_mode"] is True
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][1] == history[0]
    assert call["messages"][2] == {
        "role": "user",
        "content": "The user question to rewrite: 'What other movies is that person in?'",
    }


@pytest.mark.parametrize("response", ["not json", '{"q": "x"}', '{"question": ["x"]}'])
def test_update_rejects_unusable_response(llm, response):
    llm.completions = [response]

    with pytest.raises(LLMResponseError):
        QueryUpdater(llm).update("?")
