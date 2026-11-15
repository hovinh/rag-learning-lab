import pytest

from graphrag.agent.critic import AnswerCritic
from graphrag.llm.errors import LLMResponseError


def test_critique_returns_follow_up_questions(llm):
    llm.completions = ['{"questions": ["What other movies is Keanu Reeves in?"]}']
    history = [{"role": "assistant", "content": "Keanu Reeves is the main actor."}]

    questions = AnswerCritic(llm, model="test-model").critique("Who and what else?", history)

    assert questions == ["What other movies is Keanu Reeves in?"]
    [call] = llm.complete_calls
    assert call["model"] == "test-model"
    assert call["json_mode"] is True
    assert call["messages"][1] == history[0]
    assert call["messages"][2] == {
        "role": "user",
        "content": "The original user question to answer: Who and what else?",
    }


def test_critique_returns_empty_list_when_answered(llm):
    llm.completions = ['{"questions": []}']

    assert AnswerCritic(llm).critique("?", []) == []


def test_critique_tells_the_llm_what_data_exists(llm):
    llm.completions = ['{"questions": []}']
    schema = "Node properties:\nMovie {title: STRING}"

    AnswerCritic(llm, data_description=f"{schema}\n").critique("?", [])

    system = llm.complete_calls[0]["messages"][0]["content"]
    assert f"The answers come from a database with this schema:\n{schema}\n\n" in system
    assert "Only ask follow-up questions that this data can answer." in system


def test_critique_without_data_description_leaves_the_schema_out(llm):
    llm.completions = ['{"questions": []}']

    AnswerCritic(llm).critique("?", [])

    system = llm.complete_calls[0]["messages"][0]["content"]
    assert "schema" not in system
    assert system.endswith('"questions": ["question1", "question2"]\n}')


def test_critique_treats_empty_object_as_answered(llm):
    llm.completions = ["{}"]

    assert AnswerCritic(llm).critique("?", []) == []


@pytest.mark.parametrize(
    "response", ["not json", '{"other": 1}', '{"questions": "x"}', '{"questions": [1]}']
)
def test_critique_rejects_unusable_response(llm, response):
    llm.completions = [response]

    with pytest.raises(LLMResponseError):
        AnswerCritic(llm).critique("?", [])
