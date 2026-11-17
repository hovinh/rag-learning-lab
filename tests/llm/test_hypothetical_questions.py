import json

import pytest

from graphrag.llm.errors import LLMResponseError
from graphrag.llm.hypothetical_questions import HypotheticalQuestionGenerator

PURPOSE = "Students revising who did what in the story."


def test_generate_asks_for_questions_serving_the_purpose(llm):
    llm.completions = [json.dumps({"questions": ["Who is Ulysses?", "Where is Ithaca?"]})]
    generator = HypotheticalQuestionGenerator(
        llm, purpose=PURPOSE, questions_per_chunk=2, model="m"
    )

    questions = generator.generate("Ulysses longs for Ithaca.")

    assert questions == ["Who is Ulysses?", "Where is Ithaca?"]
    [call] = llm.complete_calls
    assert call["json_mode"] is True
    assert call["model"] == "m"
    system, user = call["messages"]
    assert PURPOSE in system["content"]
    assert "Write 2 distinct questions" in system["content"]
    assert user == {"role": "user", "content": "Ulysses longs for Ithaca."}


def test_generate_keeps_at_most_the_requested_number_and_drops_blanks(llm):
    llm.completions = [json.dumps({"questions": [" a? ", "", 3, "b?", "c?"]})]
    generator = HypotheticalQuestionGenerator(llm, purpose=PURPOSE, questions_per_chunk=2)

    assert generator.generate("chunk") == ["a?", "b?"]


@pytest.mark.parametrize(
    "response",
    [json.dumps({"questions": []}), json.dumps({"questions": "a?"}), json.dumps({"other": 1})],
)
def test_generate_rejects_a_response_without_questions(llm, response):
    llm.completions = [response]
    generator = HypotheticalQuestionGenerator(llm, purpose=PURPOSE)

    with pytest.raises(LLMResponseError):
        generator.generate("chunk")


def test_questions_per_chunk_must_be_positive(llm):
    with pytest.raises(ValueError):
        HypotheticalQuestionGenerator(llm, purpose=PURPOSE, questions_per_chunk=0)
