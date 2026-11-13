import pytest

from graphrag.llm.errors import LLMResponseError
from graphrag.llm.json_response import load_json_field


def test_load_json_field_returns_the_field():
    assert load_json_field('{"questions": ["a", "b"]}', "questions") == ["a", "b"]


@pytest.mark.parametrize("response", ["not json", "[1, 2]", '{"other": 1}'])
def test_load_json_field_rejects_unusable_response(response):
    with pytest.raises(LLMResponseError):
        load_json_field(response, "questions")
