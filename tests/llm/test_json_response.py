import pytest

from graphrag.llm.errors import LLMResponseError
from graphrag.llm.json_response import load_json_field, load_json_object


def test_load_json_field_returns_the_field():
    assert load_json_field('{"questions": ["a", "b"]}', "questions") == ["a", "b"]


@pytest.mark.parametrize("response", ["not json", "[1, 2]", '{"other": 1}'])
def test_load_json_field_rejects_unusable_response(response):
    with pytest.raises(LLMResponseError):
        load_json_field(response, "questions")


def test_load_json_object_returns_the_object():
    assert load_json_object('{"title": "T", "rating": 5}') == {"title": "T", "rating": 5}


@pytest.mark.parametrize("response", ["not json", "[1, 2]", '"text"'])
def test_load_json_object_rejects_non_objects(response):
    with pytest.raises(LLMResponseError):
        load_json_object(response)
