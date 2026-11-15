import json

import pytest

from graphrag.llm.errors import LLMResponseError
from graphrag.retrieval.global_search import GlobalSearch, KeyPoint


@pytest.fixture
def store(entity_store):
    for community_id, summary, rating in [
        (0, "Ulysses' journey home.", 8.0),
        (1, "A minor feast.", 2.0),
        (2, "The suitors in Ithaca.", 6.0),
    ]:
        entity_store.query(
            "CREATE (:Community {id: $id, summary: $summary, rating: $rating})",
            {"id": community_id, "summary": summary, "rating": rating},
        )
    return entity_store


def points(*scored):
    return json.dumps({"points": [{"description": d, "score": s} for d, s in scored]})


def test_map_asks_each_highly_rated_report_and_ranks_the_points(llm, store):
    llm.completions = [
        points(("Ulysses travels home", 60), ("Unrelated", 0)),
        points(("Suitors court Penelope", 90)),
    ]

    result = GlobalSearch(store, llm).map("What is this story about?")

    assert result == [
        KeyPoint("Suitors court Penelope", 90.0),
        KeyPoint("Ulysses travels home", 60.0),
    ]
    contexts = [call["messages"][0]["content"] for call in llm.complete_calls]
    assert ["Ulysses' journey home." in c for c in contexts] == [True, False]
    assert ["The suitors in Ithaca." in c for c in contexts] == [False, True]
    assert all(call["json_mode"] for call in llm.complete_calls)
    assert all(
        call["messages"][1] == {"role": "user", "content": "What is this story about?"}
        for call in llm.complete_calls
    )


def test_rating_threshold_selects_the_reports(llm, store):
    llm.completions = [points(), points(), points()]

    GlobalSearch(store, llm, rating_threshold=0).map("q")

    assert len(llm.complete_calls) == 3


def test_reduce_merges_the_ranked_points_into_an_answer(llm, store):
    llm.completions = ["It's about Ulysses."]
    search = GlobalSearch(store, llm, response_type="one paragraph")

    answer = search.reduce("q", [KeyPoint("Suitors", 90.0), KeyPoint("Journey", 60.5)])

    assert answer == "It's about Ulysses."
    [call] = llm.complete_calls
    system = call["messages"][0]["content"]
    assert (
        "----Analyst 1----\nImportance Score: 90\nSuitors\n\n"
        "----Analyst 2----\nImportance Score: 60.5\nJourney"
    ) in system
    assert "one paragraph" in system
    assert call["json_mode"] is False


def test_search_maps_then_reduces(llm, store):
    llm.completions = [points(("A", 10)), points(("B", 20)), "Answer"]

    assert GlobalSearch(store, llm).search("q") == "Answer"
    reduce_prompt = llm.complete_calls[-1]["messages"][0]["content"]
    assert reduce_prompt.index("Importance Score: 20\nB") < reduce_prompt.index(
        "Importance Score: 10\nA"
    )


@pytest.mark.parametrize(
    "response",
    ["not json", json.dumps({"points": "none"}), json.dumps({"points": [{"score": 5}]})],
)
def test_map_rejects_an_unusable_response(llm, store, response):
    llm.completions = [response]

    with pytest.raises(LLMResponseError):
        GlobalSearch(store, llm).map("q")
