import json

import pytest

from graphrag.extraction.community_reporter import CommunityReport, CommunityReporter
from graphrag.extraction.entity_models import (
    ChunkExtraction,
    ExtractedEntity,
    ExtractedRelationship,
)
from graphrag.llm.errors import LLMResponseError

REPORT = {
    "title": "Telemachus and Minerva",
    "summary": "Minerva guides Telemachus.",
    "rating": 7,
    "rating_explanation": "Central to the plot.",
    "findings": [],
}


@pytest.fixture
def graph(entity_store, entity_importer):
    """TELEMACHUS and MINERVA in community 0, ITHACA alone in community 1."""
    entity_importer.import_chunk(
        0,
        0,
        "a",
        ChunkExtraction(
            [
                ExtractedEntity("TELEMACHUS", "PERSON", "Son of Ulysses"),
                ExtractedEntity("MINERVA", "GOD", "A goddess"),
                ExtractedEntity("ITHACA", "LOCATION", "An island"),
            ],
            [ExtractedRelationship("MINERVA", "TELEMACHUS", "She advises him", 9.0)],
        ),
    )
    entity_store.query("MATCH (e:Entity) SET e.summary = e.description[1]")
    entity_store.query(
        "MATCH (e:Entity) SET e.community_id = CASE WHEN e.name = 'ITHACA' THEN 1 ELSE 0 END"
    )
    return entity_store


def test_report_communities_stores_a_report_per_community(llm, graph):
    llm.completions = [json.dumps(REPORT)]

    reports = CommunityReporter(graph, llm).report_communities()

    assert reports == [
        CommunityReport(
            community_id=0,
            title="Telemachus and Minerva",
            summary="Minerva guides Telemachus.",
            rating=7.0,
            rating_explanation="Central to the plot.",
        )
    ]
    rows = graph.query(
        "MATCH (e:Entity)-[:IN_COMMUNITY]->(c:Community) "
        "RETURN c.id AS id, c.title AS title, c.rating AS rating, collect(e.name) AS members"
    )
    assert len(rows) == 1
    assert {**rows[0], "members": sorted(rows[0]["members"])} == {
        "id": 0,
        "title": "Telemachus and Minerva",
        "rating": 7.0,
        "members": ["MINERVA", "TELEMACHUS"],
    }


def test_report_prompt_lists_the_communitys_entities_and_relationships(llm, graph):
    llm.completions = [json.dumps(REPORT)]

    CommunityReporter(graph, llm).report_communities()

    [call] = llm.complete_calls
    prompt = call["messages"][0]["content"]
    assert call["json_mode"] is True
    assert "{'id': 'MINERVA', 'description': 'A goddess', 'type': 'GOD'}" in prompt
    assert (
        "{'source': 'MINERVA', 'target': 'TELEMACHUS', 'description': 'She advises him'}" in prompt
    )
    assert "ITHACA" not in prompt


def test_report_communities_includes_smaller_communities_on_request(llm, graph):
    llm.completions = [json.dumps(REPORT), json.dumps({**REPORT, "title": "Ithaca"})]

    reports = CommunityReporter(graph, llm).report_communities(min_size=1)

    assert [report.title for report in reports] == ["Telemachus and Minerva", "Ithaca"]


@pytest.mark.parametrize(
    "response",
    ["not json", json.dumps({**REPORT, "rating": "high"}), json.dumps({"title": "Only a title"})],
)
def test_report_communities_rejects_an_unusable_report(llm, graph, response):
    llm.completions = [response]

    with pytest.raises(LLMResponseError):
        CommunityReporter(graph, llm).report_communities()
