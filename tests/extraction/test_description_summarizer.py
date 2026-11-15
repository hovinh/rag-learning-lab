from graphrag.extraction.description_summarizer import DescriptionSummarizer
from graphrag.extraction.entity_models import (
    ChunkExtraction,
    ExtractedEntity,
    ExtractedRelationship,
)


def test_summarize_entities_merges_multiple_descriptions_with_the_llm(
    llm, entity_store, entity_importer
):
    entity_importer.import_chunk(
        0, 0, "a", ChunkExtraction([ExtractedEntity("ULYSSES", "PERSON", "A hero")])
    )
    entity_importer.import_chunk(
        0, 1, "b", ChunkExtraction([ExtractedEntity("ULYSSES", "PERSON", "Wily")])
    )
    llm.completions = ["Ulysses is a wily hero."]

    summarized = DescriptionSummarizer(entity_store, llm).summarize_entities()

    [call] = llm.complete_calls
    prompt = call["messages"][0]["content"]
    assert "Entities: ULYSSES" in prompt
    assert "Description List: ['A hero', 'Wily']" in prompt
    assert summarized == 1
    assert entity_store.query("MATCH (e:Entity) RETURN e.summary AS summary") == [
        {"summary": "Ulysses is a wily hero."}
    ]


def test_summarize_entities_keeps_a_single_description_without_the_llm(
    llm, entity_store, entity_importer
):
    entity_importer.import_chunk(
        0,
        0,
        "a",
        ChunkExtraction(
            [ExtractedEntity("ITHACA", "LOCATION", "An island")],
            [ExtractedRelationship("ITHACA", "TROY", "Far apart", 1.0)],
        ),
    )

    summarized = DescriptionSummarizer(entity_store, llm).summarize_entities()

    assert summarized == 0
    assert llm.complete_calls == []
    # TROY was only a relationship endpoint, so it has nothing to summarize.
    assert entity_store.query(
        "MATCH (e:Entity) RETURN e.name AS name, e.summary AS summary ORDER BY name"
    ) == [{"name": "ITHACA", "summary": "An island"}, {"name": "TROY", "summary": None}]


def test_summarize_relationships_collapses_each_pair_into_one_edge(
    llm, entity_store, entity_importer
):
    entity_importer.import_chunk(
        0,
        0,
        "a",
        ChunkExtraction(
            relationships=[
                ExtractedRelationship("TELEMACHUS", "MINERVA", "He welcomes her", 7.0),
                ExtractedRelationship("MINERVA", "TELEMACHUS", "She advises him", 9.0),
                ExtractedRelationship("TELEMACHUS", "ITHACA", "He lives there", 6.0),
            ]
        ),
    )
    llm.completions = ["Minerva guides Telemachus, who welcomes her."]

    summarized = DescriptionSummarizer(entity_store, llm).summarize_relationships()

    [call] = llm.complete_calls
    prompt = call["messages"][0]["content"]
    assert "Entities: MINERVA relationship to TELEMACHUS" in prompt
    assert summarized == 1
    rows = entity_store.query(
        "MATCH (s:Entity)-[r:SUMMARIZED_RELATIONSHIP]->(t:Entity) "
        "RETURN s.name AS source, t.name AS target, r.summary AS summary, r.weight AS weight "
        "ORDER BY source"
    )
    assert rows == [
        {"source": "ITHACA", "target": "TELEMACHUS", "summary": "He lives there", "weight": 1},
        {
            "source": "MINERVA",
            "target": "TELEMACHUS",
            "summary": "Minerva guides Telemachus, who welcomes her.",
            "weight": 2,
        },
    ]
