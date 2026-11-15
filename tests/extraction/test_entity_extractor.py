from graphrag.extraction.entity_extractor import EntityRelationshipExtractor
from graphrag.extraction.entity_models import ExtractedEntity, ExtractedRelationship

LLM_OUTPUT = """("entity";TELEMACHUS;PERSON;Telemachus is the son of Ulysses)
|
("entity";MINERVA;GOD;Minerva is a goddess who advises Telemachus)
|
("relationship";MINERVA;TELEMACHUS;Minerva encourages Telemachus to find his father;8)

"""


def test_parse_reads_entities_and_relationships():
    extraction = EntityRelationshipExtractor.parse(LLM_OUTPUT)

    assert extraction.entities == [
        ExtractedEntity("TELEMACHUS", "PERSON", "Telemachus is the son of Ulysses"),
        ExtractedEntity("MINERVA", "GOD", "Minerva is a goddess who advises Telemachus"),
    ]
    assert extraction.relationships == [
        ExtractedRelationship(
            "MINERVA", "TELEMACHUS", "Minerva encourages Telemachus to find his father", 8.0
        )
    ]


def test_parse_keeps_delimiters_inside_descriptions():
    extraction = EntityRelationshipExtractor.parse(
        '("entity";ITHACA;LOCATION;An island; home of Ulysses)|'
        '("relationship";ULYSSES;ITHACA;He rules it; he longs for it;9)'
    )

    assert extraction.entities[0].description == "An island; home of Ulysses"
    assert extraction.relationships[0].description == "He rules it; he longs for it"
    assert extraction.relationships[0].strength == 9.0


def test_parse_normalizes_names_to_upper_case():
    extraction = EntityRelationshipExtractor.parse(
        '("entity";"Ulysses";person;A hero)|("relationship";Ulysses;ithaca;Home;5)'
    )

    assert extraction.entities[0].name == "ULYSSES"
    assert extraction.entities[0].type == "PERSON"
    assert (extraction.relationships[0].source, extraction.relationships[0].target) == (
        "ULYSSES",
        "ITHACA",
    )


def test_parse_skips_malformed_and_unknown_records():
    extraction = EntityRelationshipExtractor.parse(
        '("entity";ULYSSES)|("claim";A;B;C)|("relationship";A;B)|'
        '("entity";NEPTUNE;GOD;God of the sea)'
    )

    assert extraction.entities == [ExtractedEntity("NEPTUNE", "GOD", "God of the sea")]
    assert extraction.relationships == []


def test_parse_keeps_a_relationship_with_a_non_numeric_strength():
    extraction = EntityRelationshipExtractor.parse('("relationship";A;B;Related;strong)')

    assert extraction.relationships == [ExtractedRelationship("A", "B", "Related", None)]


def test_extract_fills_the_prompt_and_parses_the_response(llm):
    llm.completions = [LLM_OUTPUT]
    extractor = EntityRelationshipExtractor(llm, ["PERSON", "GOD"], model="test-model")

    extraction = extractor.extract("Minerva visits Telemachus.")

    [call] = llm.complete_calls
    [message] = call["messages"]
    assert call["model"] == "test-model"
    assert message["role"] == "user"
    assert "Entity_types: PERSON,GOD\nText: Minerva visits Telemachus." in message["content"]
    assert '("entity";CENTRAL INSTITUTION;ORGANIZATION;' in message["content"]
    assert "{" not in message["content"].split("-Real Data-")[1]
    assert len(extraction.entities) == 2
