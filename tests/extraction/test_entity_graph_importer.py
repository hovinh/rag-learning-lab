from graphrag.extraction.entity_models import (
    ChunkExtraction,
    ExtractedEntity,
    ExtractedRelationship,
)

ULYSSES = ExtractedEntity("ULYSSES", "PERSON", "Ulysses is a hero")
ITHACA = ExtractedEntity("ITHACA", "LOCATION", "Ithaca is an island")


def test_import_chunk_links_book_chunk_and_entities(entity_store, entity_importer):
    entity_importer.import_chunk(
        0, 0, "Ulysses longs for Ithaca.", ChunkExtraction([ULYSSES, ITHACA])
    )

    rows = entity_store.query(
        "MATCH (b:Book)-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity) "
        "RETURN b.id AS book, c.id AS chunk, c.text AS text, e.name AS name, e.type AS type, "
        "e.description AS description ORDER BY name"
    )
    assert rows == [
        {
            "book": 0,
            "chunk": "0-0",
            "text": "Ulysses longs for Ithaca.",
            "name": "ITHACA",
            "type": "LOCATION",
            "description": ["Ithaca is an island"],
        },
        {
            "book": 0,
            "chunk": "0-0",
            "text": "Ulysses longs for Ithaca.",
            "name": "ULYSSES",
            "type": "PERSON",
            "description": ["Ulysses is a hero"],
        },
    ]


def test_entity_mentioned_in_several_chunks_collects_every_description(
    entity_store, entity_importer
):
    entity_importer.import_chunk(0, 0, "a", ChunkExtraction([ULYSSES]))
    entity_importer.import_chunk(
        1, 0, "b", ChunkExtraction([ExtractedEntity("ULYSSES", "GOD", "Wily")])
    )

    rows = entity_store.query(
        "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) "
        "RETURN e.type AS type, e.description AS description, collect(c.id) AS chunks"
    )
    assert len(rows) == 1
    assert rows[0]["type"] == "PERSON"
    assert rows[0]["description"] == ["Ulysses is a hero", "Wily"]
    assert sorted(rows[0]["chunks"]) == ["0-0", "1-0"]


def test_every_relationship_mention_is_its_own_edge(entity_store, entity_importer):
    relationship = ExtractedRelationship("ULYSSES", "ITHACA", "He rules Ithaca", 9.0)

    entity_importer.import_chunk(0, 0, "a", ChunkExtraction([ULYSSES, ITHACA], [relationship]))
    entity_importer.import_chunk(0, 1, "b", ChunkExtraction([], [relationship]))

    rows = entity_store.query(
        "MATCH (s:Entity)-[r:RELATIONSHIP]->(t:Entity) "
        "RETURN s.name AS source, t.name AS target, r.description AS description, "
        "r.strength AS strength"
    )
    assert rows == 2 * [
        {"source": "ULYSSES", "target": "ITHACA", "description": "He rules Ithaca", "strength": 9.0}
    ]


def test_relationship_to_an_unextracted_entity_creates_a_bare_node(entity_store, entity_importer):
    entity_importer.import_chunk(
        0,
        0,
        "a",
        ChunkExtraction([ULYSSES], [ExtractedRelationship("ULYSSES", "TROY", "Sacked", 7.0)]),
    )

    rows = entity_store.query(
        "MATCH (e:Entity {name: 'TROY'}) RETURN e.type AS type, e.description AS description"
    )
    assert rows == [{"type": None, "description": None}]


def test_create_schema_reset_discards_the_graph(entity_store, entity_importer):
    entity_importer.import_chunk(0, 0, "a", ChunkExtraction([ULYSSES]))

    entity_importer.create_schema(reset=True)

    assert entity_store.query("MATCH (e:Entity) RETURN count(e) AS n") == [{"n": 0}]


def test_counts_entities_and_relationship_mentions(entity_importer):
    assert (entity_importer.count_entities(), entity_importer.count_relationships()) == (0, 0)
    relationship = ExtractedRelationship("ULYSSES", "ITHACA", "Ulysses rules Ithaca", 9.0)

    entity_importer.import_chunk(0, 0, "text", ChunkExtraction([ULYSSES, ITHACA], [relationship]))
    entity_importer.import_chunk(0, 1, "text", ChunkExtraction([ULYSSES], [relationship]))

    assert entity_importer.count_entities() == 2
    assert entity_importer.count_relationships() == 2
