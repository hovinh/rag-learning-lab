from graphrag.extraction.entity_models import (
    ChunkExtraction,
    ExtractedEntity,
    ExtractedRelationship,
)
from graphrag.retrieval.entity_index import ENTITY_VECTOR_INDEX, EntityEmbeddingIndexer


def test_index_embeds_entity_summaries_and_builds_a_vector_index(
    entity_store, entity_importer, embedder
):
    entity_importer.import_chunk(
        0,
        0,
        "a",
        ChunkExtraction(
            [
                ExtractedEntity("ULYSSES", "PERSON", "A hero"),
                ExtractedEntity("ITHACA", "LOCATION", "An island"),
            ],
            [ExtractedRelationship("ULYSSES", "TROY", "Sacked it", 8.0)],
        ),
    )
    entity_store.query(
        "MATCH (e:Entity) WHERE e.description IS NOT NULL SET e.summary = e.description[1]"
    )
    embedder.vectors = {"A hero": [1.0, 0.0, 0.0], "An island": [0.0, 1.0, 0.0]}

    indexed = EntityEmbeddingIndexer(entity_store, embedder).index()

    # TROY has no summary, so it isn't embedded.
    assert indexed == 2
    assert embedder.calls == [["An island", "A hero"]]
    rows = entity_store.query(
        f"CALL QUERY_VECTOR_INDEX('Entity', '{ENTITY_VECTOR_INDEX}', $embedding, 1) "
        "RETURN node.name AS name",
        {"embedding": [0.9, 0.1, 0.0]},
    )
    assert rows == [{"name": "ULYSSES"}]


def test_schema_reset_drops_the_vector_index_so_entities_can_be_reindexed(
    entity_store, entity_importer, embedder
):
    def import_and_index():
        entity_importer.import_chunk(
            0, 0, "a", ChunkExtraction([ExtractedEntity("ULYSSES", "PERSON", "A hero")])
        )
        entity_store.query("MATCH (e:Entity) SET e.summary = e.description[1]")
        return EntityEmbeddingIndexer(entity_store, embedder).index()

    embedder.vectors = {"A hero": [1.0, 0.0, 0.0]}
    import_and_index()

    entity_importer.create_schema(reset=True)

    assert import_and_index() == 1


def test_index_does_nothing_without_summaries(entity_store, embedder):
    assert EntityEmbeddingIndexer(entity_store, embedder).index() == 0
    assert embedder.calls == []
