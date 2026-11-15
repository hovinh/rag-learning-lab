import pytest

from graphrag.extraction.entity_models import ChunkExtraction, ExtractedEntity
from graphrag.retrieval.entity_index import EntityEmbeddingIndexer
from graphrag.retrieval.local_search import LocalContext, LocalSearch

ENTITIES = {
    "ULYSSES": ("Ulysses is a hero", [1.0, 0.0, 0.0]),
    "PENELOPE": ("Penelope is his wife", [0.9, 0.1, 0.0]),
    "CYCLOPS": ("A one-eyed giant", [0.0, 0.0, 1.0]),
}


@pytest.fixture
def store(entity_store, entity_importer, embedder):
    def entity(name):
        return ExtractedEntity(name, "PERSON", ENTITIES[name][0])

    entity_importer.import_chunk(
        0, 0, "Ulysses and Penelope.", ChunkExtraction([entity("ULYSSES"), entity("PENELOPE")])
    )
    entity_importer.import_chunk(0, 1, "Ulysses alone.", ChunkExtraction([entity("ULYSSES")]))
    entity_importer.import_chunk(0, 2, "The Cyclops.", ChunkExtraction([entity("CYCLOPS")]))
    entity_store.query("MATCH (e:Entity) SET e.summary = e.description[1]")
    entity_store.query(
        "MATCH (s:Entity {name: 'PENELOPE'}), (t:Entity {name: 'ULYSSES'}) "
        "CREATE (s)-[:SUMMARIZED_RELATIONSHIP {summary: 'Married', weight: 3}]->(t)"
    )
    entity_store.query(
        "MATCH (s:Entity {name: 'CYCLOPS'}), (t:Entity {name: 'ULYSSES'}) "
        "CREATE (s)-[:SUMMARIZED_RELATIONSHIP {summary: 'Blinded by him', weight: 1}]->(t)"
    )
    for community_id, summary, rating, members in [
        (0, "Ulysses' household", 6.0, ["ULYSSES", "PENELOPE"]),
        (1, "Monsters", 9.0, ["CYCLOPS"]),
        (2, "Ulysses' voyage", 8.0, ["ULYSSES"]),
    ]:
        entity_store.query(
            "CREATE (c:Community {id: $id, summary: $summary, rating: $rating}) "
            "WITH c MATCH (e:Entity) WHERE e.name IN $members CREATE (e)-[:IN_COMMUNITY]->(c)",
            {"id": community_id, "summary": summary, "rating": rating, "members": members},
        )
    embedder.vectors = {summary: vector for summary, vector in ENTITIES.values()}
    EntityEmbeddingIndexer(entity_store, embedder).index()
    embedder.vectors["Who is Ulysses?"] = [1.0, 0.05, 0.0]
    return entity_store


def test_build_context_gathers_the_graph_around_the_closest_entities(llm, store, embedder):
    search = LocalSearch(store, llm, embedder, k_entities=2)

    context = search.build_context("Who is Ulysses?")

    assert context == LocalContext(
        entities=["Ulysses is a hero", "Penelope is his wife"],
        # The chunk mentioning both entities ranks first.
        chunks=["Ulysses and Penelope.", "Ulysses alone."],
        # Highest rated first; the Cyclops' community isn't theirs.
        reports=["Ulysses' voyage", "Ulysses' household"],
        # Only relationships with both ends among the found entities.
        relationships=["Married"],
    )


def test_build_context_caps_each_part(llm, store, embedder):
    search = LocalSearch(
        store, llm, embedder, k_entities=2, top_chunks=1, top_communities=1, top_inside_rels=0
    )

    context = search.build_context("Who is Ulysses?")

    assert context.chunks == ["Ulysses and Penelope."]
    assert context.reports == ["Ulysses' voyage"]
    assert context.relationships == []


def test_search_answers_from_the_context(llm, store, embedder):
    llm.completions = ["Ulysses is the hero of the Odyssey."]

    answer = LocalSearch(store, llm, embedder, k_entities=2).search("Who is Ulysses?")

    assert answer == "Ulysses is the hero of the Odyssey."
    [call] = llm.complete_calls
    system, user = call["messages"]
    assert system["role"] == "system"
    assert "'Chunks': ['Ulysses and Penelope.', 'Ulysses alone.']" in system["content"]
    assert "'Relationships': ['Married']" in system["content"]
    assert "multiple paragraphs" in system["content"]
    assert user == {"role": "user", "content": "Who is Ulysses?"}


def test_to_prompt_renders_the_books_context_shape():
    context = LocalContext(entities=["e"], chunks=["c"], reports=["r"], relationships=["l"])

    assert context.to_prompt() == (
        "{'Chunks': ['c'], 'Reports': ['r'], 'Relationships': ['l'], 'Entities': ['e']}"
    )
