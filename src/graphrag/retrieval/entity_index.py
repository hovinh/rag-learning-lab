from __future__ import annotations

from graphrag.graph.cypher_store import CypherGraphStore
from graphrag.retrieval.embedder import Embedder

ENTITY_VECTOR_INDEX = "entity_vector_idx"

_ENTITY_SUMMARIES = """
MATCH (e:Entity)
WHERE e.summary IS NOT NULL
RETURN e.name AS name, e.summary AS summary
ORDER BY name
"""

_SET_EMBEDDINGS = """
UNWIND $rows AS row
MATCH (e:Entity {name: row.name})
SET e.embedding = row.embedding
"""


class EntityEmbeddingIndexer:
    """Embeds every entity's summary into ``Entity.embedding`` and builds a
    vector index over it for local search (ch07's
    ``db.create.setNodeVectorProperty`` + ``CREATE VECTOR INDEX``).

    Entities with no summary are left out. They're the ones only ever
    named as a relationship's endpoint, never extracted themselves, so
    there's nothing to embed; the book calls ``embed(None)`` on them.

    Run it once, after summarization: Kuzu won't write to a column a vector
    index has covered, so re-indexing needs a fresh schema.
    """

    def __init__(self, store: CypherGraphStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    def is_indexed(self) -> bool:
        """Whether ``index`` has built the vector index -- the last step of
        building ch07's graph, so a graph that has it is complete.
        """
        rows = self._store.query(
            "CALL SHOW_INDEXES() WHERE index_name = $name RETURN index_name",
            {"name": ENTITY_VECTOR_INDEX},
        )
        return bool(rows)

    def index(self) -> int:
        """Returns how many entities were embedded."""
        entities = self._store.query(_ENTITY_SUMMARIES)
        if not entities:
            return 0
        embeddings = self._embedder.embed([entity["summary"] for entity in entities])
        rows = [
            {"name": entity["name"], "embedding": embedding}
            for entity, embedding in zip(entities, embeddings, strict=True)
        ]
        self._store.query(_SET_EMBEDDINGS, {"rows": rows})
        self._store.query(
            f"CALL CREATE_VECTOR_INDEX('Entity', '{ENTITY_VECTOR_INDEX}', "
            "'embedding', metric := 'cosine')"
        )
        return len(rows)
