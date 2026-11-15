from __future__ import annotations

from typing import ClassVar

from graphrag.extraction.entity_models import (
    ChunkExtraction,
    ExtractedEntity,
    ExtractedRelationship,
)
from graphrag.graph.cypher_store import CypherGraphStore

_MERGE_CHUNK = """
MERGE (b:Book {id: $book_id})
MERGE (c:Chunk {id: $chunk_id})
SET c.book_id = $book_id, c.chunk_index = $chunk_index, c.text = $text
MERGE (b)-[:HAS_CHUNK]->(c)
"""

# An entity seen in several chunks collects one description per mention;
# DescriptionSummarizer later merges them into ``summary``.
_MERGE_ENTITY = """
MATCH (c:Chunk {id: $chunk_id})
MERGE (n:Entity {name: $name})
SET n.type = coalesce(n.type, $type),
    n.description = CASE WHEN n.description IS NULL THEN [$description]
                         ELSE list_append(n.description, $description) END
MERGE (c)-[:MENTIONS]->(n)
"""

# CREATE rather than MERGE, as in the book: every mention of a relationship
# is kept as its own edge, and DescriptionSummarizer collapses them.
_CREATE_RELATIONSHIP = """
MERGE (s:Entity {name: $source})
MERGE (t:Entity {name: $target})
CREATE (s)-[:RELATIONSHIP {description: $description, strength: $strength}]->(t)
"""


class EntityGraphImporter:
    """Declares ch07's entity graph and writes each chunk's extraction into
    it (the book's ``import_nodes_query`` / ``import_relationships_query``):

    ``(:Book)-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(:Entity)``,
    ``(:Entity)-[:RELATIONSHIP {description, strength}]->(:Entity)``.

    The tables filled in by later stages are declared here too, since Kuzu
    needs every table up front: ``SUMMARIZED_RELATIONSHIP``,
    ``Community`` / ``IN_COMMUNITY``, and the ``embedding`` column.

    Neo4j adds each entity's type as an extra label (``SET n:PERSON``);
    Kuzu nodes belong to exactly one table, so the type is a property.
    """

    NODE_TABLES: ClassVar[dict[str, str]] = {
        "Book": "id INT64, PRIMARY KEY(id)",
        "Chunk": "id STRING, book_id INT64, chunk_index INT64, text STRING, PRIMARY KEY(id)",
        "Entity": (
            "name STRING, type STRING, description STRING[], summary STRING, "
            "community_id INT64, embedding FLOAT[{embedding_dim}], PRIMARY KEY(name)"
        ),
        "Community": (
            "id INT64, title STRING, summary STRING, rating DOUBLE, "
            "rating_explanation STRING, PRIMARY KEY(id)"
        ),
    }
    REL_TABLES: ClassVar[dict[str, str]] = {
        "HAS_CHUNK": "FROM Book TO Chunk",
        "MENTIONS": "FROM Chunk TO Entity",
        "RELATIONSHIP": "FROM Entity TO Entity, description STRING, strength DOUBLE",
        "SUMMARIZED_RELATIONSHIP": "FROM Entity TO Entity, summary STRING, weight INT64",
        "IN_COMMUNITY": "FROM Entity TO Community",
    }

    def __init__(self, store: CypherGraphStore, embedding_dim: int) -> None:
        self._store = store
        self._embedding_dim = embedding_dim

    def create_schema(self, *, reset: bool = False) -> None:
        """``reset`` drops the tables first, discarding the whole graph."""
        if reset:
            # Kuzu won't drop a table an index still references, e.g. the
            # vector index EntityEmbeddingIndexer builds over Entity.
            for index in self._store.query(
                "CALL SHOW_INDEXES() WHERE table_name IN $tables "
                "AND index_type = 'HNSW' RETURN table_name, index_name",
                {"tables": list(self.NODE_TABLES)},
            ):
                self._store.query(
                    f"CALL DROP_VECTOR_INDEX('{index['table_name']}', '{index['index_name']}')"
                )
            # Rel tables reference node tables, so they're dropped first and
            # created last.
            for table in [*self.REL_TABLES, *self.NODE_TABLES]:
                self._store.query(f"DROP TABLE IF EXISTS {table}")
        for table, columns in self.NODE_TABLES.items():
            columns = columns.format(embedding_dim=self._embedding_dim)
            self._store.query(f"CREATE NODE TABLE IF NOT EXISTS {table}({columns})")
        for table, columns in self.REL_TABLES.items():
            self._store.query(f"CREATE REL TABLE IF NOT EXISTS {table}({columns})")

    def import_chunk(
        self, book_id: int, chunk_index: int, text: str, extraction: ChunkExtraction
    ) -> None:
        # The book merges the chunk on ``{id: chunk_index}`` under its book,
        # so ids restart per book; a Kuzu primary key must be unique across
        # the table, hence the book is part of the id.
        chunk_id = f"{book_id}-{chunk_index}"
        self._store.query(
            _MERGE_CHUNK,
            {"book_id": book_id, "chunk_id": chunk_id, "chunk_index": chunk_index, "text": text},
        )
        for entity in extraction.entities:
            self._merge_entity(chunk_id, entity)
        for relationship in extraction.relationships:
            self._create_relationship(relationship)

    def _merge_entity(self, chunk_id: str, entity: ExtractedEntity) -> None:
        self._store.query(
            _MERGE_ENTITY,
            {
                "chunk_id": chunk_id,
                "name": entity.name,
                "type": entity.type,
                "description": entity.description,
            },
        )

    def _create_relationship(self, relationship: ExtractedRelationship) -> None:
        self._store.query(
            _CREATE_RELATIONSHIP,
            {
                "source": relationship.source,
                "target": relationship.target,
                "description": relationship.description,
                "strength": relationship.strength,
            },
        )
