from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Self

import kuzu

from graphrag.graph.models import ChunkMatch
from graphrag.graph.store import GraphStore


class KuzuGraphStore(GraphStore):
    """GraphStore backed by embedded Kuzu (replaces the book's Neo4j
    server). Chunks live in a single ``Chunk`` node table; similarity
    search uses Kuzu's ``vector`` (HNSW) and ``fts`` (BM25) extensions,
    which mirror Neo4j's vector index + fulltext index used in ch02.

    ``replace_chunks`` reloads the whole corpus rather than merging into
    it incrementally: Kuzu doesn't allow writing to a column once a
    vector/FTS index has covered it, so each call tears down the indexes
    and table and rebuilds them from scratch.
    """

    _VECTOR_INDEX = "chunk_vector_idx"
    _FTS_INDEX = "chunk_fts_idx"

    def __init__(self, db_path: str | Path, embedding_dim: int) -> None:
        self._db = kuzu.Database(str(db_path))
        self._conn = kuzu.Connection(self._db)
        self._embedding_dim = embedding_dim
        self._conn.execute("INSTALL vector; LOAD EXTENSION vector;")
        self._conn.execute("INSTALL fts; LOAD EXTENSION fts;")
        self._create_table()

    def _create_table(self) -> None:
        self._conn.execute(
            "CREATE NODE TABLE IF NOT EXISTS Chunk("
            f"idx INT64, text STRING, embedding FLOAT[{self._embedding_dim}], "
            "PRIMARY KEY(idx))"
        )

    def replace_chunks(self, chunks: list[str], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        # Once a vector/FTS index has ever covered the embedding/text
        # columns, Kuzu refuses to write to them again -- and dropping the
        # index doesn't lift that (tested against 0.11.3). Only dropping
        # the whole table does, so each call replaces the full corpus:
        # drop indexes, drop table, recreate, insert, rebuild indexes.
        self._drop_indexes()
        self._drop_table()
        self._create_table()
        # Kuzu list indexing is 1-based, hence range(1, size(...)) and i - 1
        # for the 0-based chunk index we expose to callers.
        self._conn.execute(
            "UNWIND range(1, size($chunks)) AS i "
            "CREATE (c:Chunk {idx: i - 1, text: $chunks[i], embedding: $embeddings[i]})",
            {"chunks": chunks, "embeddings": embeddings},
        )
        self._create_indexes()

    def _drop_indexes(self) -> None:
        for drop in (
            f"CALL DROP_VECTOR_INDEX('Chunk', '{self._VECTOR_INDEX}')",
            f"CALL DROP_FTS_INDEX('Chunk', '{self._FTS_INDEX}')",
        ):
            try:
                self._conn.execute(drop)
            except RuntimeError:
                pass  # index doesn't exist yet (first call)

    def _drop_table(self) -> None:
        try:
            self._conn.execute("DROP TABLE Chunk")
        except RuntimeError:
            pass  # table doesn't exist yet (first call)

    def _create_indexes(self) -> None:
        self._conn.execute(
            f"CALL CREATE_VECTOR_INDEX('Chunk', '{self._VECTOR_INDEX}', "
            "'embedding', metric := 'cosine')"
        )
        self._conn.execute(f"CALL CREATE_FTS_INDEX('Chunk', '{self._FTS_INDEX}', ['text'])")

    def vector_search(self, query_embedding: list[float], k: int) -> list[ChunkMatch]:
        result = self._conn.execute(
            f"CALL QUERY_VECTOR_INDEX('Chunk', '{self._VECTOR_INDEX}', $embedding, $k) "
            "RETURN node.idx AS idx, node.text AS text, distance",
            {"embedding": query_embedding, "k": k},
        )
        # Kuzu returns cosine distance (lower is better); convert to a
        # higher-is-better score so it's comparable with keyword_search.
        return [
            ChunkMatch(index=row[0], text=row[1], score=1.0 - row[2]) for row in self._rows(result)
        ]

    def keyword_search(self, query: str, k: int) -> list[ChunkMatch]:
        result = self._conn.execute(
            f"CALL QUERY_FTS_INDEX('Chunk', '{self._FTS_INDEX}', $query) "
            "RETURN node.idx AS idx, node.text AS text, score "
            "ORDER BY score DESC LIMIT $k",
            {"query": query, "k": k},
        )
        return [ChunkMatch(index=row[0], text=row[1], score=row[2]) for row in self._rows(result)]

    @staticmethod
    def _rows(result: Any) -> Iterator[list[Any]]:
        while result.has_next():
            yield result.get_next()

    def close(self) -> None:
        self._conn.close()
        self._db.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
