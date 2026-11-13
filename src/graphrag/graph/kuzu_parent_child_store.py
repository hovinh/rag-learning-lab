from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Self

import kuzu

from graphrag.graph.models import ParentMatch
from graphrag.graph.parent_child_store import ParentChildGraphStore


class KuzuParentChildGraphStore(ParentChildGraphStore):
    """ParentChildGraphStore backed by embedded Kuzu (replaces the book's
    Neo4j server). Parents live in a ``Parent`` node table, children (with
    embeddings) in a ``Child`` node table, linked by a ``HAS_CHILD`` rel
    table. Similarity search uses Kuzu's ``vector`` (HNSW) extension over
    children, mirroring Neo4j's vector index used in ch03; results are
    aggregated up to the parent.

    ``replace_parents`` reloads the whole corpus rather than merging into
    it incrementally, for the same reason as ``KuzuGraphStore.replace_chunks``:
    Kuzu doesn't allow writing to a column once a vector index has covered
    it, so each call tears down the index and tables and rebuilds them from
    scratch.
    """

    _VECTOR_INDEX = "child_vector_idx"

    def __init__(self, db_path: str | Path, embedding_dim: int) -> None:
        self._db = kuzu.Database(str(db_path))
        self._conn = kuzu.Connection(self._db)
        self._embedding_dim = embedding_dim
        self._conn.execute("INSTALL vector; LOAD EXTENSION vector;")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute(
            "CREATE NODE TABLE IF NOT EXISTS Parent(idx INT64, text STRING, PRIMARY KEY(idx))"
        )
        self._conn.execute(
            "CREATE NODE TABLE IF NOT EXISTS Child("
            f"idx INT64, text STRING, embedding FLOAT[{self._embedding_dim}], "
            "PRIMARY KEY(idx))"
        )
        self._conn.execute("CREATE REL TABLE IF NOT EXISTS HAS_CHILD(FROM Parent TO Child)")

    def replace_parents(
        self,
        parents: list[str],
        children_per_parent: list[list[str]],
        child_embeddings_per_parent: list[list[list[float]]],
    ) -> None:
        if len(parents) != len(children_per_parent) or len(parents) != len(
            child_embeddings_per_parent
        ):
            raise ValueError(
                "parents, children_per_parent, and child_embeddings_per_parent "
                "must be the same length"
            )
        for children, embeddings in zip(children_per_parent, child_embeddings_per_parent):
            if len(children) != len(embeddings):
                raise ValueError("each parent's children and embeddings must be the same length")

        self._drop_index()
        self._drop_tables()
        self._create_tables()

        child_idx = 0
        for parent_idx, (parent_text, children, embeddings) in enumerate(
            zip(parents, children_per_parent, child_embeddings_per_parent)
        ):
            child_idxs = list(range(child_idx, child_idx + len(children)))
            child_idx += len(children)
            self._conn.execute(
                "CREATE (p:Parent {idx: $pidx, text: $ptext}) "
                "WITH p "
                "UNWIND range(1, size($children)) AS j "
                "CREATE (c:Child {idx: $child_idxs[j], text: $children[j], "
                "embedding: $embeddings[j]}) "
                "CREATE (p)-[:HAS_CHILD]->(c)",
                {
                    "pidx": parent_idx,
                    "ptext": parent_text,
                    "children": children,
                    "embeddings": embeddings,
                    "child_idxs": child_idxs,
                },
            )

        self._create_index()

    def _drop_index(self) -> None:
        try:
            self._conn.execute(f"CALL DROP_VECTOR_INDEX('Child', '{self._VECTOR_INDEX}')")
        except RuntimeError:
            pass  # index doesn't exist yet (first call)

    def _drop_tables(self) -> None:
        for drop in ("DROP TABLE HAS_CHILD", "DROP TABLE Child", "DROP TABLE Parent"):
            try:
                self._conn.execute(drop)
            except RuntimeError:
                pass  # table doesn't exist yet (first call)

    def _create_index(self) -> None:
        self._conn.execute(
            f"CALL CREATE_VECTOR_INDEX('Child', '{self._VECTOR_INDEX}', "
            "'embedding', metric := 'cosine')"
        )

    def parent_retrieval(self, query_embedding: list[float], k: int) -> list[ParentMatch]:
        result = self._conn.execute(
            f"CALL QUERY_VECTOR_INDEX('Child', '{self._VECTOR_INDEX}', $embedding, $k4) "
            "MATCH (parent:Parent)-[:HAS_CHILD]->(node) "
            "WITH parent, min(distance) AS distance "
            "RETURN parent.idx AS idx, parent.text AS text, distance "
            "ORDER BY distance ASC LIMIT $k",
            {"embedding": query_embedding, "k4": k * 4, "k": k},
        )
        # Kuzu returns cosine distance (lower is better); convert to a
        # higher-is-better score, consistent with ChunkMatch/GraphStore.
        return [
            ParentMatch(index=row[0], text=row[1], score=1.0 - row[2]) for row in self._rows(result)
        ]

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
