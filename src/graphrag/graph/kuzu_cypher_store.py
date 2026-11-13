from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import kuzu

from graphrag.graph.cypher_store import CypherGraphStore
from graphrag.graph.errors import GraphQueryError
from graphrag.graph.models import GraphSchema, Property, RelationshipPattern


class KuzuCypherGraphStore(CypherGraphStore):
    """CypherGraphStore backed by embedded Kuzu (replaces the book's
    ``neo4j_driver.execute_query``). The book reads the schema with APOC's
    ``apoc.meta.data()``, which Kuzu has no equivalent of; Kuzu's tables are
    declared up front, so the schema is read from its catalog instead
    (``show_tables``, ``table_info``, ``show_connection``).
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db = kuzu.Database(str(db_path))
        self._conn = kuzu.Connection(self._db)

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        try:
            result: Any = self._conn.execute(cypher, params or {})
        except RuntimeError as exc:
            raise GraphQueryError(f"{exc}\nQuery: {cypher}") from exc
        columns = result.get_column_names()
        rows = []
        while result.has_next():
            rows.append(dict(zip(columns, result.get_next(), strict=True)))
        return rows

    def get_schema(self) -> GraphSchema:
        node_props: dict[str, list[Property]] = {}
        rel_props: dict[str, list[Property]] = {}
        relationships: list[RelationshipPattern] = []
        # Sorted so the prompt built from the schema is deterministic.
        tables = self.query("CALL show_tables() RETURN name, type ORDER BY name")
        for table in tables:
            name = table["name"]
            props = [
                Property(row["name"], row["type"])
                for row in self.query(f"CALL table_info('{name}') RETURN name, type")
            ]
            if table["type"] == "NODE":
                node_props[name] = props
            elif table["type"] == "REL":
                if props:
                    rel_props[name] = props
                relationships.extend(
                    RelationshipPattern(row["src"], name, row["dst"])
                    for row in self.query(
                        f"CALL show_connection('{name}') "
                        "RETURN `source table name` AS src, `destination table name` AS dst"
                    )
                )
        return GraphSchema(node_props, rel_props, relationships)

    def close(self) -> None:
        self._conn.close()
        self._db.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
