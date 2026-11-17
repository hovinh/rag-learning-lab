from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from graphrag.graph.cypher_store import CypherGraphStore

_DATA_DIR = Path(__file__).parent / "data"


class MovieGraphSeeder:
    """Loads a small movies graph modeled on the Neo4j "Movies" demo
    dataset that the book's ch04/ch05/ch08 notebooks assume is already in
    the database. Kuzu ships no such sample and needs every table declared
    before data goes in, so this creates the tables and bulk-loads one CSV
    per table (``<table name lowercased>.csv`` under ``data_dir``).

    The DDL and ``COPY ... FROM`` statements are Kuzu's dialect, although
    they're sent through the backend-agnostic ``CypherGraphStore``.
    """

    NODE_TABLES: ClassVar[dict[str, str]] = {
        "Movie": "title STRING, released INT64, tagline STRING, votes INT64, PRIMARY KEY(title)",
        "Person": "name STRING, born INT64, PRIMARY KEY(name)",
    }
    REL_TABLES: ClassVar[dict[str, str]] = {
        "ACTED_IN": "FROM Person TO Movie, roles STRING[]",
        "DIRECTED": "FROM Person TO Movie",
        "PRODUCED": "FROM Person TO Movie",
        "WROTE": "FROM Person TO Movie",
    }

    def __init__(self, store: CypherGraphStore, data_dir: Path = _DATA_DIR) -> None:
        self._store = store
        self._data_dir = data_dir

    def seed(self) -> None:
        """Drop and recreate every table, then load the CSVs -- safe to
        re-run against an already-seeded database.
        """
        # Rel tables reference node tables, so they're dropped first and
        # created last.
        for table in [*self.REL_TABLES, *self.NODE_TABLES]:
            self._store.query(f"DROP TABLE IF EXISTS {table}")
        for table, columns in self.NODE_TABLES.items():
            self._store.query(f"CREATE NODE TABLE {table}({columns})")
        for table, columns in self.REL_TABLES.items():
            self._store.query(f"CREATE REL TABLE {table}({columns})")

        for table in [*self.NODE_TABLES, *self.REL_TABLES]:
            csv_path = (self._data_dir / f"{table.lower()}.csv").as_posix()
            self._store.query(f"COPY {table} FROM '{csv_path}' (header=true)")
