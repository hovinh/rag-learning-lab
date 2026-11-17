import pytest

from graphrag.domains.movies.seeder import MovieGraphSeeder
from graphrag.graph.kuzu_cypher_store import KuzuCypherGraphStore


@pytest.fixture
def store(tmp_path):
    with KuzuCypherGraphStore(tmp_path / "test_kuzu_db") as store:
        yield store


def count(store, pattern):
    [row] = store.query(f"MATCH {pattern} RETURN count(*) AS n")
    return row["n"]


def test_seed_loads_every_table(store):
    MovieGraphSeeder(store).seed()

    assert count(store, "(:Movie)") == 14
    assert count(store, "(:Person)") == 48
    assert count(store, "(:Person)-[:ACTED_IN]->(:Movie)") == 51
    assert count(store, "(:Person)-[:DIRECTED]->(:Movie)") == 16
    assert count(store, "(:Person)-[:PRODUCED]->(:Movie)") == 7
    assert count(store, "(:Person)-[:WROTE]->(:Movie)") == 9


def test_seed_is_safe_to_rerun(store):
    MovieGraphSeeder(store).seed()
    MovieGraphSeeder(store).seed()

    assert count(store, "(:Movie)") == 14


def test_seed_parses_lists_and_empty_fields(store):
    MovieGraphSeeder(store).seed()

    [row] = store.query(
        "MATCH (:Person {name: 'Tom Hanks'})-[r:ACTED_IN]->(m:Movie {title: 'Splash'}) "
        "RETURN r.roles AS roles, m.tagline AS tagline"
    )
    assert row == {"roles": ["Allen Bauer"], "tagline": None}


def test_seed_builds_schema_from_declared_tables(store):
    MovieGraphSeeder(store).seed()

    schema = store.get_schema()

    assert set(schema.node_props) == set(MovieGraphSeeder.NODE_TABLES)
    assert {r.type for r in schema.relationships} == set(MovieGraphSeeder.REL_TABLES)
