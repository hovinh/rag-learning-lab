import pytest

from graphrag.graph.errors import GraphQueryError
from graphrag.graph.kuzu_cypher_store import KuzuCypherGraphStore
from graphrag.graph.models import Property, RelationshipPattern


@pytest.fixture
def store(tmp_path):
    with KuzuCypherGraphStore(tmp_path / "test_kuzu_db") as store:
        store.query("CREATE NODE TABLE Person(name STRING, born INT64, PRIMARY KEY(name))")
        store.query("CREATE NODE TABLE Movie(title STRING, PRIMARY KEY(title))")
        store.query("CREATE REL TABLE ACTED_IN(FROM Person TO Movie, roles STRING[])")
        store.query("CREATE REL TABLE DIRECTED(FROM Person TO Movie)")
        store.query("CREATE (:Person {name: 'Tom Hanks', born: 1956})")
        store.query("CREATE (:Movie {title: 'Splash'})")
        store.query(
            "MATCH (p:Person {name: 'Tom Hanks'}), (m:Movie {title: 'Splash'}) "
            "CREATE (p)-[:ACTED_IN {roles: ['Allen Bauer']}]->(m)"
        )
        yield store


def test_query_returns_rows_keyed_by_column_name(store):
    rows = store.query(
        "MATCH (p:Person)-[r:ACTED_IN]->(m:Movie) RETURN p.name AS name, r.roles AS roles, m.title"
    )

    assert rows == [{"name": "Tom Hanks", "roles": ["Allen Bauer"], "m.title": "Splash"}]


def test_query_binds_params(store):
    rows = store.query(
        "MATCH (p:Person) WHERE p.born = $born RETURN p.name AS name", {"born": 1956}
    )

    assert rows == [{"name": "Tom Hanks"}]


def test_string_length_is_size_not_length(store):
    # text2cypher's Kuzu hints rely on this: Neo4j's length(string) is
    # rejected, size() works.
    [row] = store.query("MATCH (p:Person) RETURN size(p.name) AS n")
    assert row == {"n": 9}
    with pytest.raises(GraphQueryError):
        store.query("MATCH (p:Person) RETURN length(p.name) AS n")


def test_query_wraps_database_errors(store):
    with pytest.raises(GraphQueryError, match="MATCH \\(n:Nope\\)"):
        store.query("MATCH (n:Nope) RETURN n")


def test_get_schema_reads_tables_properties_and_connections(store):
    schema = store.get_schema()

    assert schema.node_props == {
        "Movie": [Property("title", "STRING")],
        "Person": [Property("name", "STRING"), Property("born", "INT64")],
    }
    # Rel tables without properties (DIRECTED) are left out, as in the book.
    assert schema.rel_props == {"ACTED_IN": [Property("roles", "STRING[]")]}
    assert schema.relationships == [
        RelationshipPattern("Person", "ACTED_IN", "Movie"),
        RelationshipPattern("Person", "DIRECTED", "Movie"),
    ]
