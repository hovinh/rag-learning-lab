from __future__ import annotations

from graphrag.graph.cypher_store import CypherGraphStore
from graphrag.graph.errors import GraphQueryError

_PROJECTED_GRAPH = "entity_graph"

# Kuzu marks nodes Louvain didn't cluster (no relationships) with -1; they
# get no community rather than a singleton one.
_WRITE_COMMUNITIES = f"""
CALL louvain('{_PROJECTED_GRAPH}')
WITH node, louvain_id
SET node.community_id = CASE WHEN louvain_id < 0 THEN NULL ELSE louvain_id END
"""

_COMMUNITY_SIZES = """
MATCH (e:Entity)
WHERE e.community_id IS NOT NULL
RETURN e.community_id AS id, count(*) AS size
ORDER BY id
"""


class KuzuLouvainCommunityDetector:
    """Clusters the entity graph with Louvain and writes each entity's
    cluster to ``Entity.community_id`` (ch07's ``calculate_communities``).

    The book projects the graph and runs Louvain with Neo4j GDS
    (``gds.graph.project``, ``gds.louvain.write``); this runs Kuzu's
    equivalent ``algo`` procedures (``project_graph``, ``louvain``), sent
    through the backend-agnostic ``CypherGraphStore`` like the importer's
    DDL. Kuzu's Louvain treats the ``RELATIONSHIP`` edges as undirected, as
    the book's ``undirectedRelationshipTypes`` does.
    """

    def __init__(self, store: CypherGraphStore) -> None:
        self._store = store

    def detect(self) -> dict[int, int]:
        """Returns each community's size, keyed by community id."""
        self._drop_projected_graph()
        self._store.query(f"CALL project_graph('{_PROJECTED_GRAPH}', ['Entity'], ['RELATIONSHIP'])")
        try:
            self._store.query(_WRITE_COMMUNITIES)
        finally:
            self._drop_projected_graph()
        return {row["id"]: row["size"] for row in self._store.query(_COMMUNITY_SIZES)}

    def _drop_projected_graph(self) -> None:
        try:
            self._store.query(f"CALL drop_projected_graph('{_PROJECTED_GRAPH}')")
        except GraphQueryError:
            pass  # not projected yet
