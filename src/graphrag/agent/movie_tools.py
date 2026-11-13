from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from graphrag.agent.tools import Tool, ToolResult, string_parameters
from graphrag.graph.cypher_store import CypherGraphStore

# Shared by both tools. Differs from the book's queries in three ways:
# - properties are returned instead of the whole node, so results serialize
#   into the conversation without backend-specific node objects;
# - DISTINCT, because two OPTIONAL MATCHes multiply rows, so the book's
#   plain collect() repeats every name once per row of the other match;
# - `cast` is backtick-escaped, as it's a reserved word in Kuzu.
_RETURN_MOVIE_INFO = """
OPTIONAL MATCH (m)<-[:ACTED_IN]-(c:Person)
OPTIONAL MATCH (m)<-[:DIRECTED]-(d:Person)
RETURN m.title AS title, m.released AS released, m.tagline AS tagline,
       collect(DISTINCT c.name) AS `cast`, collect(DISTINCT d.name) AS directors
ORDER BY released
"""


class MovieInfoByTitleTool(Tool):
    """Movies whose title contains the given text, with cast and
    directors.
    """

    name = "movie_info_by_title"
    description = "Get information about a movie by providing the title"
    parameters = string_parameters("title", "The movie title")

    _QUERY = "MATCH (m:Movie) WHERE lower(m.title) CONTAINS $title" + _RETURN_MOVIE_INFO

    def __init__(self, store: CypherGraphStore) -> None:
        self._store = store

    def run(self, arguments: Mapping[str, Any]) -> ToolResult:
        return self._store.query(self._QUERY, {"title": str(arguments["title"]).lower()})


class MoviesInfoByActorTool(Tool):
    """Movies an actor (matched by name substring) acted in, with each
    movie's full cast and directors.

    The book's query never actually filters by actor: its second ``MATCH``
    rebinds ``a`` and its ``WHERE`` attaches to an ``OPTIONAL MATCH``. Here
    the filter applies to the actor, and the cast gets its own variable.
    """

    name = "movies_info_by_actor"
    description = "Get information about a movie by providing an actor"
    parameters = string_parameters("actor", "The actor name")

    _QUERY = (
        "MATCH (a:Person)-[:ACTED_IN]->(m:Movie) WHERE lower(a.name) CONTAINS $actor"
        + _RETURN_MOVIE_INFO
    )

    def __init__(self, store: CypherGraphStore) -> None:
        self._store = store

    def run(self, arguments: Mapping[str, Any]) -> ToolResult:
        return self._store.query(self._QUERY, {"actor": str(arguments["actor"]).lower()})
