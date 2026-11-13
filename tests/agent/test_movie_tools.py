import pytest

from graphrag.agent.movie_tools import MovieInfoByTitleTool, MoviesInfoByActorTool
from graphrag.datasets.movie_graph import MovieGraphSeeder
from graphrag.graph.kuzu_cypher_store import KuzuCypherGraphStore

MATRIX_CAST = {"Keanu Reeves", "Carrie-Anne Moss", "Laurence Fishburne", "Hugo Weaving"}
WACHOWSKIS = {"Lana Wachowski", "Lilly Wachowski"}


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    with KuzuCypherGraphStore(tmp_path_factory.mktemp("movies") / "test_kuzu_db") as store:
        MovieGraphSeeder(store).seed()
        yield store


def test_movie_info_by_title_matches_case_insensitive_substring(store):
    rows = MovieInfoByTitleTool(store).run({"title": "MATRIX"})

    assert [row["title"] for row in rows] == ["The Matrix", "The Matrix Reloaded"]
    [matrix, _reloaded] = rows
    assert matrix["released"] == 1999
    assert matrix["tagline"] == "Welcome to the Real World"
    # Each name once, despite the cast x directors row product.
    assert sorted(matrix["cast"]) == sorted(MATRIX_CAST)
    assert sorted(matrix["directors"]) == sorted(WACHOWSKIS)


def test_movie_info_by_title_returns_nothing_for_unknown_title(store):
    assert MovieInfoByTitleTool(store).run({"title": "no such movie"}) == []


def test_movies_info_by_actor_returns_only_that_actors_movies(store):
    rows = MoviesInfoByActorTool(store).run({"actor": "keanu reeves"})

    assert [row["title"] for row in rows] == ["The Matrix", "The Matrix Reloaded"]
    for row in rows:
        assert set(row["cast"]) == MATRIX_CAST
        assert set(row["directors"]) == WACHOWSKIS


def test_movies_info_by_actor_includes_full_cast(store):
    [splash] = [
        row
        for row in MoviesInfoByActorTool(store).run({"actor": "Tom Hanks"})
        if row["title"] == "Splash"
    ]

    assert "Tom Hanks" in splash["cast"]
    assert len(splash["cast"]) > 1
