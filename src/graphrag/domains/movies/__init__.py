"""The movies domain (ch04, ch05, ch08): a small movies graph, its
text2cypher prompt sections, the agent's tools and scope, and the
benchmark that evaluates the agent.
"""

from graphrag.domains.movies.agent import MOVIES_AGENT_SCOPE, build_movie_agent, movie_tools
from graphrag.domains.movies.benchmark import MOVIES_BENCHMARK_PATH
from graphrag.domains.movies.seeder import MovieGraphSeeder
from graphrag.domains.movies.text2cypher import MOVIES_CYPHER_EXAMPLES, MOVIES_TERMINOLOGY
from graphrag.domains.movies.tools import MovieInfoByTitleTool, MoviesInfoByActorTool

__all__ = [
    "MOVIES_AGENT_SCOPE",
    "MOVIES_BENCHMARK_PATH",
    "MOVIES_CYPHER_EXAMPLES",
    "MOVIES_TERMINOLOGY",
    "MovieGraphSeeder",
    "MovieInfoByTitleTool",
    "MoviesInfoByActorTool",
    "build_movie_agent",
    "movie_tools",
]
