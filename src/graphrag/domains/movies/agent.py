from __future__ import annotations

from graphrag.agent import AgentScope, AnswerGivenTool, Text2CypherTool, Tool
from graphrag.domains.movies.text2cypher import MOVIES_CYPHER_EXAMPLES, MOVIES_TERMINOLOGY
from graphrag.domains.movies.tools import MovieInfoByTitleTool, MoviesInfoByActorTool
from graphrag.graph import CypherGraphStore, Text2Cypher
from graphrag.llm import LLMClient, OpenAILLMClient
from graphrag.pipelines import AgenticRag

# The book's wording, which its prompts hard-code.
MOVIES_AGENT_SCOPE = AgentScope(
    source="a database of movies and the people who worked on them",
    topic="movies",
    capabilities="questions about movies and their cast",
)


def movie_tools(
    llm: LLMClient, store: CypherGraphStore, *, model: str = OpenAILLMClient.DEFAULT_MODEL
) -> list[Tool]:
    """The ch05 agent's tools: two fixed Cypher lookups, text2cypher as
    the fallback, and a direct answer. ``store`` must already hold the
    movies graph, since text2cypher reads its schema.
    """
    text2cypher = Text2Cypher(
        llm,
        store.get_schema(),
        terminology=MOVIES_TERMINOLOGY,
        examples=MOVIES_CYPHER_EXAMPLES,
        model=model,
    )
    return [
        MovieInfoByTitleTool(store),
        MoviesInfoByActorTool(store),
        Text2CypherTool(text2cypher, store),
        AnswerGivenTool(MOVIES_AGENT_SCOPE),
    ]


def build_movie_agent(
    llm: LLMClient,
    store: CypherGraphStore,
    *,
    decompose: bool = False,
    model: str = OpenAILLMClient.DEFAULT_MODEL,
) -> AgenticRag:
    """The ch05 agent over the movies graph in ``store``.

    ``decompose`` splits each question into atomic questions first, as
    ch08's ``get_answer`` does, instead of ch05's one rewritten question.
    """
    return AgenticRag(
        llm,
        movie_tools(llm, store, model=model),
        scope=MOVIES_AGENT_SCOPE,
        data_description=store.get_schema().to_prompt(),
        decompose=decompose,
        model=model,
    )
