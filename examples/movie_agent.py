"""The ch05 movies agent, wired up once for the scripts that use it: ch05
runs it, ch08 evaluates it.
"""

from __future__ import annotations

from constants import DEFAULT_ANSWER_MODEL, MOVIES_CYPHER_EXAMPLES, MOVIES_TERMINOLOGY

from graphrag.agent import (
    Agent,
    AnswerCritic,
    AnswerGivenTool,
    MovieInfoByTitleTool,
    MoviesInfoByActorTool,
    QueryUpdater,
    Text2CypherTool,
    ToolRouter,
)
from graphrag.graph import CypherGraphStore, Text2Cypher
from graphrag.llm import LLMClient


def build_movie_agent(llm: LLMClient, store: CypherGraphStore) -> Agent:
    """``store`` must already hold the movies graph: its schema is read
    here, for the text2cypher tool and for the critic.
    """
    schema = store.get_schema()
    text2cypher = Text2Cypher(
        llm,
        schema,
        terminology=MOVIES_TERMINOLOGY,
        examples=MOVIES_CYPHER_EXAMPLES,
        model=DEFAULT_ANSWER_MODEL,
    )
    tools = [
        MovieInfoByTitleTool(store),
        MoviesInfoByActorTool(store),
        Text2CypherTool(text2cypher, store),
        AnswerGivenTool(),
    ]
    return Agent(
        llm,
        QueryUpdater(llm, model=DEFAULT_ANSWER_MODEL),
        ToolRouter(llm, tools, model=DEFAULT_ANSWER_MODEL),
        AnswerCritic(llm, data_description=schema.to_prompt(), model=DEFAULT_ANSWER_MODEL),
        model=DEFAULT_ANSWER_MODEL,
    )
