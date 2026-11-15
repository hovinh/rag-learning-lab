"""The ch05 movies agent, wired up once for the scripts that use it: ch05
runs it, ch08 evaluates it (optionally with ch08's question splitting).
"""

from __future__ import annotations

from constants import DEFAULT_ANSWER_MODEL, MOVIES_CYPHER_EXAMPLES, MOVIES_TERMINOLOGY

from graphrag.agent import (
    Agent,
    AnswerCritic,
    AnswerGivenTool,
    MovieInfoByTitleTool,
    MoviesInfoByActorTool,
    QueryDecomposer,
    QueryPlanner,
    QueryUpdater,
    Text2CypherTool,
    ToolRouter,
)
from graphrag.graph import CypherGraphStore, Text2Cypher
from graphrag.llm import LLMClient


def build_movie_agent(llm: LLMClient, store: CypherGraphStore, *, decompose: bool = False) -> Agent:
    """``store`` must already hold the movies graph: its schema is read
    here, for the text2cypher tool and for the critic.

    ``decompose`` splits each question into atomic questions first, as
    ch08's ``get_answer`` does, instead of ch05's one rewritten question.
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
    query_planner: QueryPlanner = (
        QueryDecomposer(llm, model=DEFAULT_ANSWER_MODEL)
        if decompose
        else QueryUpdater(llm, model=DEFAULT_ANSWER_MODEL)
    )
    return Agent(
        llm,
        query_planner,
        ToolRouter(llm, tools, model=DEFAULT_ANSWER_MODEL),
        AnswerCritic(llm, data_description=schema.to_prompt(), model=DEFAULT_ANSWER_MODEL),
        model=DEFAULT_ANSWER_MODEL,
    )
