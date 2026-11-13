"""Re-implementation of ch05 (template/kg-rag/notebooks/ch05.ipynb):
agentic RAG over the movies graph. For each question the agent rewrites
it into an atomic question, lets the LLM route it to a tool (two fixed
Cypher lookups, text2cypher as fallback, or a direct answer), has a critic
check the gathered answers for gaps and fills them with one more round,
then answers from the gathered information only.

See examples/README.md for the flow and where it departs from the book.
Requires OPENAI_API_KEY in .env (see .env.example). Run from the repo
root:

    .venv/Scripts/python.exe examples/agentic_rag_movies.py \\
        --process-name ch05_agentic_rag_movies
"""

from __future__ import annotations

import argparse
import io
import sys

import logzero
from constants import (
    DEFAULT_ANSWER_MODEL,
    LOG_DIR,
    MOVIES_CYPHER_EXAMPLES,
    MOVIES_TERMINOLOGY,
)
from dotenv import load_dotenv
from logzero import logger
from openai import OpenAI

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
from graphrag.datasets import MovieGraphSeeder
from graphrag.graph import KuzuCypherGraphStore, Text2Cypher
from graphrag.llm import OpenAILLMClient
from graphrag.settings import Settings

QUESTIONS = [
    "Who's the main actor in the movie Matrix and what other movies is that person in?",
    "How many directors and producers are in the database?",
]


def main(process_name: str) -> None:
    # Windows defaults stdout/the log file to the cp1252 codepage, which
    # can't encode some names in the movies dataset (e.g. "Miloš Forman");
    # force UTF-8 so those log lines don't get silently dropped.
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logzero.logfile(LOG_DIR / f"{process_name}.log", encoding="utf-8")

    load_dotenv()
    settings = Settings.from_env()
    llm = OpenAILLMClient(OpenAI(api_key=settings.openai_api_key))

    # Kept apart from the other chapters' databases: seeding drops and
    # recreates its tables on every run.
    db_path = settings.kuzu_db_path.with_name(f"{settings.kuzu_db_path.name}_ch05")
    with KuzuCypherGraphStore(db_path) as store:
        MovieGraphSeeder(store).seed()

        text2cypher = Text2Cypher(
            llm,
            store.get_schema(),
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
        agent = Agent(
            llm,
            QueryUpdater(llm, model=DEFAULT_ANSWER_MODEL),
            ToolRouter(llm, tools, model=DEFAULT_ANSWER_MODEL),
            AnswerCritic(llm, model=DEFAULT_ANSWER_MODEL),
            model=DEFAULT_ANSWER_MODEL,
        )

        for question in QUESTIONS:
            logger.info(f"--- Question: {question} ---")
            answer = agent.run(question)
            logger.info(f"Answer: {answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--process-name",
        default="ch05_agentic_rag_movies",
        help="Names the log file written to logs/<process_name>.log.",
    )
    args = parser.parse_args()
    main(args.process_name)
