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

from constants import DEFAULT_ANSWER_MODEL, LOG_DIR, add_verbose_argument
from dotenv import load_dotenv
from movie_agent import build_movie_agent
from openai import OpenAI

from graphrag.datasets import MovieGraphSeeder
from graphrag.graph import KuzuCypherGraphStore
from graphrag.llm import OpenAILLMClient
from graphrag.observability import RunLogging, Trace
from graphrag.settings import Settings

QUESTIONS = [
    "Who's the main actor in the movie Matrix and what other movies is that person in?",
    "How many directors and producers are in the database?",
]

trace = Trace(__name__)


def main(process_name: str, verbose: bool) -> None:
    with RunLogging(
        process_name,
        LOG_DIR,
        title="ch05 · Agentic RAG over the movies graph",
        details={"model": DEFAULT_ANSWER_MODEL},
        verbose=verbose,
    ) as run:
        load_dotenv()
        settings = Settings.from_env()
        llm = OpenAILLMClient(OpenAI(api_key=settings.openai_api_key), usage=run.usage)

        # Kept apart from the other chapters' databases: seeding drops and
        # recreates its tables on every run.
        db_path = settings.kuzu_db_path.with_name(f"{settings.kuzu_db_path.name}_ch05")
        with KuzuCypherGraphStore(db_path) as store:
            with trace.section("Seed the movies graph"):
                MovieGraphSeeder(store).seed()

            agent = build_movie_agent(llm, store)

            for index, question in enumerate(QUESTIONS, start=1):
                with trace.section(f"Question {index}/{len(QUESTIONS)}"):
                    trace.field("question", question)
                    # The agent traces its own steps, ending with the answer.
                    agent.run(question)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--process-name",
        default="ch05_agentic_rag_movies",
        help="Names the log file written to logs/<process_name>.log.",
    )
    add_verbose_argument(parser)
    args = parser.parse_args()
    main(args.process_name, args.verbose)
