"""Re-implementation of ch05 (template/kg-rag/notebooks/ch05.ipynb):
agentic RAG over the movies graph, with the ``AgenticRag``
recipe that ``build_movie_agent`` wires with the movie tools. For each
question the agent rewrites it into an atomic question, lets the LLM
route it to a tool (two fixed Cypher lookups, text2cypher as fallback, or
a direct answer), has a critic check the gathered answers for gaps and
fills them with one more round, then answers from the gathered
information only.

The movies graph is shared with ch04 and ch08 and seeded only if it's
missing (or with --reindex). See examples/README.md for the flow and where
it departs from the book. Requires OPENAI_API_KEY in .env (see
.env.example). Run from the repo root:

    .venv/Scripts/python.exe examples/agentic_rag_movies.py \\
        --process-name ch05_agentic_rag_movies
"""

from __future__ import annotations

import argparse

from constants import DEFAULT_ANSWER_MODEL, LOG_DIR, add_reindex_argument
from dotenv import load_dotenv

from graphrag.domains.movies import MovieGraphSeeder, build_movie_agent
from graphrag.graph import KuzuCypherGraphStore
from graphrag.observability import RunLogging, Trace
from graphrag.services import OpenAIServices
from graphrag.settings import Settings

QUESTIONS = [
    "Who's the main actor in the movie Matrix and what other movies is that person in?",
    "How many directors and producers are in the database?",
]

trace = Trace(__name__)


def main(process_name: str, reindex: bool, verbose: bool) -> None:
    with RunLogging(
        process_name,
        LOG_DIR,
        title="ch05 · Agentic RAG over the movies graph",
        details={"model": DEFAULT_ANSWER_MODEL},
        verbose=verbose,
    ) as run:
        load_dotenv()
        settings = Settings.from_env()
        llm = OpenAIServices.create(settings, usage=run.usage).llm

        with KuzuCypherGraphStore(settings.db_path_for("movies")) as store:
            seeder = MovieGraphSeeder(store)
            if reindex or not seeder.is_seeded():
                with trace.section("Seed the movies graph"):
                    seeder.seed()

            rag = build_movie_agent(llm, store, model=DEFAULT_ANSWER_MODEL)
            for index, question in enumerate(QUESTIONS, start=1):
                with trace.section(f"Question {index}/{len(QUESTIONS)}"):
                    trace.field("question", question)
                    rag.query(question)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    RunLogging.add_arguments(parser, default_process_name="ch05_agentic_rag_movies")
    add_reindex_argument(parser)
    args = parser.parse_args()
    main(args.process_name, args.reindex, args.verbose)
