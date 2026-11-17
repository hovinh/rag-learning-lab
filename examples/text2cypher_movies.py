"""Re-implementation of ch04 (template/kg-rag/notebooks/ch04.ipynb) with
the predefined ``Text2CypherQA`` pipeline: seed a small movies graph into
Kuzu, read its schema, have the LLM generate a Cypher query for a
question (text2cypher, with a terminology mapping and a few-shot example),
then run that query against the graph.

The movies graph is shared with ch05 and ch08 and seeded only if it's
missing (or with --reindex). See examples/README.md for how the prompt is
assembled and how the schema is read from Kuzu. Requires OPENAI_API_KEY
in .env (see .env.example). Run from the repo root:

    .venv/Scripts/python.exe examples/text2cypher_movies.py \\
        --process-name ch04_text2cypher_movies
"""

from __future__ import annotations

import argparse

from constants import DEFAULT_ANSWER_MODEL, LOG_DIR, add_reindex_argument
from dotenv import load_dotenv

from graphrag.domains.movies import (
    MOVIES_CYPHER_EXAMPLES,
    MOVIES_TERMINOLOGY,
    MovieGraphSeeder,
)
from graphrag.graph import KuzuCypherGraphStore
from graphrag.observability import RunLogging, Trace
from graphrag.pipelines import Text2CypherQA
from graphrag.services import OpenAIServices
from graphrag.settings import Settings

QUESTION = "Who directed the most movies?"

trace = Trace(__name__)


def main(process_name: str, reindex: bool, verbose: bool) -> None:
    with RunLogging(
        process_name,
        LOG_DIR,
        title="ch04 · text2cypher over the movies graph",
        details={"model": DEFAULT_ANSWER_MODEL, "question": QUESTION},
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

            qa = Text2CypherQA(
                store,
                llm,
                terminology=MOVIES_TERMINOLOGY,
                examples=MOVIES_CYPHER_EXAMPLES,
                model=DEFAULT_ANSWER_MODEL,
            )
            # The schema is what text2cypher's prompt shows the LLM.
            trace.block("schema", qa.schema.to_prompt())
            qa.query(QUESTION)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    RunLogging.add_arguments(parser, default_process_name="ch04_text2cypher_movies")
    add_reindex_argument(parser)
    args = parser.parse_args()
    main(args.process_name, args.reindex, args.verbose)
