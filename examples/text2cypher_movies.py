"""Re-implementation of ch04 (template/kg-rag/notebooks/ch04.ipynb):
seed a small movies graph into Kuzu, read its schema, have the LLM
generate a Cypher query for a question (text2cypher, with a terminology
mapping and a few-shot example), then run that query against the graph.

See examples/README.md for how the prompt is assembled and how the schema
is read from Kuzu. Requires OPENAI_API_KEY in .env (see .env.example). Run
from the repo root:

    .venv/Scripts/python.exe examples/text2cypher_movies.py \\
        --process-name ch04_text2cypher_movies
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any

from constants import (
    DEFAULT_ANSWER_MODEL,
    LOG_DIR,
    MOVIES_CYPHER_EXAMPLES,
    MOVIES_TERMINOLOGY,
)
from dotenv import load_dotenv

from graphrag.datasets import MovieGraphSeeder
from graphrag.graph import KuzuCypherGraphStore, Text2Cypher
from graphrag.observability import RunLogging, Trace
from graphrag.pipeline import ExecuteCypherStep, GenerateCypherStep, Pipeline
from graphrag.services import OpenAIServices
from graphrag.settings import Settings

QUESTION = "Who directed the most movies?"

trace = Trace(__name__)


@dataclass
class Ch04Context:
    question: str
    cypher: str = ""
    rows: list[dict[str, Any]] = field(default_factory=list)


def main(process_name: str, verbose: bool) -> None:
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

        # Kept apart from the other chapters' databases: seeding drops and
        # recreates its tables on every run.
        with KuzuCypherGraphStore(settings.db_path_for("ch04")) as store:
            with trace.section("Seed the movies graph"):
                MovieGraphSeeder(store).seed()
                schema = store.get_schema()
                # The schema is what text2cypher's prompt shows the LLM.
                trace.block("schema", schema.to_prompt())

            text2cypher = Text2Cypher(
                llm,
                schema,
                terminology=MOVIES_TERMINOLOGY,
                examples=MOVIES_CYPHER_EXAMPLES,
                model=DEFAULT_ANSWER_MODEL,
            )
            pipeline: Pipeline[Ch04Context] = Pipeline(
                [GenerateCypherStep(text2cypher), ExecuteCypherStep(store)]
            )
            pipeline.run(Ch04Context(question=QUESTION))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    RunLogging.add_arguments(parser, default_process_name="ch04_text2cypher_movies")
    args = parser.parse_args()
    main(args.process_name, args.verbose)
