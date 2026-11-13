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
import io
import sys
from dataclasses import dataclass, field
from typing import Any

import logzero
from constants import DEFAULT_ANSWER_MODEL, LOG_DIR
from dotenv import load_dotenv
from logzero import logger
from openai import OpenAI

from graphrag.datasets import MovieGraphSeeder
from graphrag.graph import CypherExample, CypherGraphStore, KuzuCypherGraphStore, Text2Cypher
from graphrag.llm import OpenAILLMClient
from graphrag.pipeline import Pipeline
from graphrag.settings import Settings

QUESTION = "Who directed the most movies?"
TERMINOLOGY = """
Persons: When a user asks about a person by trade like actor, writer, director, producer, reviewer, they are referring to a node with the label 'Person'.
Movies: When a user asks about a film or movie, they are referring to a node with the label Movie.
"""
EXAMPLES = [
    CypherExample(
        question="Who are the two people acted in most movies together?",
        cypher=(
            "MATCH (p1:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(p2:Person) "
            "WHERE p1 <> p2 RETURN p1.name, p2.name, COUNT(m) AS movieCount "
            "ORDER BY movieCount DESC LIMIT 1"
        ),
    )
]


@dataclass
class Ch04Context:
    question: str
    cypher: str = ""
    rows: list[dict[str, Any]] = field(default_factory=list)


class GenerateCypherStep:
    def __init__(self, text2cypher: Text2Cypher) -> None:
        self._text2cypher = text2cypher

    def __call__(self, context: Ch04Context) -> None:
        logger.info(f"Question: {context.question}")
        context.cypher = self._text2cypher.generate(context.question)
        logger.info(f"Generated Cypher:\n{context.cypher}")


class ExecuteCypherStep:
    """Runs the generated Cypher -- the book's ch04 notebook only prints
    it, but executing it is what shows the query is valid against the graph.
    """

    def __init__(self, store: CypherGraphStore) -> None:
        self._store = store

    def __call__(self, context: Ch04Context) -> None:
        context.rows = self._store.query(context.cypher)
        logger.info(f"{len(context.rows)} rows")
        for row in context.rows:
            logger.info(row)


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

    # Kept apart from ch02/ch03's databases: seeding drops and recreates
    # its tables on every run.
    db_path = settings.kuzu_db_path.with_name(f"{settings.kuzu_db_path.name}_ch04")
    with KuzuCypherGraphStore(db_path) as store:
        MovieGraphSeeder(store).seed()
        schema = store.get_schema()
        logger.info(f"Schema:\n{schema.to_prompt()}")

        text2cypher = Text2Cypher(
            llm, schema, terminology=TERMINOLOGY, examples=EXAMPLES, model=DEFAULT_ANSWER_MODEL
        )
        pipeline: Pipeline[Ch04Context] = Pipeline(
            [GenerateCypherStep(text2cypher), ExecuteCypherStep(store)]
        )
        pipeline.run(Ch04Context(question=QUESTION))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--process-name",
        default="ch04_text2cypher_movies",
        help="Names the log file written to logs/<process_name>.log.",
    )
    args = parser.parse_args()
    main(args.process_name)
