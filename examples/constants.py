"""Constants shared by more than one example script. Per-chapter constants
(sample questions, chunk sizes, PDF/DB paths) stay in their own script.
"""

import argparse
from pathlib import Path

from graphrag.graph import CypherExample

PDF_URL = "https://arxiv.org/pdf/1709.00666.pdf"
LOG_DIR = Path("logs")
DEFAULT_ANSWER_MODEL = "gpt-4o"


def add_verbose_argument(parser: argparse.ArgumentParser) -> None:
    """The ``--verbose`` flag every example takes (see ``RunLogging``)."""
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Also show the full payloads (DEBUG lines) on the console; "
            "logs/<process_name>.log always has them."
        ),
    )


# text2cypher prompt sections for the movies graph (ch04, reused by ch05's
# text2cypher tool).
MOVIES_TERMINOLOGY = """
Persons: When a user asks about a person by trade like actor, writer, director, producer, reviewer, they are referring to a node with the label 'Person'.
Movies: When a user asks about a film or movie, they are referring to a node with the label Movie.
"""
MOVIES_CYPHER_EXAMPLES = [
    CypherExample(
        question="Who are the two people acted in most movies together?",
        cypher=(
            "MATCH (p1:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(p2:Person) "
            "WHERE p1 <> p2 RETURN p1.name, p2.name, COUNT(m) AS movieCount "
            "ORDER BY movieCount DESC LIMIT 1"
        ),
    ),
    # Not in the book: shows the "missing information" reply the
    # instructions ask for, so the LLM doesn't pass off another property
    # (e.g. votes) as the data asked about. Deliberately not a benchmark
    # question.
    CypherExample(
        question="Which movie had the biggest budget?",
        cypher='RETURN "The database has no information about budgets" AS missing_information',
    ),
]
