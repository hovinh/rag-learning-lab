"""Re-implementation of ch07 (template/kg-rag/notebooks/ch07.ipynb) with
the ``GraphRag`` recipe: Microsoft-style GraphRAG over
Homer's Odyssey. Have the LLM extract entities and relationships from
each chunk with no predefined schema, summarize their descriptions,
cluster the entities into communities (Louvain) and write a report per
community, then answer a broad question with global search (map-reduce
over the reports) and an entity-specific one with local search (vector
search over entities plus their graph neighbourhood).

Building the graph makes roughly 100-150 LLM calls per book, so it's done
once and kept in its Kuzu database: later runs only search. --reindex
rebuilds it, which is also needed after changing --books. See
examples/README.md for the flow and where it departs from the book.
Requires OPENAI_API_KEY in .env (see .env.example). Run from the repo
root:

    .venv/Scripts/python.exe examples/graphrag_odyssey.py \\
        --process-name ch07_graphrag_odyssey
"""

from __future__ import annotations

import argparse
from pathlib import Path

from constants import DEFAULT_ANSWER_MODEL, LOG_DIR, add_reindex_argument
from dotenv import load_dotenv

from graphrag.domains.odyssey import ODYSSEY_ENTITY_TYPES, OdysseyCorpus
from graphrag.graph import KuzuCypherGraphStore
from graphrag.ingestion import TextChunker
from graphrag.llm import TokenCounter
from graphrag.observability import RunLogging, Trace
from graphrag.recipes import GraphRag
from graphrag.services import OpenAIServices
from graphrag.settings import Settings

ODYSSEY_PATH = Path("data/ch07-odyssey.txt")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 40
GLOBAL_QUESTION = "What is this story about?"
LOCAL_QUESTION = "Who is Ulysses?"

trace = Trace(__name__)


def main(process_name: str, number_of_books: int, reindex: bool, verbose: bool) -> None:
    with RunLogging(
        process_name,
        LOG_DIR,
        title="ch07 · GraphRAG over the Odyssey (global and local search)",
        details={"model": DEFAULT_ANSWER_MODEL, "books": number_of_books},
        verbose=verbose,
    ) as run:
        load_dotenv()
        settings = Settings.from_env()
        services = OpenAIServices.create(settings, usage=run.usage)
        db_path = settings.db_path_for("odyssey")

        with KuzuCypherGraphStore(db_path) as store:
            rag = GraphRag(
                store,
                services.llm,
                services.embedder,
                entity_types=ODYSSEY_ENTITY_TYPES,
                chunker=TextChunker(CHUNK_SIZE, CHUNK_OVERLAP),
                model=DEFAULT_ANSWER_MODEL,
            )
            if reindex or not rag.is_indexed():
                with trace.section("Build the knowledge graph"):
                    rag.index(
                        OdysseyCorpus(ODYSSEY_PATH),
                        limit=number_of_books,
                        token_counter=TokenCounter(),
                    )
            else:
                trace.field("graph", f"reusing {db_path} (--reindex rebuilds it)")

            with trace.section("Global search"):
                rag.global_search(GLOBAL_QUESTION)
            with trace.section("Local search"):
                rag.local_search(LOCAL_QUESTION)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    RunLogging.add_arguments(parser, default_process_name="ch07_graphrag_odyssey")
    parser.add_argument(
        "--books",
        type=int,
        default=1,
        help=(
            "How many of the Odyssey's 24 books to build the graph from (the book uses 1). "
            "Applies when the graph is built: pass --reindex to rebuild an existing one."
        ),
    )
    add_reindex_argument(parser)
    args = parser.parse_args()
    main(args.process_name, args.books, args.reindex, args.verbose)
