"""Re-implementation of ch07 (template/kg-rag/notebooks/ch07.ipynb):
Microsoft-style GraphRAG over Homer's Odyssey. Have the LLM extract
entities and relationships from each chunk with no predefined schema,
summarize their descriptions, cluster the entities into communities
(Louvain) and write a report per community, then answer a broad question
with global search (map-reduce over the reports) and an entity-specific
one with local search (vector search over entities plus their graph
neighbourhood).

See examples/README.md for the flow and where it departs from the book.
Requires OPENAI_API_KEY in .env (see .env.example). Makes roughly 100-150
LLM calls per book processed. Run from the repo root:

    .venv/Scripts/python.exe examples/graphrag_odyssey.py \\
        --process-name ch07_graphrag_odyssey
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from constants import DEFAULT_ANSWER_MODEL, LOG_DIR
from dotenv import load_dotenv

from graphrag.datasets import OdysseyCorpus
from graphrag.extraction import (
    CommunityReporter,
    DescriptionSummarizer,
    EntityGraphImporter,
    EntityRelationshipExtractor,
    KuzuLouvainCommunityDetector,
)
from graphrag.graph import KuzuCypherGraphStore
from graphrag.ingestion import Chunk, TextChunker
from graphrag.llm import TokenCounter
from graphrag.observability import RunLogging
from graphrag.pipeline import (
    ChunkStep,
    DetectCommunitiesStep,
    ExtractEntityGraphStep,
    GlobalSearchStep,
    IndexEntitiesStep,
    LoadDocumentsStep,
    LocalSearchStep,
    Pipeline,
    ReportCommunitiesStep,
    SummarizeDescriptionsStep,
)
from graphrag.retrieval import (
    EntityEmbeddingIndexer,
    GlobalSearch,
    KeyPoint,
    LocalContext,
    LocalSearch,
)
from graphrag.services import OpenAIServices
from graphrag.settings import Settings

ODYSSEY_PATH = Path("data/ch07-odyssey.txt")
ENTITY_TYPES = [
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "GOD",
    "EVENT",
    "CREATURE",
    "WEAPON_OR_TOOL",
]
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 40
GLOBAL_QUESTION = "What is this story about?"
LOCAL_QUESTION = "Who is Ulysses?"


@dataclass
class Ch07IndexContext:
    documents: list[str] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)


@dataclass
class Ch07GlobalContext:
    question: str
    key_points: list[KeyPoint] = field(default_factory=list)
    answer: str = ""


@dataclass
class Ch07LocalContext:
    question: str
    local_context: LocalContext = field(default_factory=LocalContext)
    answer: str = ""


def main(process_name: str, number_of_books: int, verbose: bool) -> None:
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
        llm, embedder = services.llm, services.embedder

        # Kept apart from the other chapters' databases, and reset on every run:
        # re-importing would append duplicate descriptions, and Kuzu won't
        # rewrite the embedding column once the vector index covers it.
        with KuzuCypherGraphStore(settings.db_path_for("ch07")) as store:
            importer = EntityGraphImporter(store, embedder.dimensions)
            importer.create_schema(reset=True)

            indexing: Pipeline[Ch07IndexContext] = Pipeline(
                [
                    LoadDocumentsStep(
                        OdysseyCorpus(ODYSSEY_PATH),
                        limit=number_of_books,
                        token_counter=TokenCounter(),
                    ),
                    ChunkStep(TextChunker(CHUNK_SIZE, CHUNK_OVERLAP)),
                    ExtractEntityGraphStep(
                        EntityRelationshipExtractor(llm, ENTITY_TYPES, model=DEFAULT_ANSWER_MODEL),
                        importer,
                    ),
                    SummarizeDescriptionsStep(
                        DescriptionSummarizer(store, llm, model=DEFAULT_ANSWER_MODEL)
                    ),
                    DetectCommunitiesStep(KuzuLouvainCommunityDetector(store)),
                    ReportCommunitiesStep(
                        CommunityReporter(store, llm, model=DEFAULT_ANSWER_MODEL)
                    ),
                    IndexEntitiesStep(EntityEmbeddingIndexer(store, embedder)),
                ]
            )
            indexing.run(Ch07IndexContext())

            global_search: Pipeline[Ch07GlobalContext] = Pipeline(
                [GlobalSearchStep(GlobalSearch(store, llm, model=DEFAULT_ANSWER_MODEL))]
            )
            global_search.run(Ch07GlobalContext(GLOBAL_QUESTION))

            local_search: Pipeline[Ch07LocalContext] = Pipeline(
                [LocalSearchStep(LocalSearch(store, llm, embedder, model=DEFAULT_ANSWER_MODEL))]
            )
            local_search.run(Ch07LocalContext(LOCAL_QUESTION))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    RunLogging.add_arguments(parser, default_process_name="ch07_graphrag_odyssey")
    parser.add_argument(
        "--books",
        type=int,
        default=1,
        help="How many of the Odyssey's 24 books to build the graph from (the book uses 1).",
    )
    args = parser.parse_args()
    main(args.process_name, args.books, args.verbose)
