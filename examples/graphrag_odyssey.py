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

import logzero
from constants import DEFAULT_ANSWER_MODEL, LOG_DIR
from dotenv import load_dotenv
from logzero import logger
from openai import OpenAI

from graphrag.datasets import OdysseyCorpus
from graphrag.extraction import (
    CommunityReporter,
    DescriptionSummarizer,
    EntityGraphImporter,
    EntityRelationshipExtractor,
    KuzuLouvainCommunityDetector,
)
from graphrag.graph import CypherGraphStore, KuzuCypherGraphStore
from graphrag.ingestion import TextChunker
from graphrag.llm import OpenAILLMClient
from graphrag.pipeline import Pipeline
from graphrag.retrieval import EntityEmbeddingIndexer, GlobalSearch, LocalSearch, OpenAIEmbedder
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
class Ch07Context:
    global_question: str
    local_question: str
    books: list[str] = field(default_factory=list)
    chunked_books: list[list[str]] = field(default_factory=list)
    global_answer: str = ""
    local_answer: str = ""


class LoadBooksStep:
    def __init__(self, corpus: OdysseyCorpus, number_of_books: int) -> None:
        self._corpus = corpus
        self._number_of_books = number_of_books

    def __call__(self, context: Ch07Context) -> None:
        books = self._corpus.load_books()
        context.books = books[: self._number_of_books]
        logger.info(f"Loaded {len(books)} books; processing {len(context.books)}")


class ChunkBooksStep:
    def __init__(self, chunker: TextChunker) -> None:
        self._chunker = chunker

    def __call__(self, context: Ch07Context) -> None:
        context.chunked_books = [self._chunker.split(book) for book in context.books]
        logger.info(f"Chunks per book: {[len(chunks) for chunks in context.chunked_books]}")


class ExtractGraphStep:
    def __init__(
        self,
        extractor: EntityRelationshipExtractor,
        importer: EntityGraphImporter,
        store: CypherGraphStore,
    ) -> None:
        self._extractor = extractor
        self._importer = importer
        self._store = store

    def __call__(self, context: Ch07Context) -> None:
        for book_id, chunks in enumerate(context.chunked_books):
            for chunk_index, chunk in enumerate(chunks):
                extraction = self._extractor.extract(chunk)
                self._importer.import_chunk(book_id, chunk_index, chunk, extraction)
                logger.info(
                    f"Book {book_id} chunk {chunk_index + 1}/{len(chunks)}: "
                    f"{len(extraction.entities)} entities, "
                    f"{len(extraction.relationships)} relationships"
                )
        [entities] = self._store.query("MATCH (e:Entity) RETURN count(*) AS n")
        [relationships] = self._store.query("MATCH ()-[r:RELATIONSHIP]->() RETURN count(*) AS n")
        logger.info(f"Graph: {entities['n']} entities, {relationships['n']} relationships")


class SummarizeDescriptionsStep:
    def __init__(self, summarizer: DescriptionSummarizer) -> None:
        self._summarizer = summarizer

    def __call__(self, context: Ch07Context) -> None:
        entities = self._summarizer.summarize_entities()
        pairs = self._summarizer.summarize_relationships()
        logger.info(f"LLM-summarized {entities} entities and {pairs} relationship pairs")


class DetectCommunitiesStep:
    def __init__(self, detector: KuzuLouvainCommunityDetector) -> None:
        self._detector = detector

    def __call__(self, context: Ch07Context) -> None:
        sizes = sorted(self._detector.detect().values())
        if not sizes:
            raise ValueError("Louvain found no communities; the graph has no relationships")
        logger.info(
            f"{len(sizes)} communities: min {sizes[0]}, max {sizes[-1]}, "
            f"mean {sum(sizes) / len(sizes):.1f}"
        )


class ReportCommunitiesStep:
    def __init__(self, reporter: CommunityReporter) -> None:
        self._reporter = reporter

    def __call__(self, context: Ch07Context) -> None:
        for report in self._reporter.report_communities():
            logger.info(f"Community {report.community_id} ({report.rating:g}): {report.title}")


class IndexEntitiesStep:
    def __init__(self, indexer: EntityEmbeddingIndexer) -> None:
        self._indexer = indexer

    def __call__(self, context: Ch07Context) -> None:
        logger.info(f"Embedded {self._indexer.index()} entity summaries")


class GlobalSearchStep:
    def __init__(self, search: GlobalSearch) -> None:
        self._search = search

    def __call__(self, context: Ch07Context) -> None:
        logger.info(f"--- Global search: {context.global_question} ---")
        points = self._search.map(context.global_question)
        logger.info(f"{len(points)} key points from the community reports")
        context.global_answer = self._search.reduce(context.global_question, points)
        logger.info(f"Answer:\n{context.global_answer}")


class LocalSearchStep:
    def __init__(self, search: LocalSearch) -> None:
        self._search = search

    def __call__(self, context: Ch07Context) -> None:
        logger.info(f"--- Local search: {context.local_question} ---")
        local_context = self._search.build_context(context.local_question)
        logger.info(f"Context: {local_context.to_prompt()}")
        context.local_answer = self._search.answer(context.local_question, local_context)
        logger.info(f"Answer:\n{context.local_answer}")


def main(process_name: str, number_of_books: int) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logzero.logfile(LOG_DIR / f"{process_name}.log", encoding="utf-8")

    load_dotenv()
    settings = Settings.from_env()
    openai_client = OpenAI(api_key=settings.openai_api_key)
    llm = OpenAILLMClient(openai_client)
    embedder = OpenAIEmbedder(openai_client)

    # Kept apart from the other chapters' databases, and reset on every run:
    # re-importing would append duplicate descriptions, and Kuzu won't
    # rewrite the embedding column once the vector index covers it.
    db_path = settings.kuzu_db_path.with_name(f"{settings.kuzu_db_path.name}_ch07")
    with KuzuCypherGraphStore(db_path) as store:
        importer = EntityGraphImporter(store, embedder.dimensions)
        importer.create_schema(reset=True)

        pipeline: Pipeline[Ch07Context] = Pipeline(
            [
                LoadBooksStep(OdysseyCorpus(ODYSSEY_PATH), number_of_books),
                ChunkBooksStep(TextChunker(CHUNK_SIZE, CHUNK_OVERLAP)),
                ExtractGraphStep(
                    EntityRelationshipExtractor(llm, ENTITY_TYPES, model=DEFAULT_ANSWER_MODEL),
                    importer,
                    store,
                ),
                SummarizeDescriptionsStep(
                    DescriptionSummarizer(store, llm, model=DEFAULT_ANSWER_MODEL)
                ),
                DetectCommunitiesStep(KuzuLouvainCommunityDetector(store)),
                ReportCommunitiesStep(CommunityReporter(store, llm, model=DEFAULT_ANSWER_MODEL)),
                IndexEntitiesStep(EntityEmbeddingIndexer(store, embedder)),
                GlobalSearchStep(GlobalSearch(store, llm, model=DEFAULT_ANSWER_MODEL)),
                LocalSearchStep(LocalSearch(store, llm, embedder, model=DEFAULT_ANSWER_MODEL)),
            ]
        )
        pipeline.run(Ch07Context(global_question=GLOBAL_QUESTION, local_question=LOCAL_QUESTION))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--process-name",
        default="ch07_graphrag_odyssey",
        help="Names the log file written to logs/<process_name>.log.",
    )
    parser.add_argument(
        "--books",
        type=int,
        default=1,
        help="How many of the Odyssey's 24 books to build the graph from (the book uses 1).",
    )
    args = parser.parse_args()
    main(args.process_name, args.books)
