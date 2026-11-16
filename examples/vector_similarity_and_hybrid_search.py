"""Re-implementation of ch02 (template/kg-rag/notebooks/ch02.ipynb):
download a PDF, chunk + embed it, store it in Kuzu, then answer a question
with plain vector search and with hybrid (vector + keyword) search.

See examples/README.md for how Kuzu's vector (HNSW) and FTS (BM25) indexes
work under the hood. Requires OPENAI_API_KEY in .env (see .env.example).
Run from the repo root:

    .venv/Scripts/python.exe examples/vector_similarity_and_hybrid_search.py \\
        --process-name ch02_pdf_qa
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from constants import DEFAULT_ANSWER_MODEL, LOG_DIR, PDF_URL, add_verbose_argument
from dotenv import load_dotenv
from openai import OpenAI

from graphrag.graph import ChunkMatch, GraphStore, KuzuGraphStore
from graphrag.ingestion import PdfLoader, TextChunker
from graphrag.llm import OpenAILLMClient
from graphrag.observability import RunLogging, Trace
from graphrag.pipeline import AnswerStep, LoadPdfTextStep, Pipeline
from graphrag.retrieval import Embedder, OpenAIEmbedder
from graphrag.settings import Settings

PDF_PATH = Path("data/ch02-downloaded.pdf")
QUESTION = "At what time was Einstein really interested in experimental works?"

trace = Trace(__name__)


@dataclass
class Ch02Context:
    question: str
    text: str = ""
    chunks: list[str] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)
    matches: list[ChunkMatch] = field(default_factory=list)
    answer: str = ""


class ChunkStep:
    def __init__(self, chunker: TextChunker) -> None:
        self._chunker = chunker

    def __call__(self, context: Ch02Context) -> None:
        context.chunks = self._chunker.split(context.text)
        trace.field("chunks", len(context.chunks))


class EmbedChunksStep:
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def __call__(self, context: Ch02Context) -> None:
        context.embeddings = self._embedder.embed(context.chunks)
        trace.field("embedded", f"{len(context.embeddings)} chunks")


class VectorSearchStep:
    def __init__(self, store: GraphStore, question_embedding: list[float], k: int) -> None:
        self._store = store
        self._question_embedding = question_embedding
        self._k = k

    def __call__(self, context: Ch02Context) -> None:
        context.matches = self._store.vector_search(self._question_embedding, k=self._k)


class HybridSearchStep:
    title = "Hybrid search (vector + keyword)"

    def __init__(
        self, store: GraphStore, question: str, question_embedding: list[float], k: int
    ) -> None:
        self._store = store
        self._question = question
        self._question_embedding = question_embedding
        self._k = k

    def __call__(self, context: Ch02Context) -> None:
        context.matches = self._store.hybrid_search(
            self._question, self._question_embedding, k=self._k
        )


def main(process_name: str, verbose: bool) -> None:
    with RunLogging(
        process_name,
        LOG_DIR,
        title="ch02 · PDF ingestion, vector search and hybrid search",
        details={"model": DEFAULT_ANSWER_MODEL, "question": QUESTION},
        verbose=verbose,
    ) as run:
        load_dotenv()
        settings = Settings.from_env()
        openai_client = OpenAI(api_key=settings.openai_api_key)
        embedder = OpenAIEmbedder(openai_client, usage=run.usage)
        llm = OpenAILLMClient(openai_client, usage=run.usage)

        context = Ch02Context(question=QUESTION)

        with trace.section("Ingest the PDF"):
            ingestion: Pipeline[Ch02Context] = Pipeline(
                [
                    LoadPdfTextStep(PdfLoader(), PDF_URL, PDF_PATH),
                    ChunkStep(TextChunker(chunk_size=500, overlap=40)),
                    EmbedChunksStep(embedder),
                ]
            )
            ingestion.run(context)

        with KuzuGraphStore(settings.kuzu_db_path, embedding_dim=embedder.dimensions) as store:
            with trace.section("Store chunks in Kuzu"):
                store.replace_chunks(context.chunks, context.embeddings)

            question_embedding = embedder.embed([context.question])[0]

            with trace.section("Answer with vector search"):
                vector_pipeline: Pipeline[Ch02Context] = Pipeline(
                    [
                        VectorSearchStep(store, question_embedding, k=4),
                        AnswerStep(llm, model=DEFAULT_ANSWER_MODEL),
                    ]
                )
                vector_pipeline.run(context)

            with trace.section("Answer with hybrid search"):
                hybrid_pipeline: Pipeline[Ch02Context] = Pipeline(
                    [
                        HybridSearchStep(store, context.question, question_embedding, k=4),
                        AnswerStep(llm, model=DEFAULT_ANSWER_MODEL),
                    ]
                )
                hybrid_pipeline.run(context)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--process-name",
        default="ch02_pdf_qa",
        help="Names the log file written to logs/<process_name>.log.",
    )
    add_verbose_argument(parser)
    args = parser.parse_args()
    main(args.process_name, args.verbose)
