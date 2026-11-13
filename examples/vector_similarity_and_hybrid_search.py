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
import io
import sys
from dataclasses import dataclass, field
from pathlib import Path

import logzero
from constants import DEFAULT_ANSWER_MODEL, LOG_DIR, PDF_URL
from dotenv import load_dotenv
from logzero import logger
from openai import OpenAI

from graphrag.graph import ChunkMatch, GraphStore, KuzuGraphStore
from graphrag.ingestion import PdfLoader, TextChunker
from graphrag.llm import OpenAILLMClient
from graphrag.pipeline import AnswerStep, LoadPdfTextStep, Pipeline
from graphrag.retrieval import Embedder, OpenAIEmbedder
from graphrag.settings import Settings

PDF_PATH = Path("data/ch02-downloaded.pdf")
QUESTION = "At what time was Einstein really interested in experimental works?"


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
        logger.info(f"{len(context.chunks)} chunks")


class EmbedChunksStep:
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def __call__(self, context: Ch02Context) -> None:
        context.embeddings = self._embedder.embed(context.chunks)


class VectorSearchStep:
    def __init__(self, store: GraphStore, question_embedding: list[float], k: int) -> None:
        self._store = store
        self._question_embedding = question_embedding
        self._k = k

    def __call__(self, context: Ch02Context) -> None:
        context.matches = self._store.vector_search(self._question_embedding, k=self._k)


class HybridSearchStep:
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


def main(process_name: str) -> None:
    # Windows defaults stdout/the log file to the cp1252 codepage, which
    # can't encode punctuation pdfplumber pulls out of PDFs (curly quotes,
    # U+2010 hyphens); force UTF-8 so those log lines don't get silently
    # dropped with a "Logging error" printed to the terminal.
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logzero.logfile(LOG_DIR / f"{process_name}.log", encoding="utf-8")

    load_dotenv()
    settings = Settings.from_env()
    openai_client = OpenAI(api_key=settings.openai_api_key)
    embedder = OpenAIEmbedder(openai_client)
    llm = OpenAILLMClient(openai_client)

    context = Ch02Context(question=QUESTION)

    ingestion: Pipeline[Ch02Context] = Pipeline(
        [
            LoadPdfTextStep(PdfLoader(), PDF_URL, PDF_PATH),
            ChunkStep(TextChunker(chunk_size=500, overlap=40)),
            EmbedChunksStep(embedder),
        ]
    )
    ingestion.run(context)

    with KuzuGraphStore(settings.kuzu_db_path, embedding_dim=embedder.dimensions) as store:
        store.replace_chunks(context.chunks, context.embeddings)

        question_embedding = embedder.embed([context.question])[0]

        vector_pipeline: Pipeline[Ch02Context] = Pipeline(
            [
                VectorSearchStep(store, question_embedding, k=4),
                AnswerStep(llm, model=DEFAULT_ANSWER_MODEL, label="Vector search"),
            ]
        )
        vector_pipeline.run(context)

        hybrid_pipeline: Pipeline[Ch02Context] = Pipeline(
            [
                HybridSearchStep(store, context.question, question_embedding, k=4),
                AnswerStep(llm, model=DEFAULT_ANSWER_MODEL, label="Hybrid search"),
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
    args = parser.parse_args()
    main(args.process_name)
