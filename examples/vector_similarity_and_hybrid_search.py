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

from constants import DEFAULT_ANSWER_MODEL, LOG_DIR, PDF_URL
from dotenv import load_dotenv

from graphrag.graph import KuzuGraphStore
from graphrag.ingestion import Chunk, PdfSource, TextChunker
from graphrag.observability import RunLogging, Trace
from graphrag.pipeline import (
    AnswerStep,
    ChunkStep,
    EmbedChunksStep,
    LoadDocumentsStep,
    Pipeline,
    RetrieveStep,
    StoreChunksStep,
)
from graphrag.retrieval import HybridRetriever, ScoredMatch, VectorRetriever
from graphrag.services import OpenAIServices
from graphrag.settings import Settings

PDF_PATH = Path("data/ch02-downloaded.pdf")
QUESTION = "At what time was Einstein really interested in experimental works?"

trace = Trace(__name__)


@dataclass
class Ch02Context:
    question: str
    documents: list[str] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)
    matches: list[ScoredMatch] = field(default_factory=list)
    answer: str = ""


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
        services = OpenAIServices.create(settings, usage=run.usage)
        embedder, llm = services.embedder, services.llm

        context = Ch02Context(question=QUESTION)

        with KuzuGraphStore(
            settings.db_path_for("ch02"), embedding_dim=embedder.dimensions
        ) as store:
            with trace.section("Ingest the PDF into Kuzu"):
                ingestion: Pipeline[Ch02Context] = Pipeline(
                    [
                        LoadDocumentsStep(PdfSource(PDF_URL, PDF_PATH)),
                        ChunkStep(TextChunker(chunk_size=500, overlap=40)),
                        EmbedChunksStep(embedder),
                        StoreChunksStep(store),
                    ]
                )
                ingestion.run(context)

            for label, retriever in [
                ("vector search", VectorRetriever(store, embedder, k=4)),
                ("hybrid search", HybridRetriever(store, embedder, k=4)),
            ]:
                with trace.section(f"Answer with {label}"):
                    answering: Pipeline[Ch02Context] = Pipeline(
                        [
                            RetrieveStep(retriever, label=label),
                            AnswerStep(llm, model=DEFAULT_ANSWER_MODEL),
                        ]
                    )
                    answering.run(context)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    RunLogging.add_arguments(parser, default_process_name="ch02_pdf_qa")
    args = parser.parse_args()
    main(args.process_name, args.verbose)
