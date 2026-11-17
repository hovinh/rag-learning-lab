"""Re-implementation of ch02 (template/kg-rag/notebooks/ch02.ipynb) with
the ``VectorRag`` recipe: download a PDF, chunk + embed it
into Kuzu, then answer a question with plain vector search and with
hybrid (vector + keyword) search.

The index persists in its Kuzu database, so only the first run (or one
with --reindex) downloads, chunks and embeds; later runs go straight to
the questions. See examples/README.md for how Kuzu's vector (HNSW) and FTS
(BM25) indexes work under the hood. Requires OPENAI_API_KEY in .env (see
.env.example). Run from the repo root:

    .venv/Scripts/python.exe examples/vector_similarity_and_hybrid_search.py \\
        --process-name ch02_pdf_qa
"""

from __future__ import annotations

import argparse
from pathlib import Path

from constants import DEFAULT_ANSWER_MODEL, LOG_DIR, PDF_URL, add_reindex_argument
from dotenv import load_dotenv

from graphrag.graph import KuzuGraphStore
from graphrag.ingestion import PdfSource, TextChunker
from graphrag.observability import RunLogging, Trace
from graphrag.recipes import VectorRag
from graphrag.services import OpenAIServices
from graphrag.settings import Settings

PDF_PATH = Path("data/ch02-downloaded.pdf")
QUESTION = "At what time was Einstein really interested in experimental works?"

trace = Trace(__name__)


def main(process_name: str, reindex: bool, verbose: bool) -> None:
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
        db_path = settings.db_path_for("ch02")

        with KuzuGraphStore(db_path, embedding_dim=services.embedder.dimensions) as store:
            rag = VectorRag(
                store,
                services.embedder,
                services.llm,
                chunker=TextChunker(chunk_size=500, overlap=40),
                k=4,
                model=DEFAULT_ANSWER_MODEL,
            )
            if reindex or not rag.is_indexed():
                with trace.section("Index the PDF into Kuzu"):
                    rag.index(PdfSource(PDF_URL, PDF_PATH))
            else:
                trace.field("index", f"reusing {db_path} (--reindex rebuilds it)")

            with trace.section("Answer with vector search"):
                rag.query(QUESTION, search="vector")
            with trace.section("Answer with hybrid search"):
                rag.query(QUESTION, search="hybrid")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    RunLogging.add_arguments(parser, default_process_name="ch02_pdf_qa")
    add_reindex_argument(parser)
    args = parser.parse_args()
    main(args.process_name, args.reindex, args.verbose)
