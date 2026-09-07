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
from pathlib import Path

import logzero
from dotenv import load_dotenv
from logzero import logger
from openai import OpenAI

from graphrag.graph import ChunkMatch, KuzuGraphStore
from graphrag.ingestion import PdfLoader, TextChunker
from graphrag.llm import LLMClient, Message, OpenAILLMClient
from graphrag.retrieval import OpenAIEmbedder
from graphrag.settings import Settings

PDF_URL = "https://arxiv.org/pdf/1709.00666.pdf"
PDF_PATH = Path("data/ch02-downloaded.pdf")
LOG_DIR = Path("logs")
QUESTION = "At what time was Einstein really interested in experimental works?"
SYSTEM_MESSAGE = (
    "You're en Einstein expert, but can only use the provided documents to "
    "respond to the questions."
)


def build_prompt(question: str, matches: list[ChunkMatch]) -> list[Message]:
    context = "\n".join(match.text for match in matches)
    user_message = (
        f"Use the following documents to answer the question that will follow:\n"
        f"{context}\n\n---\n\n"
        f"The question to answer using information only from the above documents: "
        f"{question}"
    )
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_message},
    ]


def log_matches(matches: list[ChunkMatch], label: str) -> None:
    """Log each match's full (untruncated) chunk text, for inspecting
    what was actually retrieved before it's sent to the LLM.
    """
    logger.info(f"--- {label} ---")
    for match in matches:
        logger.info(f"{match.score} {match.text}")


def ask(llm: LLMClient, question: str, matches: list[ChunkMatch]) -> None:
    logger.info(f"Question: {question}")
    deltas = []
    for delta in llm.stream(build_prompt(question, matches), model="gpt-4o"):
        print(delta, end="", flush=True)
        deltas.append(delta)
    print()
    logger.info(f"Answer: {''.join(deltas)}")


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

    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not PDF_PATH.exists():
        PdfLoader().download(PDF_URL, PDF_PATH)
    text = PdfLoader().extract_text(PDF_PATH)

    chunks = TextChunker(chunk_size=500, overlap=40).split(text)
    logger.info(f"{len(chunks)} chunks")

    embeddings = embedder.embed(chunks)

    with KuzuGraphStore(settings.kuzu_db_path, embedding_dim=embedder.dimensions) as store:
        store.replace_chunks(chunks, embeddings)

        question_embedding = embedder.embed([QUESTION])[0]

        vector_matches = store.vector_search(question_embedding, k=4)
        log_matches(vector_matches, "Vector search")
        ask(llm, QUESTION, vector_matches)

        hybrid_matches = store.hybrid_search(QUESTION, question_embedding, k=4)
        log_matches(hybrid_matches, "Hybrid search")
        ask(llm, QUESTION, hybrid_matches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--process-name",
        default="ch02_pdf_qa",
        help="Names the log file written to logs/<process_name>.log.",
    )
    args = parser.parse_args()
    main(args.process_name)
