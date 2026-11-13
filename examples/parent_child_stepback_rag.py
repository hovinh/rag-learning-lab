"""Re-implementation of ch03 (template/kg-rag/notebooks/ch03.ipynb):
split a PDF into titled sections, chunk each section into parent chunks and
each parent into child chunks, embed only the children, store the
parent/child hierarchy in Kuzu, then answer a question via step-back
prompting + parent-level retrieval.

See examples/README.md for how parent-child retrieval and step-back
prompting fit together. Requires OPENAI_API_KEY in .env (see
.env.example). Run from the repo root:

    .venv/Scripts/python.exe examples/parent_child_stepback_rag.py \\
        --process-name ch03_parent_child_stepback_rag
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

from graphrag.graph import KuzuParentChildGraphStore, ParentMatch
from graphrag.ingestion import PdfLoader, SectionSplitter, TextChunker
from graphrag.llm import OpenAILLMClient, QueryRewriter
from graphrag.pipeline import AnswerStep, LoadPdfTextStep, Pipeline
from graphrag.retrieval import Embedder, OpenAIEmbedder
from graphrag.settings import Settings

PDF_PATH = Path("data/ch03-downloaded.pdf")
QUESTION = "Who was Einstein's collaborator on sound reproduction system?"
PARENT_CHUNK_SIZE, PARENT_OVERLAP = 2000, 40
CHILD_CHUNK_SIZE, CHILD_OVERLAP = 500, 20


@dataclass
class Ch03Context:
    question: str
    text: str = ""
    sections: list[str] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)
    children_per_parent: list[list[str]] = field(default_factory=list)
    child_embeddings_per_parent: list[list[list[float]]] = field(default_factory=list)
    step_back_question: str = ""
    matches: list[ParentMatch] = field(default_factory=list)
    answer: str = ""


class SectionSplitStep:
    def __init__(self, splitter: SectionSplitter) -> None:
        self._splitter = splitter

    def __call__(self, context: Ch03Context) -> None:
        context.sections = self._splitter.split(context.text)
        logger.info(f"{len(context.sections)} sections")


class ParentChildChunkStep:
    def __init__(self, parent_chunker: TextChunker, child_chunker: TextChunker) -> None:
        self._parent_chunker = parent_chunker
        self._child_chunker = child_chunker

    def __call__(self, context: Ch03Context) -> None:
        parents: list[str] = []
        for section in context.sections:
            parents.extend(self._parent_chunker.split(section))
        context.parents = parents
        logger.info(f"{len(context.parents)} parent chunks")

        context.children_per_parent = [self._child_chunker.split(parent) for parent in parents]


class EmbedChildrenStep:
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def __call__(self, context: Ch03Context) -> None:
        context.child_embeddings_per_parent = [
            self._embedder.embed(children) for children in context.children_per_parent
        ]


class StepBackRewriteStep:
    def __init__(self, query_rewriter: QueryRewriter) -> None:
        self._query_rewriter = query_rewriter

    def __call__(self, context: Ch03Context) -> None:
        context.step_back_question = self._query_rewriter.step_back(context.question)
        logger.info(f"Step-back question: {context.step_back_question}")


class ParentRetrievalStep:
    def __init__(self, store: KuzuParentChildGraphStore, embedder: Embedder, k: int) -> None:
        self._store = store
        self._embedder = embedder
        self._k = k

    def __call__(self, context: Ch03Context) -> None:
        step_back_embedding = self._embedder.embed([context.step_back_question])[0]
        context.matches = self._store.parent_retrieval(step_back_embedding, k=self._k)


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
    query_rewriter = QueryRewriter(llm)

    context = Ch03Context(question=QUESTION)

    ingestion: Pipeline[Ch03Context] = Pipeline(
        [
            LoadPdfTextStep(PdfLoader(), PDF_URL, PDF_PATH),
            SectionSplitStep(SectionSplitter()),
            ParentChildChunkStep(
                parent_chunker=TextChunker(chunk_size=PARENT_CHUNK_SIZE, overlap=PARENT_OVERLAP),
                child_chunker=TextChunker(chunk_size=CHILD_CHUNK_SIZE, overlap=CHILD_OVERLAP),
            ),
            EmbedChildrenStep(embedder),
        ]
    )
    ingestion.run(context)

    # Kept apart from ch02's database so running one chapter doesn't wipe
    # the other's corpus (every replace_* call rebuilds its store).
    db_path = settings.kuzu_db_path.with_name(f"{settings.kuzu_db_path.name}_ch03")
    with KuzuParentChildGraphStore(db_path, embedding_dim=embedder.dimensions) as store:
        store.replace_parents(
            context.parents, context.children_per_parent, context.child_embeddings_per_parent
        )

        retrieval: Pipeline[Ch03Context] = Pipeline(
            [
                StepBackRewriteStep(query_rewriter),
                ParentRetrievalStep(store, embedder, k=4),
                AnswerStep(llm, model=DEFAULT_ANSWER_MODEL, label="Parent retrieval (step-back)"),
            ]
        )
        retrieval.run(context)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--process-name",
        default="ch03_parent_child_stepback_rag",
        help="Names the log file written to logs/<process_name>.log.",
    )
    args = parser.parse_args()
    main(args.process_name)
