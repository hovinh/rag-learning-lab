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
from dataclasses import dataclass, field
from pathlib import Path

from constants import DEFAULT_ANSWER_MODEL, LOG_DIR, PDF_URL
from dotenv import load_dotenv

from graphrag.graph import KuzuParentChildGraphStore, ParentChildGraphStore, ParentMatch
from graphrag.ingestion import PdfSource, SectionSplitter, TextChunker
from graphrag.llm import QueryRewriter, TokenCounter
from graphrag.observability import RunLogging, Trace
from graphrag.pipeline import (
    AnswerStep,
    EmbedChildrenStep,
    LoadDocumentsStep,
    ParentChildChunkStep,
    Pipeline,
    SplitSectionsStep,
    StoreParentChildStep,
)
from graphrag.retrieval import Embedder
from graphrag.services import OpenAIServices
from graphrag.settings import Settings

PDF_PATH = Path("data/ch03-downloaded.pdf")
QUESTION = "Who was Einstein's collaborator on sound reproduction system?"
PARENT_CHUNK_SIZE, PARENT_OVERLAP = 2000, 40
CHILD_CHUNK_SIZE, CHILD_OVERLAP = 500, 20

trace = Trace(__name__)


@dataclass
class Ch03Context:
    question: str
    documents: list[str] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)
    children_per_parent: list[list[str]] = field(default_factory=list)
    child_embeddings_per_parent: list[list[list[float]]] = field(default_factory=list)
    step_back_question: str = ""
    matches: list[ParentMatch] = field(default_factory=list)
    answer: str = ""


class StepBackRewriteStep:
    title = "Step-back rewrite"

    def __init__(self, query_rewriter: QueryRewriter) -> None:
        self._query_rewriter = query_rewriter

    def __call__(self, context: Ch03Context) -> None:
        context.step_back_question = self._query_rewriter.step_back(context.question)
        trace.fields({"question": context.question, "step-back": context.step_back_question})


class ParentRetrievalStep:
    title = "Parent retrieval (match children, return parents)"

    def __init__(self, store: ParentChildGraphStore, embedder: Embedder, k: int) -> None:
        self._store = store
        self._embedder = embedder
        self._k = k

    def __call__(self, context: Ch03Context) -> None:
        step_back_embedding = self._embedder.embed([context.step_back_question])[0]
        context.matches = self._store.parent_retrieval(step_back_embedding, k=self._k)


def main(process_name: str, verbose: bool) -> None:
    with RunLogging(
        process_name,
        LOG_DIR,
        title="ch03 · Parent-child retrieval with step-back prompting",
        details={"model": DEFAULT_ANSWER_MODEL, "question": QUESTION},
        verbose=verbose,
    ) as run:
        load_dotenv()
        settings = Settings.from_env()
        services = OpenAIServices.create(settings, usage=run.usage)
        embedder, llm = services.embedder, services.llm

        context = Ch03Context(question=QUESTION)

        with KuzuParentChildGraphStore(
            settings.db_path_for("ch03"), embedding_dim=embedder.dimensions
        ) as store:
            with trace.section("Ingest the PDF into Kuzu"):
                ingestion: Pipeline[Ch03Context] = Pipeline(
                    [
                        LoadDocumentsStep(PdfSource(PDF_URL, PDF_PATH)),
                        SplitSectionsStep(SectionSplitter(), TokenCounter()),
                        ParentChildChunkStep(
                            parent_chunker=TextChunker(PARENT_CHUNK_SIZE, PARENT_OVERLAP),
                            child_chunker=TextChunker(CHILD_CHUNK_SIZE, CHILD_OVERLAP),
                        ),
                        EmbedChildrenStep(embedder),
                        StoreParentChildStep(store),
                    ]
                )
                ingestion.run(context)

            with trace.section("Answer with step-back + parent retrieval"):
                retrieval: Pipeline[Ch03Context] = Pipeline(
                    [
                        StepBackRewriteStep(QueryRewriter(llm)),
                        ParentRetrievalStep(store, embedder, k=4),
                        AnswerStep(llm, model=DEFAULT_ANSWER_MODEL),
                    ]
                )
                retrieval.run(context)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    RunLogging.add_arguments(parser, default_process_name="ch03_parent_child_stepback_rag")
    args = parser.parse_args()
    main(args.process_name, args.verbose)
