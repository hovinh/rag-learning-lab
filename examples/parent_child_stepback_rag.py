"""Re-implementation of ch03 (template/kg-rag/notebooks/ch03.ipynb) with
the predefined ``ParentChildRag`` pipeline: split a PDF into titled
sections, chunk each section into parent chunks and each parent into
child chunks, embed only the children, store the parent/child hierarchy
in Kuzu, then answer a question via step-back prompting + parent-level
retrieval.

The index persists in its Kuzu database, so only the first run (or one
with --reindex) builds it. See examples/README.md for how parent-child
retrieval and step-back prompting fit together. Requires OPENAI_API_KEY in
.env (see .env.example). Run from the repo root:

    .venv/Scripts/python.exe examples/parent_child_stepback_rag.py \\
        --process-name ch03_parent_child_stepback_rag
"""

from __future__ import annotations

import argparse
from pathlib import Path

from constants import DEFAULT_ANSWER_MODEL, LOG_DIR, PDF_URL, add_reindex_argument
from dotenv import load_dotenv

from graphrag.graph import KuzuParentChildGraphStore
from graphrag.ingestion import PdfSource, SectionSplitter, TextChunker
from graphrag.llm import TokenCounter
from graphrag.observability import RunLogging, Trace
from graphrag.pipelines import ParentChildRag
from graphrag.services import OpenAIServices
from graphrag.settings import Settings

PDF_PATH = Path("data/ch03-downloaded.pdf")
QUESTION = "Who was Einstein's collaborator on sound reproduction system?"
PARENT_CHUNK_SIZE, PARENT_OVERLAP = 2000, 40
CHILD_CHUNK_SIZE, CHILD_OVERLAP = 500, 20

trace = Trace(__name__)


def main(process_name: str, reindex: bool, verbose: bool) -> None:
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
        db_path = settings.db_path_for("ch03")

        with KuzuParentChildGraphStore(
            db_path, embedding_dim=services.embedder.dimensions
        ) as store:
            rag = ParentChildRag(
                store,
                services.embedder,
                services.llm,
                parent_chunker=TextChunker(PARENT_CHUNK_SIZE, PARENT_OVERLAP),
                child_chunker=TextChunker(CHILD_CHUNK_SIZE, CHILD_OVERLAP),
                section_splitter=SectionSplitter(),
                token_counter=TokenCounter(),
                step_back=True,
                k=4,
                model=DEFAULT_ANSWER_MODEL,
            )
            if reindex or not rag.is_indexed():
                with trace.section("Index the PDF into Kuzu"):
                    rag.index(PdfSource(PDF_URL, PDF_PATH))
            else:
                trace.field("index", f"reusing {db_path} (--reindex rebuilds it)")

            with trace.section("Answer with step-back + parent retrieval"):
                rag.query(QUESTION)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    RunLogging.add_arguments(parser, default_process_name="ch03_parent_child_stepback_rag")
    add_reindex_argument(parser)
    args = parser.parse_args()
    main(args.process_name, args.reindex, args.verbose)
