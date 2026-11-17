"""Re-implementation of ch06 (template/kg-rag/notebooks/ch06.ipynb):
extract a typed Contract (parties, their locations and roles, dates,
term, scope, amount, governing law) from a licensing agreement with
structured outputs, import it into Kuzu as a small knowledge graph, then
query the graph back to show what landed.

See examples/README.md for the graph model and where it departs from the
book. Requires OPENAI_API_KEY in .env (see .env.example). Run from the
repo root:

    .venv/Scripts/python.exe examples/structured_extraction_contracts.py \\
        --process-name ch06_structured_extraction_contracts
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from constants import DEFAULT_ANSWER_MODEL, LOG_DIR
from dotenv import load_dotenv

from graphrag.domains.contracts import (
    LICENSE_AGREEMENT_PATH,
    Contract,
    ContractGraphImporter,
    build_contract_extractor,
)
from graphrag.graph import KuzuCypherGraphStore
from graphrag.ingestion import TextFileSource
from graphrag.observability import RunLogging, Trace
from graphrag.pipeline import (
    ExtractStructuredStep,
    ImportExtractedStep,
    LoadDocumentsStep,
    Pipeline,
)
from graphrag.services import OpenAIServices
from graphrag.settings import Settings

trace = Trace(__name__)


@dataclass
class Ch06Context:
    documents: list[str] = field(default_factory=list)
    extracted: list[Contract] = field(default_factory=list)
    imported_ids: list[str] = field(default_factory=list)


def main(process_name: str, verbose: bool) -> None:
    with RunLogging(
        process_name,
        LOG_DIR,
        title="ch06 · Structured extraction of a contract into a graph",
        details={"model": DEFAULT_ANSWER_MODEL},
        verbose=verbose,
    ) as run:
        load_dotenv()
        settings = Settings.from_env()
        llm = OpenAIServices.create(settings, usage=run.usage).llm
        extractor = build_contract_extractor(llm, model=DEFAULT_ANSWER_MODEL)

        # Kept apart from the other chapters' databases, and reset on every run
        # so re-running doesn't pile up copies of the same contract.
        with KuzuCypherGraphStore(settings.db_path_for("ch06")) as store:
            importer = ContractGraphImporter(store)
            importer.create_schema(reset=True)

            pipeline: Pipeline[Ch06Context] = Pipeline(
                [
                    LoadDocumentsStep(TextFileSource(LICENSE_AGREEMENT_PATH)),
                    ExtractStructuredStep(extractor),
                    ImportExtractedStep(importer.import_contract),
                ]
            )
            context = pipeline.run(Ch06Context())

            # The book's ch06 notebook stops at the import; reading the
            # parties back shows the nodes and relationships actually landed.
            with trace.section("Query the parties back"):
                for contract_id in context.imported_ids:
                    trace.table("parties", importer.parties(contract_id), max_width=60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    RunLogging.add_arguments(parser, default_process_name="ch06_structured_extraction_contracts")
    args = parser.parse_args()
    main(args.process_name, args.verbose)
