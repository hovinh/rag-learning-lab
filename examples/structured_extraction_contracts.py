"""Re-implementation of ch06 (template/kg-rag/notebooks/ch06.ipynb) with
the ``StructuredGraphExtraction`` recipe: extract a typed
Contract (parties, their locations and roles, dates, term, scope, amount,
governing law) from a licensing agreement with structured outputs, import
it into Kuzu as a small knowledge graph, then query the graph back to
show what landed.

The graph persists in its Kuzu database: the contract is extracted only
if none has been imported yet (or with --reindex, which starts from an
empty graph). See examples/README.md for the graph model and where it
departs from the book. Requires OPENAI_API_KEY in .env (see
.env.example). Run from the repo root:

    .venv/Scripts/python.exe examples/structured_extraction_contracts.py \\
        --process-name ch06_structured_extraction_contracts
"""

from __future__ import annotations

import argparse

from constants import DEFAULT_ANSWER_MODEL, LOG_DIR, add_reindex_argument
from dotenv import load_dotenv

from graphrag.domains.contracts import (
    LICENSE_AGREEMENT_PATH,
    ContractGraphImporter,
    build_contract_extractor,
)
from graphrag.graph import KuzuCypherGraphStore
from graphrag.ingestion import TextFileSource
from graphrag.observability import RunLogging, Trace
from graphrag.recipes import StructuredGraphExtraction
from graphrag.services import OpenAIServices
from graphrag.settings import Settings

trace = Trace(__name__)


def main(process_name: str, reindex: bool, verbose: bool) -> None:
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
        db_path = settings.db_path_for("contracts")

        with KuzuCypherGraphStore(db_path) as store:
            importer = ContractGraphImporter(store)
            importer.create_schema(reset=reindex)

            if not importer.contract_ids():
                extraction = StructuredGraphExtraction(
                    build_contract_extractor(llm, model=DEFAULT_ANSWER_MODEL),
                    importer.import_contract,
                )
                with trace.section("Extract the contract into the graph"):
                    extraction.index(TextFileSource(LICENSE_AGREEMENT_PATH))
            else:
                trace.field("graph", f"reusing {db_path} (--reindex rebuilds it)")

            # The book's ch06 notebook stops at the import; reading the
            # parties back shows the nodes and relationships actually landed.
            with trace.section("Query the parties back"):
                for contract_id in importer.contract_ids():
                    trace.table(contract_id, importer.parties(contract_id), max_width=60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    RunLogging.add_arguments(parser, default_process_name="ch06_structured_extraction_contracts")
    add_reindex_argument(parser)
    args = parser.parse_args()
    main(args.process_name, args.reindex, args.verbose)
