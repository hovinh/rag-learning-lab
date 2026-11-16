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
from pathlib import Path
from typing import Any

from constants import DEFAULT_ANSWER_MODEL, LOG_DIR, add_verbose_argument
from dotenv import load_dotenv
from openai import OpenAI

from graphrag.datasets import LICENSE_AGREEMENT_PATH
from graphrag.extraction import Contract, ContractGraphImporter, StructuredExtractor
from graphrag.graph import CypherGraphStore, KuzuCypherGraphStore
from graphrag.llm import OpenAILLMClient
from graphrag.observability import RunLogging, Trace
from graphrag.pipeline import Pipeline
from graphrag.prompts import PromptRepository
from graphrag.settings import Settings

CONTRACT_EXTRACTION_PROMPT = "contract_extraction_system.txt"

_PARTIES_QUERY = """
MATCH (o:Organization)-[r:HAS_PARTY]->(c:Contract)
WHERE c.id = CAST($contract_id AS UUID)
MATCH (o)-[:LOCATED_AT]->(l:Location)
RETURN o.name AS organization, r.role AS role, l.full_address AS location
ORDER BY organization
"""

trace = Trace(__name__)


@dataclass
class Ch06Context:
    document_path: Path
    document: str = ""
    contract: Contract | None = None
    contract_id: str = ""
    parties: list[dict[str, Any]] = field(default_factory=list)


class LoadDocumentStep:
    def __call__(self, context: Ch06Context) -> None:
        context.document = context.document_path.read_text(encoding="utf-8")
        trace.field("document", f"{context.document_path.name} ({len(context.document):,} chars)")


class ExtractContractStep:
    title = "Extract a Contract (structured outputs)"

    def __init__(self, extractor: StructuredExtractor[Contract]) -> None:
        self._extractor = extractor

    def __call__(self, context: Ch06Context) -> None:
        context.contract = self._extractor.extract(context.document)
        trace.block("contract", context.contract.model_dump(mode="json"))


class ImportContractStep:
    title = "Import into the graph"

    def __init__(self, importer: ContractGraphImporter) -> None:
        self._importer = importer

    def __call__(self, context: Ch06Context) -> None:
        if context.contract is None:
            raise ValueError("ImportContractStep needs an extracted contract")
        context.contract_id = self._importer.import_contract(context.contract)
        trace.field("contract", f"id {context.contract_id}")


class QueryPartiesStep:
    """Reads the imported contract back from the graph -- the book's ch06
    notebook stops at the import, but querying it shows the nodes and
    relationships actually landed.
    """

    title = "Query the parties back"

    def __init__(self, store: CypherGraphStore) -> None:
        self._store = store

    def __call__(self, context: Ch06Context) -> None:
        context.parties = self._store.query(_PARTIES_QUERY, {"contract_id": context.contract_id})
        trace.table("parties", context.parties, max_width=60)


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
        llm = OpenAILLMClient(OpenAI(api_key=settings.openai_api_key), usage=run.usage)
        extractor = StructuredExtractor(
            llm,
            Contract,
            PromptRepository().load(CONTRACT_EXTRACTION_PROMPT),
            model=DEFAULT_ANSWER_MODEL,
        )

        # Kept apart from the other chapters' databases, and reset on every run
        # so re-running doesn't pile up copies of the same contract.
        db_path = settings.kuzu_db_path.with_name(f"{settings.kuzu_db_path.name}_ch06")
        with KuzuCypherGraphStore(db_path) as store:
            importer = ContractGraphImporter(store)
            importer.create_schema(reset=True)

            pipeline: Pipeline[Ch06Context] = Pipeline(
                [
                    LoadDocumentStep(),
                    ExtractContractStep(extractor),
                    ImportContractStep(importer),
                    QueryPartiesStep(store),
                ]
            )
            pipeline.run(Ch06Context(document_path=LICENSE_AGREEMENT_PATH))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--process-name",
        default="ch06_structured_extraction_contracts",
        help="Names the log file written to logs/<process_name>.log.",
    )
    add_verbose_argument(parser)
    args = parser.parse_args()
    main(args.process_name, args.verbose)
