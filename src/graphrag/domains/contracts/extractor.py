from __future__ import annotations

from pathlib import Path

from graphrag.domains.contracts.models import Contract
from graphrag.extraction.extractor import StructuredExtractor
from graphrag.llm.client import LLMClient, OpenAILLMClient
from graphrag.prompts import PromptRepository

_HERE = Path(__file__).parent

# The licensing and web site hosting agreement (an SEC filing exhibit) that
# the book's ch06 extracts a Contract from.
LICENSE_AGREEMENT_PATH = _HERE / "license_agreement.txt"

_EXTRACTION_PROMPT = "contract_extraction_system.txt"


def build_contract_extractor(
    llm: LLMClient, *, model: str = OpenAILLMClient.DEFAULT_MODEL
) -> StructuredExtractor[Contract]:
    """A ``StructuredExtractor`` for ``Contract``, with ch06's system
    prompt (kept next to the models it describes).
    """
    system_message = PromptRepository(_HERE).load(_EXTRACTION_PROMPT)
    return StructuredExtractor(llm, Contract, system_message, model=model)
