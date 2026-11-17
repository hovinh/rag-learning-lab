"""The contracts domain (ch06): a typed ``Contract`` model, the prompt
that extracts it, the graph it's imported into, and a sample agreement.
"""

from graphrag.domains.contracts.extractor import LICENSE_AGREEMENT_PATH, build_contract_extractor
from graphrag.domains.contracts.importer import ContractGraphImporter
from graphrag.domains.contracts.models import Contract, ContractType, Location, Organization

__all__ = [
    "LICENSE_AGREEMENT_PATH",
    "Contract",
    "ContractGraphImporter",
    "ContractType",
    "Location",
    "Organization",
    "build_contract_extractor",
]
