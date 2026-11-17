from __future__ import annotations

from typing import Any, ClassVar

from graphrag.extraction.contract_models import Contract, Location, Organization
from graphrag.graph.cypher_store import CypherGraphStore

_CREATE_CONTRACT = """
CREATE (c:Contract {
  id: gen_random_uuid(),
  contract_type: $contract_type,
  effective_date: $effective_date,
  term: $term,
  contract_scope: $contract_scope,
  end_date: $end_date,
  total_amount: $total_amount,
  governing_law: $governing_law
})
RETURN c.id AS id
"""

_MERGE_PARTY = """
MERGE (o:Organization {name: $name})
MERGE (l:Location {full_address: $full_address})
SET l.address = $address, l.city = $city, l.state = $state, l.country = $country
MERGE (o)-[:LOCATED_AT]->(l)
WITH o
MATCH (c:Contract {id: CAST($contract_id AS UUID)})
MERGE (o)-[r:HAS_PARTY]->(c)
SET r.role = $role
"""


_PARTIES = """
MATCH (o:Organization)-[r:HAS_PARTY]->(c:Contract)
WHERE c.id = CAST($contract_id AS UUID)
MATCH (o)-[:LOCATED_AT]->(l:Location)
RETURN o.name AS organization, r.role AS role, l.full_address AS location
ORDER BY organization
"""


class ContractGraphImporter:
    """Writes an extracted ``Contract`` into the graph (ch06's
    ``import_query``): ``(:Organization)-[:HAS_PARTY {role}]->(:Contract)``
    and ``(:Organization)-[:LOCATED_AT]->(:Location)``. Organizations and
    locations are merged on their keys, so contracts that share a party
    share its node; every import creates a new contract.

    Kuzu needs the tables declared before any data goes in -- they stand in
    for the book's uniqueness constraints, as each key is the table's
    primary key. The DDL is Kuzu's dialect, sent through the
    backend-agnostic ``CypherGraphStore``.
    """

    NODE_TABLES: ClassVar[dict[str, str]] = {
        "Contract": (
            "id UUID, contract_type STRING, effective_date STRING, term STRING, "
            "contract_scope STRING, end_date STRING, total_amount DOUBLE, "
            "governing_law STRING, PRIMARY KEY(id)"
        ),
        "Organization": "name STRING, PRIMARY KEY(name)",
        "Location": (
            "full_address STRING, address STRING, city STRING, state STRING, "
            "country STRING, PRIMARY KEY(full_address)"
        ),
    }
    REL_TABLES: ClassVar[dict[str, str]] = {
        "HAS_PARTY": "FROM Organization TO Contract, role STRING",
        "LOCATED_AT": "FROM Organization TO Location",
    }

    def __init__(self, store: CypherGraphStore) -> None:
        self._store = store

    def create_schema(self, *, reset: bool = False) -> None:
        """``reset`` drops the tables first, discarding every imported
        contract.
        """
        if reset:
            # Rel tables reference node tables, so they're dropped first and
            # created last.
            for table in [*self.REL_TABLES, *self.NODE_TABLES]:
                self._store.query(f"DROP TABLE IF EXISTS {table}")
        for table, columns in self.NODE_TABLES.items():
            self._store.query(f"CREATE NODE TABLE IF NOT EXISTS {table}({columns})")
        for table, columns in self.REL_TABLES.items():
            self._store.query(f"CREATE REL TABLE IF NOT EXISTS {table}({columns})")

    def import_contract(self, contract: Contract) -> str:
        """Returns the new contract node's id."""
        rows = self._store.query(
            _CREATE_CONTRACT,
            {
                "contract_type": contract.contract_type,
                "effective_date": contract.effective_date,
                "term": contract.term,
                "contract_scope": contract.contract_scope,
                "end_date": contract.end_date,
                "total_amount": contract.total_amount,
                "governing_law": _governing_law(contract.governing_law),
            },
        )
        contract_id = str(rows[0]["id"])
        for party in contract.parties:
            self._merge_party(party, contract_id)
        return contract_id

    def contract_ids(self) -> list[str]:
        """Every imported contract's id, so a caller can tell whether
        anything has been imported yet.
        """
        rows = self._store.query("MATCH (c:Contract) RETURN c.id AS id ORDER BY id")
        return [str(row["id"]) for row in rows]

    def parties(self, contract_id: str) -> list[dict[str, Any]]:
        """Reads a contract's parties back from the graph, with their role
        and address.
        """
        return self._store.query(_PARTIES, {"contract_id": contract_id})

    def _merge_party(self, party: Organization, contract_id: str) -> None:
        location = party.location
        self._store.query(
            _MERGE_PARTY,
            {
                "name": party.name,
                "full_address": location.full_address(),
                "address": location.address,
                "city": location.city,
                "state": location.state,
                "country": location.country,
                "contract_id": contract_id,
                "role": party.role,
            },
        )


def _governing_law(location: Location | None) -> str | None:
    """The book stores ``state + ' ' + country``; parts that are missing are
    skipped rather than nulling the whole value.
    """
    if location is None:
        return None
    return " ".join(part for part in [location.state, location.country] if part)
