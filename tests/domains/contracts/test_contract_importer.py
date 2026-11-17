import pytest

from graphrag.domains.contracts.importer import ContractGraphImporter
from graphrag.domains.contracts.models import Contract, Location, Organization
from graphrag.graph.kuzu_cypher_store import KuzuCypherGraphStore

CLIENT = Organization(
    name="Mortgage Logic.com, Inc.",
    location=Location(address="2 Venture", city="Irvine", state="California", country="US"),
    role="Client",
)
PROVIDER = Organization(
    name="TrueLink, Inc.",
    location=Location(address=None, city="San Luis Obispo", state="California", country="US"),
    role="Provider",
)


def make_contract(parties, *, governing_law=None, total_amount=None):
    return Contract(
        contract_type="Licensing Agreement",
        parties=parties,
        effective_date="1999-02-26",
        term="One year, renewing automatically",
        contract_scope="License to use the Interface",
        end_date=None,
        total_amount=total_amount,
        governing_law=governing_law,
    )


@pytest.fixture
def store(tmp_path):
    with KuzuCypherGraphStore(tmp_path / "test_kuzu_db") as store:
        yield store


@pytest.fixture
def importer(store):
    importer = ContractGraphImporter(store)
    importer.create_schema()
    return importer


def test_import_contract_stores_contract_properties(store, importer):
    contract = make_contract(
        [CLIENT],
        governing_law=Location(address=None, city=None, state="California", country="US"),
        total_amount=1500.0,
    )

    contract_id = importer.import_contract(contract)

    rows = store.query(
        "MATCH (c:Contract) RETURN c.id AS id, c.contract_type AS type, "
        "c.total_amount AS amount, c.end_date AS end_date, c.governing_law AS law"
    )
    assert [{**row, "id": str(row["id"])} for row in rows] == [
        {
            "id": contract_id,
            "type": "Licensing Agreement",
            "amount": 1500.0,
            "end_date": None,
            "law": "California US",
        }
    ]


def test_import_contract_links_parties_with_roles_and_locations(store, importer):
    importer.import_contract(make_contract([CLIENT, PROVIDER]))

    rows = store.query(
        "MATCH (l:Location)<-[:LOCATED_AT]-(o:Organization)-[r:HAS_PARTY]->(:Contract) "
        "RETURN o.name AS name, r.role AS role, l.full_address AS address, l.address AS street "
        "ORDER BY name"
    )
    assert rows == [
        {
            "name": "Mortgage Logic.com, Inc.",
            "role": "Client",
            "address": "2 Venture Irvine California US",
            "street": "2 Venture",
        },
        {
            "name": "TrueLink, Inc.",
            "role": "Provider",
            "address": "San Luis Obispo California US",
            "street": None,
        },
    ]


def test_contracts_sharing_a_party_share_its_nodes(store, importer):
    importer.import_contract(make_contract([CLIENT, PROVIDER]))
    importer.import_contract(make_contract([CLIENT]))

    counts = store.query(
        "MATCH (c:Contract) WITH count(c) AS contracts "
        "MATCH (o:Organization) WITH contracts, count(o) AS orgs "
        "MATCH ()-[r:LOCATED_AT]->() RETURN contracts, orgs, count(r) AS located_at"
    )
    client_contracts = store.query(
        "MATCH (o:Organization {name: $name})-[:HAS_PARTY]->(c:Contract) RETURN count(c) AS n",
        {"name": CLIENT.name},
    )
    assert counts == [{"contracts": 2, "orgs": 2, "located_at": 2}]
    assert client_contracts == [{"n": 2}]


def test_create_schema_reset_discards_imported_contracts(store, importer):
    importer.import_contract(make_contract([CLIENT]))

    importer.create_schema(reset=True)

    assert store.query("MATCH (c:Contract) RETURN count(c) AS n") == [{"n": 0}]


def test_create_schema_is_idempotent_without_reset(store, importer):
    importer.import_contract(make_contract([CLIENT]))

    importer.create_schema()

    assert store.query("MATCH (c:Contract) RETURN count(c) AS n") == [{"n": 1}]


def test_contract_ids_lists_every_imported_contract(importer):
    assert importer.contract_ids() == []

    first = importer.import_contract(make_contract([CLIENT]))
    second = importer.import_contract(make_contract([PROVIDER]))

    assert sorted(importer.contract_ids()) == sorted([first, second])


def test_parties_reads_a_contracts_parties_back(importer):
    contract_id = importer.import_contract(make_contract([PROVIDER, CLIENT]))

    assert importer.parties(contract_id) == [
        {
            "organization": "Mortgage Logic.com, Inc.",
            "role": "Client",
            "location": CLIENT.location.full_address(),
        },
        {
            "organization": "TrueLink, Inc.",
            "role": "Provider",
            "location": PROVIDER.location.full_address(),
        },
    ]
