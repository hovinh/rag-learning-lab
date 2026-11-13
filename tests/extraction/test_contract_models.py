import pytest
from pydantic import ValidationError

from graphrag.extraction.contract_models import Contract, Location


def test_full_address_joins_all_parts():
    location = Location(address="2 Venture", city="Irvine", state="California", country="US")

    assert location.full_address() == "2 Venture Irvine California US"


def test_full_address_skips_missing_parts():
    location = Location(address=None, city="Irvine", state=None, country="US")

    assert location.full_address() == "Irvine US"


def test_nullable_fields_are_still_required():
    with pytest.raises(ValidationError, match="address"):
        Location.model_validate({"city": None, "state": None, "country": "US"})


def test_contract_type_must_be_one_of_the_listed_types():
    with pytest.raises(ValidationError, match="contract_type"):
        Contract.model_validate(
            {
                "contract_type": "Employment Agreement",
                "parties": [],
                "effective_date": "1999-02-26",
                "term": "1 year",
                "contract_scope": "x",
                "end_date": None,
                "total_amount": None,
                "governing_law": None,
            }
        )
