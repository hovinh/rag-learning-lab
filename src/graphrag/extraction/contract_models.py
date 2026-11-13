from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# OpenAI's structured outputs require every field to be present, so optional
# values are modeled as required-but-nullable (``str | None`` with no
# default), as in the book.

ContractType = Literal[
    "Service Agreement",
    "Licensing Agreement",
    "Non-Disclosure Agreement (NDA)",
    "Partnership Agreement",
    "Lease Agreement",
]


class Location(BaseModel):
    """
    Represents a physical location including address, city, state, and country.
    """

    address: str | None = Field(..., description="The street address of the location.")
    city: str | None = Field(..., description="The city of the location.")
    state: str | None = Field(..., description="The state or region of the location.")
    country: str = Field(
        ...,
        description="The country of the location. Use the two-letter ISO standard.",
    )

    def full_address(self) -> str:
        """The parts that are present, space-separated. The book
        concatenates all four with Cypher's ``+``, which yields null as
        soon as one part is null.
        """
        parts = [self.address, self.city, self.state, self.country]
        return " ".join(part for part in parts if part)


class Organization(BaseModel):
    """
    Represents an organization, including its name and location.
    """

    name: str = Field(..., description="The name of the organization.")
    location: Location = Field(..., description="The primary location of the organization.")
    role: str = Field(
        ...,
        description=(
            "The role of the organization in the contract, such as 'provider', "
            "'client', 'supplier', etc."
        ),
    )


class Contract(BaseModel):
    """
    Represents the key details of the contract.
    """

    contract_type: ContractType = Field(..., description="The type of contract being entered into.")
    parties: list[Organization] = Field(
        ...,
        description=(
            "List of parties involved in the contract, with details of each party's role."
        ),
    )
    effective_date: str = Field(
        ...,
        description="The date when the contract becomes effective. Use yyyy-MM-dd format.",
    )
    term: str = Field(
        ...,
        description=(
            "The duration of the agreement, including provisions for renewal or termination."
        ),
    )
    contract_scope: str = Field(
        ...,
        description=(
            "Description of the scope of the contract, including rights, duties, "
            "and any limitations."
        ),
    )
    end_date: str | None = Field(
        ...,
        description="The date when the contract becomes expires. Use yyyy-MM-dd format.",
    )
    total_amount: float | None = Field(..., description="Total value of the contract.")
    governing_law: Location | None = Field(
        ..., description="The jurisdiction's laws governing the contract."
    )
