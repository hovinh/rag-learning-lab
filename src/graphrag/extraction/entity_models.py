from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExtractedEntity:
    name: str
    type: str
    description: str


@dataclass(frozen=True)
class ExtractedRelationship:
    source: str
    target: str
    description: str
    # None when the LLM's strength isn't a number; the book keeps the raw
    # string in that case.
    strength: float | None


@dataclass
class ChunkExtraction:
    """The entities and relationships the LLM found in one chunk."""

    entities: list[ExtractedEntity] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)
