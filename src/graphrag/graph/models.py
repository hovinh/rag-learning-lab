from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkMatch:
    """A retrieved chunk. ``score`` is always higher-is-better, regardless
    of which search produced it (vector, keyword, or hybrid).
    """

    index: int
    text: str
    score: float


@dataclass(frozen=True)
class ParentMatch:
    """A retrieved parent chunk (ch03 parent-child retrieval). ``score`` is
    always higher-is-better, aggregated from the parent's matching children.
    """

    index: int
    text: str
    score: float


@dataclass(frozen=True)
class Property:
    name: str
    type: str


@dataclass(frozen=True)
class RelationshipPattern:
    """One ``(:start)-[:type]->(:end)`` connection allowed by the schema."""

    start: str
    type: str
    end: str


@dataclass(frozen=True)
class GraphSchema:
    """Node/relationship tables and how they connect (ch04), as read from
    the database -- what text2cypher shows the LLM so it only uses labels
    and properties that exist.
    """

    node_props: dict[str, list[Property]]
    rel_props: dict[str, list[Property]]
    relationships: list[RelationshipPattern]

    def to_prompt(self) -> str:
        """Same layout as the book's ``schema_utils.get_schema()``."""

        def format_props(props: list[Property]) -> str:
            return ", ".join(f"{prop.name}: {prop.type}" for prop in props)

        return "\n".join(
            [
                "Node properties:",
                *(f"{label} {{{format_props(props)}}}" for label, props in self.node_props.items()),
                "Relationship properties:",
                *(f"{rel} {{{format_props(props)}}}" for rel, props in self.rel_props.items()),
                "The relationships:",
                *(f"(:{r.start})-[:{r.type}]->(:{r.end})" for r in self.relationships),
            ]
        )
