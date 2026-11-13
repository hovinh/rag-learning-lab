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
