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
