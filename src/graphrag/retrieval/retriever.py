from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Protocol


class ScoredMatch(Protocol):
    """A retrieval match -- satisfied structurally by both ``ChunkMatch``
    and ``ParentMatch`` without either depending on this module. Declared
    as read-only properties (rather than plain attributes) so mypy checks
    them covariantly -- both dataclasses are frozen, so their fields are
    read-only in practice too.
    """

    @property
    def text(self) -> str: ...

    @property
    def score(self) -> float: ...


class Retriever(ABC):
    """Question in, scored context out: the one shape vector, hybrid and
    parent-child retrieval share, so a pipeline, an agent tool or a
    benchmark can take any of them. ``score`` is higher-is-better.
    """

    @abstractmethod
    def retrieve(self, question: str) -> Sequence[ScoredMatch]: ...
