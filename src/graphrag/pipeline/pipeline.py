from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Generic, TypeVar

TContext = TypeVar("TContext")

PipelineStep = Callable[[TContext], None]


class Pipeline(Generic[TContext]):
    """Runs an ordered list of steps against a shared, mutable context.

    Not tied to any chapter's context shape -- each example defines its own
    context dataclass and step classes, and assembles them into a Pipeline.
    """

    def __init__(self, steps: Sequence[PipelineStep[TContext]]) -> None:
        self._steps = steps

    def run(self, context: TContext) -> TContext:
        for step in self._steps:
            step(context)
        return context
