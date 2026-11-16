from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Generic, TypeVar

from graphrag.observability import Trace

TContext = TypeVar("TContext")

PipelineStep = Callable[[TContext], None]

_trace = Trace(__name__)


class Pipeline(Generic[TContext]):
    """Runs an ordered list of steps against a shared, mutable context.

    Not tied to any chapter's context shape -- each example defines its own
    context dataclass and step classes, and assembles them into a Pipeline.

    Each step runs in its own trace section, ``[2/5] Chunk books``, timed
    and holding whatever the step logs. The title is the step's ``title``
    attribute if it has one, else its class name without ``Step``.
    """

    def __init__(self, steps: Sequence[PipelineStep[TContext]]) -> None:
        self._steps = steps

    def run(self, context: TContext) -> TContext:
        for index, step in enumerate(self._steps, start=1):
            with _trace.section(f"[{index}/{len(self._steps)}] {step_title(step)}"):
                step(context)
        return context


def step_title(step: Callable[..., object]) -> str:
    """``ExtractGraphStep`` -> ``Extract graph``; a function
    ``load_books`` -> ``Load books``.
    """
    title = getattr(step, "title", None)
    if isinstance(title, str):
        return title
    name = getattr(step, "__name__", None) or type(step).__name__
    name = name.removesuffix("Step")
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name).replace("_", " ").split()
    text = " ".join(words).lower()
    return text[:1].upper() + text[1:]
