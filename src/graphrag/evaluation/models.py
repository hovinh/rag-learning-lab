from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkCase:
    """One benchmark row: a question, and the Cypher whose result is the
    ground truth for it.
    """

    question: str
    cypher: str


@dataclass(frozen=True)
class EvaluationSample:
    """A benchmark question as the system under test handled it.

    ``ground_truth`` is the ground-truth Cypher's rows as JSON. ``error``
    is set (and ``answer``/``contexts`` empty) if answering raised; the
    book records ``None`` there and later scores it as "I don't know".
    """

    question: str
    ground_truth: str
    answer: str
    contexts: list[str]
    latency: float
    error: str | None = None
